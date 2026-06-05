Goal: Module 2 v8 — Fix mu_nlos magnitude collapse by replacing 
suppression-based direction loss with pure pairwise ranking loss,
restore magnitude anchor, retrain all 4 models, and produce 
final positioning results. This is the targeted final fix before 
Module 3 decision.

The v7 diagnostic conclusively identified the exact bug:
  - MuDirectionLoss.LOS_suppression (weight=2.0) pushed mu_LOS from
    248-465m down to 25-37m. This is CORRECT behavior.
  - BUT the same pressure also pushed mu_NLOS from 162-395m down to
    181-223m. This COLLAPSED the correction magnitude.
  - Root cause: suppressing LOS mu also weakens NLOS mu gradient path
    through the shared backbone. The model's easiest solution to 
    "mu_NLOS > mu_LOS" is to push both to near-zero with NLOS slightly 
    higher, not to push NLOS up.
  - Fix: use ONLY a pairwise ranking loss (no LOS suppression), and 
    restore LAMBDA_MU_REG to anchor NLOS magnitude around empirical values.

================================================================
PART 1: FIX MuDirectionLoss IN GAT_V2025.py
================================================================

File: part1_GAT/model/GAT_V2025.py

--- Remove LOS suppression, keep only pairwise ranking ---

Replace the existing MuDirectionLoss class with this version:

class MuDirectionLoss(nn.Module):
  """
  Pure pairwise ranking loss: enforce mu_nlos[NLOS] > mu_nlos[LOS].
  NO magnitude suppression — only the relative ordering is constrained.
  This preserves mu magnitude while fixing direction.
  """
  def forward(self, mu_nlos, labels):
    los_mask = (labels == 0)
    nlos_mask = (labels == 1)
    
    if not (los_mask.any() and nlos_mask.any()):
      return torch.tensor(0.0, device=mu_nlos.device)
    
    mu_los = mu_nlos[los_mask]
    mu_nlos_vals = mu_nlos[nlos_mask]
    
    # Ranking loss: mean(mu_NLOS) should exceed mean(mu_LOS) by margin
    # Use soft margin: hinge loss relu(margin - (mean_NLOS - mean_LOS))
    mu_los_mean = mu_los.mean()
    mu_nlos_mean = mu_nlos_vals.mean()
    
    margin = 0.15  # km = 150m — NLOS should exceed LOS by at least 150m
    ranking_loss = torch.relu(margin - (mu_nlos_mean - mu_los_mean))
    
    # Pairwise version: sample pairs and enforce ordering
    # This provides richer gradient signal than mean comparison
    n_pairs = min(len(mu_los), len(mu_nlos_vals), 32)
    if n_pairs > 0:
      idx_los = torch.randperm(len(mu_los), device=mu_nlos.device)[:n_pairs]
      idx_nlos = torch.randperm(len(mu_nlos_vals), device=mu_nlos.device)[:n_pairs]
      
      # Each NLOS sample should exceed each sampled LOS sample
      pairwise_margin = 0.10  # 100m per pair
      pair_loss = torch.relu(
          pairwise_margin - (mu_nlos_vals[idx_nlos] - mu_los[idx_los])
      ).mean()
      
      total_loss = 0.5 * ranking_loss + 0.5 * pair_loss
    else:
      total_loss = ranking_loss
    
    return total_loss

Key changes vs v7:
  - REMOVED: LOS suppression term (was penalizing mu_LOS > 0.05)
  - REMOVED: 2.0x weight on LOS suppression 
  - KEPT: mean ordering constraint (mu_NLOS_mean > mu_LOS_mean + margin)
  - ADDED: pairwise sampling for richer gradient signal
  - Weight reduced from 3.0 to 1.0 (applied in training loop, see below)

--- Update training loop loss weights ---

In train_epoch(), for the NLL stage:
  # The direction loss weight was 1.0 in v7 (too high with suppression)
  # Now that suppression is removed, 1.0 is appropriate for ranking only
  loss_mu_direction = mu_direction_loss_fn(mu_nlos, labels_batch)
  
  total_loss = (dynamic_bce_weight * loss_bce + 
                0.1 * loss_mog_nll + 
                0.5 * loss_mu_supervised +    # SupervisedMuRegressionLoss
                1.0 * loss_mu_direction)      # Pure ranking (no suppression)

In the Blend stage:
  loss_mu_direction = mu_direction_loss_fn(mu_nlos, labels_batch)
  total_loss = blend_loss + 0.5 * loss_mu_direction  # same as v7

--- Update config.py ---

Change ONLY these values vs v7:
  LAMBDA_MU_REG = 0.20       # Restore from 0.05 (v7) — re-anchor magnitude
                              # 0.20 is between v5's 0.50 (too strong) and 
                              # v7's 0.05 (too weak)
  LAMBDA_MU_DIRECTION = 1.0  # Keep same as v7
  MU_NLOS_TARGET = 0.30      # km — anchor NLOS mu around empirical 166-236m
                              # was 0.50 in v5 (too high), 0.50 in v7
                              # 0.30 km = 300m, between empirical range

  # Remove MU_DIRECTION_LOS_TARGET = 0.05 (no longer needed, no suppression)

--- Expected outcome ---
  mu_LOS: should land ~80-120m (less suppressed than v7's 25-37m,
          more suppressed than v5's 248-465m)
  mu_NLOS: should land ~250-350m (up from v7's 181-223m, near empirical)
  Direction margin: mu_NLOS - mu_LOS > 150m (target)

================================================================
PART 2: QUICK VERIFICATION ON BERLIN1 BEFORE FULL TRAINING
================================================================

Before training all 4 models (which takes ~3.5 hours), run a 
20-epoch quick check on berlin1 only:

  python run_full_training.py --dataset berlin1_potsdamer_platz \
    --exp-name exp_048_quick --num-epochs 20

After 20 epochs, run analyze_mog.py on exp_048_quick and check:
  1. Is mu_nlos[NLOS] > mu_nlos[LOS]? (direction check)
  2. Is mu_nlos[NLOS] > 0.15 km? (magnitude check — not collapsed)
  3. Is mu_nlos[LOS] < 0.20 km? (not over-suppressed either)

Print verdict:
  If ALL 3 checks pass: proceed to full training (exp_048-051)
  If direction FAILS: increase LAMBDA_MU_DIRECTION to 1.5, re-verify
  If magnitude < 0.15 km: increase MU_NLOS_TARGET to 0.40, re-verify
  If LOS > 0.20 km: add light LOS penalty 
      loss_los_light = 0.3 * torch.relu(mu_los.mean() - 0.15)
      (much lighter than v7's 2.0 suppression)

Document the quick check result before proceeding.

================================================================
PART 3: FULL TRAINING exp_048-051
================================================================

After quick check passes, train all 4 models:
  python run_full_training.py --dataset berlin1_potsdamer_platz  → exp_048
  python run_full_training.py --dataset berlin2_gendarmenmarkt   → exp_049
  python run_full_training.py --dataset frankfurt1_maintower     → exp_050
  python run_full_training.py --dataset frankfurt2_westendtower  → exp_051

After EACH training, immediately run analyze_mog.py and save 
to result/exp_04X/mu_direction_check.json with:
  {
    "mu_los_mean_km": float,
    "mu_nlos_mean_km": float,  
    "direction_correct": bool,
    "direction_margin_km": float,
    "mu_magnitude_ok": bool,      // nlos_mean > 0.15 km
    "f1": float,
    "p_los_gap": float,
    "sigma_nlos_ratio": float     // sigma_nlos[NLOS] / sigma_nlos[LOS]
  }

STOP if F1 < 0.78 for any dataset. If this happens:
  Print: "CLASSIFICATION DEGRADED: reduce LAMBDA_MU_DIRECTION to 0.5"
  Do NOT proceed with Module 2 evaluation.

Expected targets:
  mu_NLOS mean: 0.20-0.40 km (was 0.18-0.22 in v7, target higher)
  mu_LOS mean: 0.05-0.15 km (was 0.03-0.04 in v7, slightly higher ok)
  direction margin > 0.12 km in all 4 datasets
  F1 >= 0.82 for all datasets

================================================================
PART 4: DELETE STALE CACHES AND REBUILD
================================================================

After training exp_048-051, delete old inference caches:
  Delete: fusion/cache/berlin1_potsdamer_platz_mog_outputs.pkl
  Delete: fusion/cache/berlin2_gendarmenmarkt_mog_outputs.pkl  
  Delete: fusion/cache/frankfurt1_maintower_mog_outputs.pkl
  Delete: fusion/cache/frankfurt2_westendtower_mog_outputs.pkl

Update run_fusion.py model mapping:
  berlin1 → exp_048
  berlin2 → exp_049
  frankfurt1 → exp_050
  frankfurt2 → exp_051

The caches will be rebuilt automatically on next evaluation run.

================================================================
PART 5: FOCUSED EVALUATION — KEY METHODS ONLY
================================================================

File: fusion/evaluate_fusion.py

Do NOT run all 20 methods (too slow, most are known failures).
Run only the 8 most informative methods:

  1.  Standard LS              (baseline, no Module 1)
  2.  WLS-MoG-linear           (v3 standard, uses p_los/sigma)
  3.  WLS-debiased             (v4, uses mu_nlos — should benefit from fix)
  4.  Debiased-WLS-v2          (v7, cleaner debiased version)
  5.  PRNC-mu-corrected        (v7, with direction safety gate)
  6.  PRNC-adaptive            (v5, does not use mu — control case)
  7.  FG-debiased              (factor graph with debiasing)
  8.  FG-MoG+2A                (v6 best method, test if v8 maintains it)

For each method × dataset, report:
  CEP50 (m), delta vs Standard LS (%), delta vs v7 same method (%)

Additionally compute for debiasing methods (3, 4, 5):
  mean_correction_applied_km: mean(p_nlos × mu_nlos) across all sats
  Expected: >0.10 km (was near 0.05 in v5 when mu was wrong direction)
  This metric confirms the debiasing is now meaningful

Save to fusion/result/exp_v8/positioning_results_v8.json

================================================================
PART 6: mu_nlos DEBIASING EFFECTIVENESS ANALYSIS
================================================================

File: fusion/verify_debiasing_effectiveness.py (NEW)

After evaluation, run this analysis to verify the debiasing chain works:

For each dataset:
  Step 1: Compute mean pseudorange error per satellite by type
    LOS satellites: mean(pr_mes - dist - clk_standard)
    NLOS satellites: mean(pr_mes - dist - clk_standard)
    
  Step 2: Compute mean mu_nlos by satellite type (v8 model)
    mu_predicted_LOS: mean(mu_nlos[p_los > 0.7])
    mu_predicted_NLOS: mean(mu_nlos[p_los < 0.3])
    
  Step 3: Compute correction quality
    For NLOS sats: correction = (1-p_los) × mu_nlos
    ideal_correction = mean NLOS pseudorange error (positive)
    correction_accuracy = 1 - |correction - ideal| / ideal
    
  Step 4: Compute residual after debiasing
    pr_debiased = pr_mes - (1-p_los) × mu_nlos
    new_residual = pr_debiased - dist - clk
    residual_reduction = |original_residual| - |new_residual| (positive = better)
    
Print per-dataset summary:
  "Dataset X: NLOS mean error = Y m, mean correction = Z m, 
   residual reduction = W m (X% improvement)"

This directly verifies whether the debiasing is working end-to-end.
Save to fusion/result/exp_v8/debiasing_analysis.json

================================================================
PART 7: FINAL SUCCESS CRITERIA AND MODULE 3 DECISION
================================================================

File: fusion/generate_final_report_v8.py (NEW)

Print the following report at the end of run_fusion.py:

=== Module 2 v8 Final Results ===

--- mu_nlos Direction Status ---
Dataset    | mu_LOS (m) | mu_NLOS (m) | Margin (m) | Direction
berlin1    |            |             |            |
berlin2    |            |             |            |
frankfurt1 |            |             |            |
frankfurt2 |            |             |            |
Target: mu_NLOS > mu_LOS + 120m in ALL 4 datasets

--- Positioning Results vs Standard LS ---
Method              | berlin1 | berlin2 | frankfurt1 | frankfurt2 | Count > Std LS
WLS-MoG-linear      |         |         |            |            |
WLS-debiased        |         |         |            |            |
Debiased-WLS-v2     |         |         |            |            |
FG-MoG+2A           |         |         |            |            |

--- v6 vs v8 Comparison (key method: FG-MoG+2A) ---
frankfurt1 v6: 445.7m (+15.1%) | v8: ??? 
berlin2 v6: 778.8m (-27.5%) | v8: ???

=== Module 3 Readiness Decision ===

Compute: count_methods_beating_std_ls = number of (method, dataset) 
  combinations where CEP50 < Standard LS × 0.97 (> 3% improvement)

If count_methods_beating_std_ls >= 4:  // at least 4 combinations
  print "=== MODULE 2 COMPLETE: PROCEED TO MODULE 3 ==="
  print "Best performing method: {best_method}"
  print "Datasets with improvement: {list}"
  
Elif count_methods_beating_std_ls >= 1 OR frankfurt1_fg_mog_2a < 500:
  print "=== MODULE 2 PARTIAL SUCCESS: PROCEED TO MODULE 3 ==="
  print "Frankfurt1 FG-MoG+2A demonstrates soft information value"
  print "Module 3 residual feedback expected to generalize this result"
  print "Accepting current Module 2 as sufficient for research contribution"
  
Else:
  print "=== MODULE 2 STILL FAILING: mu_magnitude_check ==="
  print "mu_NLOS means: {values}"
  print "If mu_NLOS < 0.20 km, magnitude is still collapsed"
  print "Increase MU_NLOS_TARGET to 0.50 and retrain"

Save full report to fusion/result/exp_v8/FINAL_REPORT_v8.md including:
  - Version history table (v3-v8 best CEP50 per dataset)
  - Key scientific findings (5 bullet points)
  - mu_nlos evolution across versions
  - Recommendation for Module 3

================================================================
PART 8: PREPARE MODULE 1 OUTPUTS FOR MODULE 3
================================================================

File: fusion/export_for_module3.py (NEW)

If Module 3 decision is PROCEED, export the final Module 1+2 outputs
in a format ready for Module 3 (Residual Feedback):

For each dataset, using exp_048-051 models, compute and save:
fusion/module3_inputs/{dataset}_module3_ready.pkl with:
  {
    'epoch_positions': np.array (N, 3) ECEF km,  // Module 2 best method result
    'epoch_errors': np.array (N,),  // 2D error vs GT in meters
    'per_sat_outputs': [  // per epoch
      {
        'p_los': np.array (n_sats,),
        'mu_nlos': np.array (n_sats,),  
        'sigma_los': np.array (n_sats,),
        'sigma_nlos': np.array (n_sats,),
        'p_los_gap': float,  // epoch-level metric
        'gt_ecef': np.array (3,),
      }
    ],
    'metadata': {
      'dataset': str,
      'module1_model': str,  // exp_048/049/050/051
      'module2_best_method': str,
      'module2_cep50': float,
      'standard_ls_cep50': float,
      'improvement_pct': float
    }
  }

This file will be the primary input for Module 3.

================================================================
IMPLEMENTATION ORDER (STRICTLY FOLLOW)
================================================================

Step 1: Modify MuDirectionLoss in GAT_V2025.py (Part 1)
        Update config.py values (Part 1)
        Time: ~30 min

Step 2: Quick 20-epoch check on berlin1 (Part 2)
        Run analyze_mog.py, verify direction + magnitude
        Time: ~8 min
        GATE: if checks fail, fix and retry before proceeding

Step 3: Full training exp_048-051 (Part 3)
        Run all 4 datasets sequentially
        Time: ~3.5 hours (using run_full_training.py × 4)
        After each: run analyze_mog.py and save direction_check.json

Step 4: Delete stale caches (Part 4)
        Update run_fusion.py mapping
        Time: ~5 min

Step 5: Run 8-method evaluation (Part 5)
        Time: ~25 min
        Save to exp_v8/

Step 6: Run debiasing effectiveness analysis (Part 6)
        Time: ~5 min

Step 7: Generate final report and Module 3 decision (Part 7)
        Time: ~5 min

Step 8: If decision = PROCEED, export for Module 3 (Part 8)
        Time: ~15 min

Total estimated time: ~4.5 hours

================================================================
CONSTRAINTS
================================================================
- Do NOT change GAT architecture, 11-dim features, block-diagonal batching
- Do NOT use the old LOS suppression term (mu_LOS < 0.05 penalty)
  Only pairwise ranking is allowed in MuDirectionLoss
- F1 floor: if any dataset drops below 0.78, reduce LAMBDA_MU_DIRECTION
  to 0.5 before proceeding
- LAMBDA_MU_REG MUST be set to 0.20 (not 0.05 from v7, not 0.30 from v5)
  This is the precise fix identified from the v7 failure analysis
- All 8 evaluation methods must run on all 4 datasets without crash
- The quick check (Part 2) is MANDATORY before full training
  Do not skip it even if confident the fix is correct
- Keep all existing methods in evaluate_fusion.py (just don't run them 
  all by default — add a flag --full to run all 20 methods)
- Log mu_nlos[LOS] and mu_nlos[NLOS] means at every 10th training epoch
  to track convergence of the direction fix during training
- If Step 3 quick check passes but full training (Step 3) produces 
  wrong direction for any dataset, that dataset gets 1 extra training 
  run with LAMBDA_MU_DIRECTION = 1.5 before giving up