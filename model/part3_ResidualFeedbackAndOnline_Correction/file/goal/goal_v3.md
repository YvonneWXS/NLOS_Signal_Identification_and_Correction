Goal: Module 3 v3 — Final polish: fix C4 regression without breaking C3,
fix TCN architecture mismatch via key remapping, and produce definitive
ablation study and final paper outputs.

This is the last optimization pass. All core components work correctly.
Target: 5/6 success criteria PASS (C1-C5 all pass, bonus TCN optional).

================================================================
PART 0: DIAGNOSE WHY C4 REGRESSED FROM v1 TO v2
================================================================

Before making any changes, run a diagnostic to understand the
exact mechanism of the C4 regression.

In run_module3.py, add temporary verbose output for frankfurt1:
  For each epoch in frankfurt1:
    If epoch < 50 or epoch % 100 == 0:
      Print: epoch, quality_class, score, method_selected, 
             fg_threshold, actual_2d_error_fg, actual_2d_error_ls

Then compare epoch-level behavior between v1 and v2 configs:
  v1 config: window=20, min_history=5, fg_threshold=0.70, gap_threshold=0.55
  v2 config: window=50, min_history=15, fg_threshold=0.75, gap_threshold=0.45

The v2 regression hypothesis: with window=50 and min_history=15,
the tracker needs 15 epochs of history before ANY HIGH_QUALITY 
classification can happen. During those first 15 epochs, all 
selections are LOW (Standard LS). This delay reduces the fraction
of FG-selected epochs. With only 2.5% FG usage in frankfurt1,
the improvement is diluted.

Expected finding: v1 used FG in the BEST epochs (high p_los gap moments),
v2 misses many of those due to the min_history delay.

Save diagnosis output to result/exp_003/diagnosis_frankfurt1.txt

================================================================
PART 1: FIX C4 — DECOUPLE WINDOW SIZE FROM QUALITY DETECTION DELAY
================================================================

File: model/residual_feedback.py

The fix: allow early HIGH_QUALITY classification based on CURRENT
epoch features (p_los gap, DOP ratio) even before the innovation
window fills up. The window is used for ADJUSTING thresholds, not
for BLOCKING classification.

Change ResidualInnovationTracker.get_scene_quality():

  def get_scene_quality(self):
    if len(self.innovation_history) < self.min_history:
      # Not enough history for tracker-based assessment
      # Return UNCERTAIN (not LOW) so detector can still classify
      return 'UNCERTAIN', 0.5
    
    # ... existing logic ...

And change AdaptivePosCorrector.process_epoch() to handle UNCERTAIN:

  quality, score, features = self.detector.classify_epoch(...)
  tracker_quality, tracker_conf = self.tracker.get_scene_quality()
  
  # Combine detector and tracker signals
  if tracker_quality == 'UNCERTAIN':
    # Use detector alone (epoch features) when no history yet
    final_quality = quality
    final_score = score * 0.8  # slight confidence reduction
  elif tracker_quality == 'HIGH_QUALITY' and quality == 'HIGH':
    # Both agree: high confidence
    final_quality = 'HIGH'
    final_score = min(0.95, score + 0.1)
  elif tracker_quality == 'LOW_QUALITY' and quality == 'LOW':
    # Both agree: definitely low
    final_quality = 'LOW'
    final_score = 0.1
  else:
    # Disagreement: use more conservative
    final_quality = 'LOW' if tracker_quality == 'LOW_QUALITY' else quality
    final_score = score * 0.7

This allows FG to be used in early epochs based on current features
(p_los gap + DOP ratio) without waiting for window to fill.

Also for frankfurt1 specifically, slightly relax the score threshold
back toward v1 values while keeping the larger window:
  In DATASET_CONFIGS for 'frankfurt1_maintower':
    fg_threshold: 0.75 → 0.68  (was 0.70 in v1, split the difference)
    window_size: 50 (keep)
    min_history: 15 (keep, but UNCERTAIN now allows early use)

Expected outcome: frankfurt1 FG usage increases from 2.5% back toward
5-8%, while C3 online learning remains positive due to window=50.
Target: frankfurt1 Adaptive-M3 CEP50 ≤ 490m.

================================================================
PART 2: FIX TCN ARCHITECTURE MISMATCH VIA KEY REMAPPING
================================================================

File: model/residual_feedback.py — update make_fg_tcn_solver()

The saved TCN state_dict has keys:
  input_proj.weight, input_proj.bias
  conv1.weight, conv1.bias
  conv2.weight, conv2.bias  
  conv3.weight, conv3.bias
  out.weight, out.bias

The current MotionGeometryPredictor expects keys like:
  tcn_layers.0.conv.weight, input_proj.0.weight, etc.

Solution: write a key remapper that loads the old architecture:

def load_tcn_with_key_remapping(model_path, device='cpu'):
  """
  Load TCN model handling architecture mismatch between old 
  (3-layer flat keys) and new (4-layer Sequential keys) versions.
  """
  state_dict = torch.load(model_path, map_location=device)
  
  # Check architecture version from keys
  keys = list(state_dict.keys())
  
  if 'conv1.weight' in keys:
    # Old architecture: TCNPriorPredictor with flat naming
    # Build a compatible simple TCN on the fly
    
    class SimpleTCN_v1(nn.Module):
      """Matches the old 3-layer architecture exactly."""
      def __init__(self, input_dim=63, hidden_dim=64, output_dim=20):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, 
                               padding=2, dilation=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 3, 
                               padding=4, dilation=4)
        self.out = nn.Linear(hidden_dim, output_dim)
      
      def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = torch.relu(self.input_proj(x))  # (B, T, H)
        x = x.transpose(1, 2)  # (B, H, T)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.transpose(1, 2)  # (B, T, H)
        x = torch.sigmoid(self.out(x[:, -1, :]))  # last timestep
        return x  # (B, output_dim)
    
    # Check actual dimensions from weights
    input_dim = state_dict['input_proj.weight'].shape[1]
    hidden_dim = state_dict['input_proj.weight'].shape[0]
    output_dim = state_dict['out.weight'].shape[0]
    
    model = SimpleTCN_v1(input_dim, hidden_dim, output_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model
  
  else:
    # New architecture: use MotionGeometryPredictor directly
    from fusion.motion_geometry_predictor import MotionGeometryPredictor
    model = MotionGeometryPredictor()
    model.model.load_state_dict(state_dict)
    model.eval()
    return model

Update make_fg_tcn_solver() to call load_tcn_with_key_remapping()
instead of directly loading MotionGeometryPredictor.

After loading, verify TCN forward pass works:
  test_input = torch.zeros(1, 10, input_dim)
  test_output = model(test_input)
  assert test_output.shape == (1, 20), f"TCN output shape wrong: {test_output.shape}"
  print(f"TCN loaded successfully: {model_path}, input_dim={input_dim}")

================================================================
PART 3: INVESTIGATE FRANKFURT2 LATE-EPOCH DEGRADATION
================================================================

File: model/run_module3.py — add epoch-level analysis for frankfurt2

The -490.7% online learning in frankfurt2 (first 100: 132m → last 100: 781m)
is the most severe remaining issue. It can't be fixed without understanding
the root cause.

Add diagnostic analysis for frankfurt2:

Step 1 — Plot error vs epoch (save as text histogram):
  Divide frankfurt2 epochs into 20 bins of ~178 epochs each.
  For each bin, compute:
    mean Standard LS error
    mean Adaptive-M3 error
    fraction of FG selections
    
  Print as ASCII table:
    Epoch range | StdLS CEP | Adaptive CEP | FG% | Scene quality
    0-178       |           |              |     |
    179-357     |           |              |     |
    ...
    3397-3575   |           |              |     |

Step 2 — Identify transition point:
  Find the first epoch where Adaptive-M3 > Standard LS significantly
  (error ratio > 1.2). Print epoch index and surrounding context.

Step 3 — Check if it's a data issue:
  For last 200 epochs of frankfurt2:
    Print mean p_los gap, mean sigma_nlos/sigma_los ratio
    Compare to first 200 epochs
    
  If p_los gap drops significantly in last 200 epochs → 
    Module 1 output quality degraded (data distribution shift)
    Module 3 cannot fix a Module 1 failure
    
  If p_los gap stays similar → 
    Scene geometry changed, FG weighting becomes harmful
    Should be caught by CUSUM but isn't

Step 4 — Fix if recoverable:
  If root cause is data distribution shift:
    Add an additional criterion to DATASET_CONFIGS for frankfurt2:
      Require that mean p_los gap > 0.4 (from last 10 epochs) 
      before allowing HIGH_QUALITY classification
      This prevents FG usage when Module 1 becomes uncertain
  
  If root cause is geometry change:
    Increase CUSUM sensitivity for frankfurt2:
      allowance: 20m → 10m (detect earlier)
      threshold: 100m → 50m (trigger override sooner)
    In DATASET_CONFIGS['frankfurt2_westendtower']:
      Add: cusum_allowance = 10.0, cusum_threshold = 50.0

Save full diagnosis to result/exp_003/frankfurt2_diagnosis.txt

================================================================
PART 4: ABLATION STUDY (PAPER CONTRIBUTION)
================================================================

File: model/run_ablation.py (create or update)

Run 6 configurations on all 4 datasets:

Config A: Static Standard LS (no Module 1, no Module 3)
Config B: Static WLS-MoG (Module 1 weights, no adaptation)
Config C: Static FG-MoG+2A (Module 1 FG, no adaptation)
Config D: Adaptive selection only (no CUSUM, no posterior, no TCN)
Config E: Adaptive + CUSUM (no posterior, no TCN) 
Config F: Full Adaptive-M3 v3 (all components)
Config G: Full + TCN (if TCN loads successfully)

Report marginal contribution of each component:
  "CUSUM contribution: D→E delta CEP50 per dataset"
  "Posterior correction: E→F delta CEP50"
  "TCN: F→G delta CEP50"

Format as table for paper:
  Component Added  | berlin1 | berlin2 | frankfurt1 | frankfurt2
  Adaptive select  | +X%     | +X%     | +X%        | +X%
  + CUSUM          | +X%     | +X%     | +X%        | +X%
  + Posterior      | +X%     | +X%     | +X%        | +X%
  + TCN (if avail) | +X%     | +X%     | +X%        | +X%

Save to result/exp_003/ablation_table.md and ablation_table.tex

================================================================
PART 5: FINAL COMPREHENSIVE EVALUATION (exp_003)
================================================================

Run full evaluation with all v3 fixes.

Expected behavior:
  frankfurt1 FG usage: 2.5% → 5-8% (due to early classification fix)
  frankfurt1 CEP50: 520.2m → ~490m (recovering v1 level while keeping C3)
  TCN now loads successfully with key remapping
  frankfurt2 degradation: understood and either fixed or documented

Report table (ECEF-consistent metric throughout):

=== v3 Final CEP50 Table ===
Method              | berlin1 | berlin2 | frankfurt1 | frankfurt2
Standard LS         | 904.5   | 610.8   | 525.2      | 382.6
WLS-MoG v8         |         |         |            |
FG-MoG+2A v8       |         |         |            |
Adaptive-M3 v3      |         |         |            |
Adaptive-M3+TCN v3  |         |         |            |

=== Success Criteria ===
C1: Adaptive ≤ LS ALL 4: [PASS/FAIL]
C2: Adaptive ≤ best static ≥3/4: [PASS/FAIL]
C3: Online learning ≥2/4: [PASS/FAIL]
C4: frankfurt1 ≤ 490m: [PASS/FAIL]
C5: CUSUM functional: [PASS/FAIL]
Bonus: TCN loads and works: [PASS/FAIL]

Target: 5/6 PASS (C1-C5 all pass)
Save to result/exp_003/FINAL_RESULTS_v3.md

================================================================
PART 6: UPDATE PAPER TABLE AND GENERATE FINAL OUTPUTS
================================================================

File: model/generate_cross_module_table.py (update)

Update paper_table.md with v3 results.
Add ablation table.
Generate final summary narrative:

The research progression from Module 1 to Module 3:
  Module 1: GAT with MoG outputs — F1 0.84-0.91, p_los gap 0.52-0.68
  Module 2: Static fusion — only frankfurt1 benefits (+9.2%), 
            3/4 datasets WORSE than Standard LS due to DOP inflation
  Module 3: Adaptive feedback — ALL 4 datasets improved (+0.5% to +4.1%)
            Key insight: residual feedback learns which scenes allow 
            non-uniform satellite weighting without DOP penalty

Key scientific contributions (print as numbered list):
  1. mu_nlos direction inversion discovered and fixed via pairwise ranking loss
  2. DOP inflation identified as primary failure mode for urban WLS
  3. Residual innovation tracking enables scene-adaptive method selection
  4. Online threshold adaptation generalizes frankfurt1 result to all datasets
  5. Safety fallback guarantee ensures never worse than Standard LS

Save final cross-module comparison as:
  result/exp_003/paper_table_final.md
  result/exp_003/paper_table_final.tex
  result/exp_003/key_findings.md

================================================================
IMPLEMENTATION ORDER
================================================================

Step 1: Run frankfurt1 + frankfurt2 diagnostics (Part 0, Part 3)
  Time: ~15 min
  Read output before making code changes

Step 2: Implement early classification fix (Part 1)
  Change UNCERTAIN behavior in get_scene_quality()
  Adjust frankfurt1 fg_threshold to 0.68
  Test on frankfurt1 only: does FG% increase? Does CEP50 drop?

Step 3: Implement TCN key remapping (Part 2)
  Test: does TCN load for all 4 datasets?
  Test: does FG+TCN actually differ from FG in any dataset?

Step 4: Apply frankfurt2 fix based on diagnosis (Part 3)
  Either p_los gap gate or tighter CUSUM thresholds

Step 5: Run ablation study (Part 4) — ~20 min

Step 6: Run full exp_003 evaluation (Part 5) — ~5 min

Step 7: Update paper outputs (Part 6) — ~10 min

Total estimated time: ~1.5 hours

================================================================
SUCCESS CRITERIA FOR v3
================================================================

[REQUIRED C1]: Adaptive ≤ Standard LS in ALL 4 datasets
[REQUIRED C2]: Adaptive ≤ best static in ≥3/4 datasets
[TARGET C3]: Online learning ≥2/4 (already achieved in v2, must maintain)
[TARGET C4]: frankfurt1 ≤ 490m (regressed in v2, fix in v3)
[REQUIRED C5]: CUSUM functional and integrated
[BONUS]: TCN loads and FG+TCN improves ≥1/4 dataset vs FG alone

If 5/6 pass (C1-C5 all pass): print "MODULE 3 COMPLETE — RESEARCH DONE"
If 4/6 pass with C4 close miss (<510m): print "MODULE 3 ACCEPTED — PROCEED"

================================================================
CONSTRAINTS
================================================================
- Do NOT re-run Module 1 training (use exp_048-051 as-is)
- Do NOT change Module 2 algorithm files
- The TCN key remapping must be backward-compatible:
  if old keys not found, fall back to new architecture loading
- ECEF xy-plane error metric from v2 must remain unchanged
  All new results must use this metric for consistency
- frankfurt2 diagnosis MUST come before any parameter changes
  for that dataset — don't tune blindly
- ablation study uses same ECEF metric as main results
- All experiments saved to result/exp_003/ (not overwriting exp_002)
- If C4 cannot be fixed without breaking C3 after 2 attempts:
  accept C4 miss and print clear explanation in key_findings.md
  The C4/C3 tradeoff is itself a scientific finding worth documenting