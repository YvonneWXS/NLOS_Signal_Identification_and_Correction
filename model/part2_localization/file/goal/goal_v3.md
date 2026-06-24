Goal: Complete Module 2 — run Frankfurt retrained models, fix TCN 
degradation, and produce the final 6-method positioning table.

================================================================
PART 1: RUN FRANKFURT RETRAINED MODELS (P0, blocking)
================================================================

The config overrides for Frankfurt (exp_038/039) were implemented
but the models were never trained and evaluated. Do the following:

Step 1 — Verify config overrides are correctly applied:
  In GAT_V2025.py, confirm DATASET_OVERRIDES is applied when
  dataset name contains "frankfurt". Check that:
    - LAMBDA_ENTROPY = 0.005 (not 0.03)
    - SIGMA_NLOS_CLAMP_LOG_MAX = 3.5 (not 2.5)
    - SIGMA_GAP_TARGET = 1.0 (not 0.5)
  Print which overrides are active at training start.

Step 2 — Train exp_038 (frankfurt1) and exp_039 (frankfurt2):
  Use run_full_training.py --dataset frankfurt1_maintower
       run_full_training.py --dataset frankfurt2_westendtower
  with USE_MIXTURE_GAUSSIAN=True and the Frankfurt overrides.
  Save best_model.pth to result/exp_038/ and result/exp_039/.

Step 3 — Analyze with analyze_mog.py:
  Run analyze_mog.py for exp_038 and exp_039.
  Check: p_los gap > 0.55 and sigma_nlos(NLOS)/sigma_nlos(LOS) > 1.2
  If not met, print a WARNING but proceed anyway.

Step 4 — Update run_fusion.py model mapping:
  Change frankfurt1 → exp_038, frankfurt2 → exp_039.
  Re-run full Module 2 evaluation with new Frankfurt models.
  Report delta CEP50 vs exp_036/037 for WLS-MoG method.

================================================================
PART 2: FIX TCN 2A DEGRADATION (P1)
================================================================

TCN is currently making positioning worse (berlin2: +9% CEP50).
Two root causes to fix:

Fix A — Build FULL training sequences (not truncated):
  In fusion/train_tcn.py, build_sequences() currently uses only
  ~790 sequences. The actual dataset sizes are:
    berlin1:    1,377 epochs → 1,367 sequences (T=10 window)
    berlin2:    5,925 epochs → 5,915 sequences
    frankfurt1: 5,851 epochs → 5,841 sequences
    frankfurt2: 3,575 epochs → 3,565 sequences
  
  Remove any epoch limit / truncation. Build full caches.
  Retrain all 4 TCN models with full data:
    Epochs: 50 (up from current ~11)
    Batch size: 128 (up from current default)
    Early stopping patience: 10
  
  Expected val_loss targets: <0.48 for berlin2/frk1, <0.30 for frk2.
  Save updated models to fusion/models/tcn_{dataset}.pth.

Fix B — Tighten Bayesian prior update gate:
  In evaluate_fusion.py, the current condition for applying TCN prior
  uses |p_nlos - 0.5| > 0.15 (too loose). Change to:
  
    Only apply TCN prior update when ALL conditions are true:
      1. confidence > 0.65 (not 0.5)
      2. |p_nlos_prior - 0.5| > 0.25 (TCN is confident about direction)
      3. The TCN prediction DISAGREES with Module 1:
         i.e., (p_nlos_prior > 0.6 and p_los_gat < 0.5) OR
               (p_nlos_prior < 0.4 and p_los_gat > 0.5)
         Only update when TCN provides genuinely new information.
  
  This prevents TCN from "correcting" already-correct Module 1 outputs.

Fix C — Soft blending instead of hard Bayesian update:
  Replace the current hard Bayesian update with a soft blend:
    alpha = confidence * |p_nlos_prior - 0.5| * 2  # in [0, 1]
    alpha = min(alpha, 0.3)  # cap at 30% TCN influence
    p_los_updated = (1 - alpha) * p_los_gat + alpha * (1 - p_nlos_prior)
  
  This is more robust than the product-of-likelihoods approach,
  which can produce extreme values when both models are overconfident.

================================================================
PART 3: FINAL 6-METHOD EVALUATION TABLE (P1)
================================================================

After Part 1 and Part 2, run the complete evaluation on all 4 datasets
with all 6 methods. Save to fusion/result/exp_final/.

Required output format (print to console AND save to
fusion/result/exp_final/positioning_results.json):

=== CEP50 (meters) ===
Method               | berlin1 | berlin2 | frankfurt1 | frankfurt2
Standard LS          |         |         |            |
WLS-elevation        |         |         |            |
WLS-MoG              |         |         |            |
Hard-threshold       |         |         |            |
FactorGraph-MoG      |         |         |            |
FactorGraph-MoG+2A   |         |         |            |

=== FG-MoG vs WLS-MoG improvement (%) ===
(positive = FG better)

=== FG+2A vs FG improvement (%) ===
(positive = TCN prior helps)

=== Convergence stats (FactorGraph only) ===
% epochs IMPROVED | % epochs STABLE | % epochs DEGRADED
mean NLL improvement per epoch

After printing the table, add a diagnostic section:
  For each dataset where FG does NOT beat WLS-MoG:
    - Print mean p_los gap (LOS avg - NLOS avg)
    - Print mean sigma_nlos / sigma_los ratio
    - Print NLL range (min, mean, max across epochs)
    - Print diagnosis: "NLL surface flat" or "p_los undifferentiated"

Success criteria:
  - FactorGraph-MoG beats WLS-MoG in >= 2/4 datasets by >3% CEP50
  - FactorGraph+2A does NOT degrade vs FactorGraph in any dataset
    (if it does, fall back to FactorGraph-MoG for that dataset)
  - All 6 methods run without crash on all 4 datasets

================================================================
CONSTRAINTS
================================================================
- Do NOT retrain Module 1 BCE baseline models (exp_007-011)
- Do NOT change Module 1 GAT architecture
- Frankfurt retrain MUST use dataset overrides from config.py
- TCN training must use full epoch sequences (no truncation)
- If TCN still degrades after Fix B/C, report it clearly but do
  NOT include FactorGraph+2A as a recommended method in the table
  (mark it as "experimental")
- All results must be reproducible via: python run_fusion.py