Goal: Final documentation cleanup — update README with v4 numbers,
reconcile the Module 2 vs Module 3 FG evaluation discrepancy in paper
table, add Frankfurt2 degradation explanation, and produce one clean
final evaluation script that reproduces all paper numbers.

This is a documentation-only pass. NO code changes to any algorithm.
NO new model training. Expected time: ~30 minutes.

================================================================
TASK 1: UPDATE README.md WITH v4 FINAL NUMBERS
================================================================

File: part3_ResidualFeedbackAndOnline_Correction/model/README.md

The README currently shows v3 results (exp_004) in the "Final Results"
table. Update to show v4 final numbers (exp_006):

Replace the v3 results table:
  OLD: berlin1 899.7m (+0.5%), berlin2 592.8m (+3.0%), 
       frankfurt1 521.9m (+0.6%), frankfurt2 373.8m (+2.3%)
  
  NEW: berlin1 872.8m (+3.5%), berlin2 598.5m (+2.0%),
       frankfurt1 467.4m (+11.0%), frankfurt2 368.0m (+3.8%)

Update success criteria table to show all 5/5 PASS including C4.

Update the "Known Limitations" section — remove C4 as a known
limitation (it now passes). Keep the other 3 limitations.

Update the version header from "v3 (2026-06-06)" to 
"v4 (2026-06-07) — ALL 5/5 success criteria PASS. FINAL VERSION."

================================================================
TASK 2: RECONCILE Module 2 vs Module 3 FG VALUES IN PAPER TABLE
================================================================

File: result/exp_006/paper_table_v4.md

The paper table shows two different numbers for Module 2 FG-MoG+2A
frankfurt1:
  - Module 2 v8 standalone: 476.9m (+9.2%)
  - Module 3 internal FG evaluation: 596.9m (-13.7%)

These are the SAME algorithm but produce different results because:
  Module 2 uses exp_038-039 models (Frankfurt P0 retrain, Frankfurt config overrides)
  Module 3 uses exp_050 model (v8 mu_nlos direction fix)
  Different models → different p_los outputs → different WLS/FG positioning

Add a footnote to paper_table_v4.md explaining this:

  "* Module 2 FG-MoG+2A (frankfurt1 = 476.9m) uses exp_038 model with 
   Frankfurt-specific training configuration (LAMBDA_ENTROPY=0.005, 
   SIGMA_GAP_TARGET=1.0). Module 3 internal FG evaluation (596.9m) 
   uses exp_050 (v8 universal training). The difference demonstrates 
   that dataset-specific Module 1 tuning improves downstream positioning.
   Module 3 Adaptive-M3 (467.4m) outperforms both."

This explains the discrepancy without hiding it.
Keep the Module 2 v8 value (476.9m) in the paper table as it represents
the best achievable with static Module 2 for fair comparison.

================================================================
TASK 3: EXPLAIN FRANKFURT2 ONLINE LEARNING DEGRADATION
================================================================

File: result/exp_006/key_findings.md

In the "Online Learning is Scene-Dependent" section, add:

  "Frankfurt2 shows apparent -490.7% degradation in the 'first vs last
  100 epoch' metric, but epoch-bin diagnosis (exp_004/frankfurt2_diagnosis)
  revealed this is NOT progressive degradation — it is caused by a small
  number of high-error outlier bins in late epochs (bin 3382-3560: StdLS=1021m)
  that dominate the last-100-epoch average. The safety fallback mechanism
  (1.05x relative threshold) prevents per-epoch CEP50 from exceeding
  Standard LS at the individual epoch level. The -490.7% metric is an
  artifact of epoch-binning sensitivity, not a genuine learning failure."

This gives reviewers the honest explanation for what looks like
a significant limitation.

================================================================
TASK 4: CREATE REPRODUCIBLE FINAL EVALUATION SCRIPT
================================================================

File: model/reproduce_paper_results.py (NEW)

Create a single script that reproduces all numbers in the paper table.
This is standard practice for reproducible research.

"""
reproduce_paper_results.py
Reproduces all numbers in paper_table_v4.md and ablation_report.md.
Run: python reproduce_paper_results.py
Expected time: ~2 minutes
"""

import os
import json

def main():
  print("=" * 60)
  print("Reproducing Paper Results (Urban GNSS NLOS PI-PEM)")
  print("=" * 60)
  
  # Step 1: Verify all required models exist
  required_models = {
    'exp_048': 'berlin1_potsdamer_platz',
    'exp_049': 'berlin2_gendarmenmarkt',  
    'exp_050': 'frankfurt1_maintower',
    'exp_051': 'frankfurt2_westendtower',
  }
  
  for exp, dataset in required_models.items():
    model_path = f'../../part1_GAT/result/{exp}/best_model.pth'
    assert os.path.exists(model_path), f"Missing model: {model_path}"
    print(f"  [OK] Module 1 model: {exp} ({dataset})")
  
  # Step 2: Run Standard LS baseline (Table Row 1)
  print("\nRunning Standard LS baseline...")
  # [call existing evaluate functions]
  
  # Step 3: Run Module 2 FG-MoG+2A static (Table Row 2)
  # Note: uses Module 2 exp_038-039 for Frankfurt for best comparison
  print("\nRunning Module 2 static evaluation...")
  
  # Step 4: Run Module 3 Adaptive-M3 v4 (Table Row 3)
  print("\nRunning Module 3 Adaptive-M3 v4...")
  # USE_POSTERIOR_CORRECTION=False, USE_TCN=False
  
  # Step 5: Run ablation configs A-G (ablation table)
  print("\nRunning ablation study...")
  
  # Step 6: Print final table
  print("\n" + "=" * 60)
  print("PAPER TABLE (Cross-Module CEP50 Comparison)")
  print("=" * 60)
  print(f"{'Method':<30} | {'Berlin1':>8} | {'Berlin2':>8} | {'Frankfurt1':>10} | {'Frankfurt2':>10}")
  print("-" * 75)
  # print rows with actual computed values
  
  # Step 7: Verify against stored exp_006 results
  with open('result/exp_006/berlin1/metrics.json') as f:
    stored = json.load(f)
  computed_cep50 = ...  # from above computation
  assert abs(computed_cep50 - stored['adaptive_cep50']) < 5, \
    f"Reproducibility check FAILED: {computed_cep50} vs {stored['adaptive_cep50']}"
  print("\n[PASS] All results match stored exp_006 values within 5m tolerance")

if __name__ == '__main__':
  main()

Implement the actual computation inside main() by calling the
existing functions from run_module3.py and evaluate_module3.py.
The script should print the exact numbers from paper_table_v4.md
and produce a summary matching FINAL_RESEARCH_SUMMARY.md.

================================================================
TASK 5: UPDATE MAIN PROJECT README
================================================================

If there is a top-level README.md for the entire project
(at the root of NLOS Signal Identification and Correction/),
update it with the final results and module status:

Module 1: COMPLETE (exp_048-051, F1 0.84-0.91)
Module 2: COMPLETE (DOP inflation analysis, frankfurt1 +9.2%)
Module 3: COMPLETE (Adaptive-M3 v4, all 4 datasets +2.0% to +11.0%)

Status: RESEARCH COMPLETE — ready for paper submission

If no top-level README exists, create one with this summary.

================================================================
CONSTRAINTS
================================================================
- NO algorithm changes whatsoever
- NO model retraining
- All existing result files preserved (add, don't overwrite)
- reproduce_paper_results.py must run to completion without error
- The Frankfurt2 explanation must be factual and honest
  (do NOT claim the algorithm fixes the degradation — it doesn't,
   the safety fallback handles individual epochs but the
   first-vs-last metric is genuinely dominated by outlier bins)
- The Module 2 vs Module 3 FG discrepancy must be explained,
  not hidden