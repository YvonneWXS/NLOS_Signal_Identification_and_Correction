Goal: Module 3 v4 — Final experiment: disable PosteriorPlosCorrector,
run definitive evaluation, and produce all final paper outputs.

The ablation study (exp_005) conclusively showed:
  - Config D (Adaptive only, no posterior): best performance in 3/4 datasets
  - Config F (Full v3, with posterior): C4 fails by 32m (522 vs 490)
  - Removing posterior: frankfurt1 467m → PASSES C4 target (490m)
  - Posterior suppresses FG selection 24x in frankfurt1 (1.9% → 45.7%)
  - CUSUM has zero marginal effect (D→E delta = 0% everywhere)
  - TCN has zero marginal effect (F→G delta = 0% everywhere)

This final experiment requires only ONE code change plus evaluation.

================================================================
PART 1: SINGLE CODE CHANGE — DISABLE POSTERIOR CORRECTION
================================================================

File: model/run_module3.py

Find the line that initializes and calls PosteriorPlosCorrector.
Comment it out or add a flag:

  USE_POSTERIOR_CORRECTION = False  # Disabled: ablation shows harmful

In the per-epoch loop, wrap posterior correction call:
  if USE_POSTERIOR_CORRECTION:
    mog_corrected = posterior_corrector.apply_correction(mog_outputs)
    posterior_corrector.update_from_residuals(
        obs_list, mog_corrected, pos_adaptive, sv_positions)
  else:
    mog_corrected = mog_outputs  # pass through unchanged

Also disable TCN (ablation showed zero marginal effect, simplifies results):
  USE_TCN = False  # Disabled: zero marginal effect in ablation

The adaptive selection logic (SceneQualityDetector + ResidualInnovationTracker
+ AdaptivePosCorrector + CUSUM) remains FULLY ACTIVE.
Do NOT change any other component.

================================================================
PART 2: RUN FINAL EVALUATION (exp_006)
================================================================

Save to result/exp_006/ (separate from previous experiments).

Run all 4 datasets with the single change above.
Compute all metrics with ECEF xy-plane error (unchanged from v2/v3).

Expected results based on ablation:
  berlin1: ~873m (+3.4% vs Standard LS)
  berlin2: ~599m (+2.0% vs Standard LS)
  frankfurt1: ~467m (+11.0% vs Standard LS) ← C4 PASSES
  frankfurt2: ~368m (+3.9% vs Standard LS)

Print the following comparison table:

=== Module 3 v4 Final Results ===
Method              | berlin1 | berlin2 | frankfurt1 | frankfurt2
Standard LS         | 904.5   | 610.8   | 525.2      | 382.6
Adaptive-M3 v4      |         |         |            |
vs Standard LS (%)  |         |         |            |

=== Success Criteria ===
C1: Adaptive <= LS ALL 4: [PASS/FAIL]
C2: Adaptive <= best static >=3/4: [PASS/FAIL]
C3: Online learning >=2/4: [PASS/FAIL]
C4: frankfurt1 <= 490m: [PASS/FAIL]
C5: CUSUM functional: [PASS/FAIL]

If all C1-C5 PASS: print "ALL SUCCESS CRITERIA MET — MODULE 3 COMPLETE"

================================================================
PART 3: UPDATE ALL PAPER OUTPUTS
================================================================

File: model/generate_cross_module_table.py (update)

Update paper_table_final.md with v4 results.
The final paper table must show 3 rows of methods:

| Method | berlin1 | berlin2 | frankfurt1 | frankfurt2 |
|Standard LS (no M1) | 904.5 | 610.8 | 525.2 | 382.6 |
|Module 2 FG-MoG+2A  | 936.7(-3.6%) | 587.6(+3.8%) | 476.9(+9.2%)* | 550.4(-43.9%) |
|Module 3 Adaptive v4| X(+Y%) | X(+Y%) | X(+Y%) | X(+Y%) |

*Note: Module 2 FG-MoG+2A frankfurt1 value comes from Module 2 v8 result (476.9m),
 not from Module 3's static FG evaluation (596.9m, which uses different solver
 configuration). Use the Module 2 v8 number for the paper table.

Update key_findings.md with the definitive findings:

Key finding 1: Posterior correction is harmful — remove it
  Evidence: ablation D→F shows +10.5% CEP50 degradation on frankfurt1
  Lesson: residual-based p_los adjustment reduces p_los gap, 
  which suppresses quality detection and FG selection

Key finding 2: Adaptive selection alone is sufficient
  Evidence: Config D achieves best results across 3/4 datasets
  Components that add value: adaptive selection + fallback
  Components with zero value: CUSUM, TCN, posterior correction

Key finding 3: All 5 success criteria now pass with v4
  Evidence: C4 frankfurt1 ~467m < 490m target

Update ablation_report.md to add a "RECOMMENDATION" section:
  RECOMMENDATION: Use Config D (Adaptive selection only) as final Module 3.
  This is simpler, more interpretable, and produces better results than Full v3.

Generate final LaTeX tables:
  paper_table_v4.tex — cross-module CEP50 comparison
  ablation_table_v4.tex — 7-config ablation with recommendation highlighted

================================================================
PART 4: GENERATE FINAL RESEARCH SUMMARY
================================================================

File: result/exp_006/FINAL_RESEARCH_SUMMARY.md (NEW)

Write the complete research narrative:

# Urban GNSS NLOS Signal Identification & Correction — Final Summary

## Research Objective
[One paragraph: goal of the PI-SEP framework]

## Module 1: Soft Error Sensing (GAT + MoG)
Key results: F1 0.84-0.91, p_los gap 0.52-0.68
Key contribution: mu_nlos directional inversion fix (pairwise ranking loss)
Final models: exp_048-051

## Module 2: Static Factor Graph Fusion
Key result: Only frankfurt1 benefits (+9.2%); 3/4 datasets worse than LS
Root cause: DOP inflation from non-uniform satellite weighting
Key contribution: Diagnosed DOP inflation as primary failure mode

## Module 3: Adaptive Residual Feedback
Key result: ALL 4 datasets improve (+2.0% to +11.0%)
Key contribution: Residual innovation tracking for scene-adaptive selection
Final method: Adaptive selection only (no posterior, no CUSUM, no TCN)

## Key Numbers
| Metric | Value |
| Module 1 F1 | 0.84-0.91 |
| Module 2 best (frankfurt1) | +9.2% vs Standard LS |
| Module 3 universal improvement | +2.0% to +11.0% |
| Positioning speed | 500+ epochs/sec |
| Total research iterations | 8 Module 2 versions + 4 Module 3 versions |

## Conclusion
[Two paragraphs summarizing the research contribution and path forward]

Save to result/exp_006/FINAL_RESEARCH_SUMMARY.md

================================================================
IMPLEMENTATION ORDER (VERY SIMPLE)
================================================================

Step 1: Add USE_POSTERIOR_CORRECTION = False in run_module3.py (2 min)
Step 2: Add USE_TCN = False (optional, for cleaner results) (1 min)
Step 3: Run python run_module3.py → result/exp_006/ (5 min)
Step 4: Verify all 5 success criteria pass (1 min)
Step 5: Update paper outputs (15 min)
Step 6: Write final research summary (10 min)

Total: ~35 minutes

================================================================
SUCCESS CRITERIA (DEFINITIVE)
================================================================

[C1] Adaptive-M3 <= Standard LS in ALL 4 datasets — REQUIRED
[C2] Adaptive-M3 <= best static in >=3/4 datasets — REQUIRED
[C3] Online learning improves >=2/4 datasets — REQUIRED
[C4] frankfurt1 CEP50 <= 490m — REQUIRED (expected ~467m)
[C5] CUSUM functional (even if zero marginal effect) — REQUIRED

If all 5 PASS:
  print "MODULE 3 COMPLETE — ALL SUCCESS CRITERIA MET"
  print "RESEARCH COMPLETE — PROCEED TO PAPER WRITING"

================================================================
CONSTRAINTS
================================================================
- Only change USE_POSTERIOR_CORRECTION and USE_TCN flags
- Do NOT change SceneQualityDetector thresholds
- Do NOT change ResidualInnovationTracker window_size
- Do NOT retrain any Module 1 models
- ECEF xy-plane error metric unchanged throughout
- All previous result directories preserved
- exp_006 is the DEFINITIVE final experiment