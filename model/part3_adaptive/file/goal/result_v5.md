# Module 3 v5 — Documentation Final Polish

**Date**: 2026-06-07
**Goal**: Paper-ready documentation cleanup. NO algorithm changes, NO retraining.
**Status**: COMPLETE

---

## Changes Made

| # | Task | File(s) | Description |
|---|------|---------|-------------|
| 1 | README v4 results | model/README.md | Updated header, Known Limitations, references to v4 final |
| 2 | FG discrepancy | 
esult/exp_006/paper_table_v4.md/.tex | Added footnote explaining M2 vs M3 FG frankfurt1 values |
| 3 | Frankfurt2 degradation | 
esult/exp_004/key_findings.md | Added honest explanation: artifact of outlier bins, not algorithmic failure |
| 4 | Reproduction script | model/reproduce_paper_results.py | Single script to reproduce all paper numbers (~2 min) |
| 5 | Top-level README | ../../README.md | Complete rewrite with final cross-module results, module status, quick start |

---

## Key Documentation Decisions

### Module 2 vs Module 3 FG frankfurt1 discrepancy

The paper table uses Module 2 v8 standalone value (476.9m, exp_038, dataset-specific M1 tuning)
for the `Module 2 FG-MoG+2A'' row, rather than Module 3 internal evaluation (472.6m v4, 596.9m v3).
This choice represents the best achievable with static fusion for fair comparison.
The footnote explains the difference transparently.

### Frankfurt2 online learning (-490.7%)

Epoch-bin diagnosis revealed this is an artifact of the `first vs last 100 epoch'' metric
being dominated by a single high-error bin (epochs 3382–3560, StdLS=1021m), not progressive
degradation. No transition point was found. The safety fallback prevents per-epoch from
exceeding Standard LS. This is a data characteristic, not an algorithm flaw.

---

## Files Created/Modified

| File | Action |
|------|--------|
| model/README.md | Updated: header, Known Limitations, references |
| 
esult/exp_006/paper_table_v4.md | Added: FG discrepancy footnote |
| 
esult/exp_006/paper_table_v4.tex | Added: FG discrepancy footnote |
| 
esult/exp_004/key_findings.md | Added: frankfurt2 degradation explanation |
| model/reproduce_paper_results.py | Created: reproduction script |
| ../../README.md | Rewritten: final cross-module summary |
| ile/goal/result_v5.md | Created: this file |
| ile/goal/change_v5.md | Created: change log |

---

*Generated: 2026-06-07 | v5 FINAL*
