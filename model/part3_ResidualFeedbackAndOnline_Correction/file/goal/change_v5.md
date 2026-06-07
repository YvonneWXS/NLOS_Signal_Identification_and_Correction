# Module 3 v5 — Documentation Changes

**Date**: 2026-06-07
**From**: v4 (commit 2d4872e)
**To**: v5 (documentation only)
**Type**: Documentation-only — NO algorithm changes, NO retraining

---

## Change 1: README.md Cleanup

**File**: model/README.md

- Version header: removed stale “(6/7 with bonus)” and “C4 miss accepted”
- Known Limitations: removed C4 (now passes), added M2/M3 FG discrepancy explanation, expanded frankfurt2 note
- References: updated to v4 documents

---

## Change 2: Paper Table FG Discrepancy Footnote

**File**: 
esult/exp_006/paper_table_v4.md + .tex

Added footnote explaining why Module 2 FG-MoG+2A frankfurt1 value differs between M2 standalone (476.9m, exp_038) and M3 internal (472.6m, exp_050). Both improve over Standard LS; Adaptive-M3 v4 outperforms both.

---

## Change 3: Frankfurt2 Degradation Explanation

**File**: 
esult/exp_004/key_findings.md

Added detailed explanation: the -490.7% metric is an artifact of outlier bins in the epoch-binned comparison, not progressive algorithmic degradation. Epoch-bin diagnosis shows no transition point.

---

## Change 4: Reproduction Script

**File**: model/reproduce_paper_results.py (NEW)

Single script that reproduces all numbers in paper_table_v4.md. Verifies against stored exp_006 results within 5m tolerance. Run: python reproduce_paper_results.py (~2 min).

---

## Change 5: Top-Level README

**File**: ../../README.md

Complete rewrite with:
- Final cross-module CEP50 table
- Module status (all COMPLETE)
- Updated repository structure
- Module summaries with key findings
- Quick start with reproduction command

---

*Generated: 2026-06-07*
