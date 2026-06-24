# Model Directory Restructuring Plan

**Version**: v3
**Date**: 2026-06-24
**Source**: model_2 renamed to model, to be restructured

## Summary

Reorganize model/ into three clean modules: Part1 NLOS GAT, Part2 Localization, Part3 Adaptive. Each gets a single run.py entry point, 8-chapter README, and results/ output directory. Part1 uses GAT_V2025 (full MoG model). All 6 protected folders untouched.

## Part1: part1_GAT/model/ -- NLOS Perception

### Deletions (14 files)
Remove analysis/diagnostic/backup/training-variant scripts:
- analyze_experiment.py, analyze_model.py, analyze_mog.py
- generate_report.py, gen_predictions.py, positioning_test.py
- resume_hk.py, run_hk_bce.py, run_urbannav.py, run_serial.py
- train_wrapper.py, GAT_V2026.py

### Renames
- run_full_training.py -> run.py

### Retained (10 files)
- config.py, Data_read.py, Depth_Adj_Generate.py, GAT_V2025.py, NodeFeature_Generate.py, Radio_Depth_Generate.py, sp3_reader.py, New_axis40.txt, stations_position.txt, README.md

### Edits
- config.py: change RESULT_DIR from model/part1_GAT/result to model/part1_GAT/results
- run.py: rename GAT_V2025 import if needed; add argparse for --exp-name, --dataset, --epochs

### New files
- README.md: 8-chapter format covering GAT+MoG model, block-diagonal batching, TensorBoard
- .gitignore: ignore results/, __pycache__/, *.pth, runs/
- results/ directory (empty)

## Part2: part2_localization/model/ -- Fusion Localization

### Move
- All 13 .py files from model/fusion/ -> model/
- After move, delete empty fusion/ directory and __pycache__/

### Renames
- run_fusion.py -> run.py

### Deletions
- run_positioning.py, run_pos_quick.py, run_all.py (merged into run.py CLI)

### Edits
- Fix 'from fusion.xxx import ...' -> 'from xxx import ...' in all moved files
- run.py: consolidate sub-commands, add argparse for --method, --dataset, --input

### New files
- README.md: 8-chapter format covering LS/WLS/FG/Hard-threshold/RAIM/IRLS/EKF methods
- .gitignore
- results/ directory

## Part3: part3_adaptive/model/ -- Adaptive Selection

### Renames
- run_module3.py -> run.py

### Edits
- run.py: add argparse for --dataset, --input, --output
- README.md: ensure 8-chapter format

### New files
- .gitignore
- results/ directory

## Model Root

### New files
- README.md: project-level overview of all 3 parts, quick start
- .gitignore: */results/, __pycache__/, *.pth, *.pyc, runs/

## Execution Order

1. Git backup current state
2. Part1 deletions + rename + config fix + README + .gitignore
3. Part2 move fusion->model + rename + fix imports + README + .gitignore
4. Part3 rename + README + .gitignore
5. Root README + .gitignore
6. Git backup final state + push

## Assumptions

- No existing checkpoints or cache files to preserve
- Part1 must be retrained from scratch on all 4 datasets
- GAT_V2025.py is the authoritative model file; GAT_V2026.py is discarded
- Protected folders (file/, project/) are never touched
- PowerShell is the shell, Python 3.9 conda env smartLoc
