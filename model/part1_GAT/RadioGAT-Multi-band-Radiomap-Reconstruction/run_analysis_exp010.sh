#!/usr/bin/env bash
# Run analysis for exp_010
set -e
SRC_DIR="D:/3_document/4_research/NLOS Signal Identification and Correction/model/part1_GAT/RadioGAT-Multi-band-Radiomap-Reconstruction"
PYTHON="D:/1_developTool/4_conda/envs/smartLoc/python.exe"
cd "$SRC_DIR"
echo "=== Running analysis for exp_010 (frankfurt1_maintower) ==="
"$PYTHON" analyze_experiment.py --exp exp_010 --dataset frankfurt1_maintower 2>&1
echo "=== Done ==="