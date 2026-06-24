# GNSS NLOS Signal Identification and Correction

PI-PEM Research Framework: **Perception-Integration-Positioning with Environmental Modeling**

## Structure

| Part | Module | Function |
|------|--------|----------|
| part1_GAT | NLOS Perception | GAT + MoG: per-satellite LOS probability and error modeling |
| part2_localization | Fusion Positioning | 9 solvers: LS, WLS, Factor Graph, Hard-threshold, RAIM, IRLS |
| part3_adaptive | Adaptive Selection | Online method selection via residual tracking + scene quality |

## Quick Start

`ash
# Part1: Train NLOS model
cd part1_GAT/model
python run.py --dataset berlin1_potsdamer_platz --exp-name exp_001

# Part2: Run positioning methods
cd part2_localization/model
python run.py --dataset berlin1_potsdamer_platz --methods all

# Part3: Adaptive selection
cd part3_adaptive/model
python run.py --dataset berlin1_potsdamer_platz --input ../../part2_localization/results/exp_001/
`

## Datasets

4 European urban datasets in data/dataset/:
- berlin1_potsdamer_platz (1377 epochs, 48.3% NLOS)
- berlin2_gendarmenmarkt (5925 epochs, 38.8% NLOS)
- frankfurt1_maintower (5851 epochs, 43.0% NLOS)
- frankfurt2_westendtower (3575 epochs, 26.6% NLOS)

## Environment

`ash
conda activate smartLoc
pip install torch numpy scipy matplotlib
`

## Key Findings

- EKF > Standard LS > all weighted/ML methods on European urban data
- MoG-based weighting degrades positioning (+15-33% vs Standard LS)
- Simple weighting (C/N0, SNR, elevation) is unreliable across datasets
