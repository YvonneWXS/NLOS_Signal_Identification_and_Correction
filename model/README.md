# GNSS NLOS Signal Identification and Correction

**PI-PEM Research Framework**: Perception-Integration-Positioning with Environmental Modeling

## Overview

This project implements a three-module pipeline for GNSS positioning in urban environments:

| Module | Name | Function |
|--------|------|----------|
| Module 1 | `module1_nlos/` | NLOS perception & error distribution modeling (GAT + MoG) |
| Module 2 | `module2_localization/` | Multi-source fusion localization (13 methods: LS/WLS/FG/RAIM/EKF/... ) |
| Module 3 | `module3_adaptive/` | Adaptive localization method selection (residual tracking + scene detection) |
| Module 4 | `module4_experiments/` | Experiment framework (parameter sweep, baseline comparison, statistical tests) |
| Module 5 | `module5_visualization/` | Visualization (trajectory, error analysis, baseline comparison, report generation) |
| `common/` | Shared library | Coordinate transforms, SP3 reader, metrics, logging, config management |

## Quick Start

```bash
# Setup
cd model
pip install -r requirements.txt

# Module 1: Train NLOS GAT model
python -m module1_nlos.run --dataset berlin1_potsdamer_platz --mode train

# Module 2: Run localization with all methods
python module2_localization/run_fusion.py --dataset berlin1_potsdamer_platz

# Module 3: Adaptive selection
python module3_adaptive/run_module3.py --dataset berlin1_potsdamer_platz
```

## Directory Structure

```
model/
├── common/                 # Shared utilities
├── module1_nlos/           # NLOS GAT model
├── module2_localization/   # Fusion localization
├── module3_adaptive/       # Adaptive selection
├── module4_experiments/    # Experiment framework
├── module5_visualization/  # Visualization
├── scripts/                # Batch run scripts
├── results/                # Output directory
├── README.md               # This file
├── requirements.txt
└── pytest.ini
```

## Data

European 4-city dataset (Berlin1/2, Frankfurt1/2) stored in `data/`.

## Results

Key findings:
- Standard LS achieves best performance in 3/4 cities
- Factor Graph with MoG priors effective only in high-NLOS scenarios (Frankfurt1, NLOS > 50%)
- Adaptive selection provides safety fallback, matching best static method per city

## Reference

If using this code, please cite:
```
[Paper under preparation]
```

## License

Research code — contact authors for usage.
