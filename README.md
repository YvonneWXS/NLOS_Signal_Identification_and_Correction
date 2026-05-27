# NLOS Signal Identification and Correction

**Urban GNSS NLOS Signal Soft Error Perception and Correction Framework**

---

## Overview

This project addresses the challenge of **GNSS Non-Line-of-Sight (NLOS) signal identification and correction** in dense urban environments. Instead of traditional hard binary classification (LOS/NLOS), we adopt a **soft error perception** paradigm: for each visible satellite at each GNSS epoch, the model outputs **mixture Gaussian distribution parameters** that can be optimally fused by downstream factor graph optimization.

### Research Framework: PI-PEM

The project follows the **PI-PEM (NLOS Perception and Error Distribution Modeling)** framework, organized into progressive modules:

| Module | Name | Description | Status |
|--------|------|-------------|--------|
| Module 1 | NLOS Perception and Error Distribution Modeling | GAT-based soft error perception | In Progress |
| Module 2 | Motion-Geometry Joint State Prediction | Temporal-geometric fusion for positioning | Planned |
| Module 3 | Factor Graph Optimization Positioning | Soft-information-driven optimal estimation | Planned |
| Module 4 | Residual Feedback and Correction | Closed-loop error compensation | Planned |

---

## Repository Structure

```
NLOS_Signal_Identification_and_Correction/
|
+-- model/part1_GAT/               # Module 1: GAT-based NLOS perception
|   +-- RadioGAT-Multi-band-Radiomap-Reconstruction/
|   |   +-- GAT_V2025.py           # Main: model + loss + training + evaluation
|   |   +-- config.py              # Centralized configuration
|   |   +-- run_full_training.py   # One-click training entry
|   |   +-- run_serial.py          # Serial 4-city training pipeline
|   |   +-- generate_report.py     # Analysis report generator
|   |   +-- Data_read.py           # GNSS data loading and preprocessing
|   |   +-- NodeFeature_Generate.py # 11-dim node feature extraction
|   |   +-- Depth_Adj_Generate.py  # Azimuth-based graph construction
|   |   +-- Radio_Depth_Generate.py # Satellite geometry computation
|   |   +-- sp3_reader.py          # SP3 precise ephemeris parser
|   |   +-- analyze_experiment.py  # Experiment analysis module
|   |   +-- analyze_model.py       # Model analysis utilities
|   |   +-- train_wrapper.py       # Training wrapper
|   |   +-- positioning_test.py    # Positioning evaluation
|   |   '-- README.md              # Module documentation
|   '-- result/                    # Training outputs (per experiment)
|
+-- baseline/                      # Baseline comparison models
|   +-- CNN_Attention_BiLSTM/      # CNN + Attention + BiLSTM
|   +-- CNN_LSTM/                  # CNN + LSTM
|   +-- TC_CNN_BiLSTM/             # Temporal Convolutional CNN + BiLSTM
|   +-- Random_Forest/             # Random Forest classifier
|   +-- XGboost/                   # XGBoost classifier
|   '-- K_means/                   # K-Means clustering baseline
|
+-- data/                          # Datasets
|   +-- dataset/                   # Raw GNSS data (4 cities)
|   '-- processedData/             # Cached preprocessed data
|
+-- file/                          # Research documents
|   '-- (research framework paper)
|
'-- README.md                      # This file
```

---

## Datasets

4 urban GNSS datasets from Berlin and Frankfurt, Germany:

| Dataset | City | Location | Epochs | LOS Rate | NLOS Rate |
|---------|------|----------|--------|----------|-----------|
| berlin1_potsdamer_platz | Berlin | Potsdamer Platz | 1,377 | 51.7% | 48.3% |
| berlin2_gendarmenmarkt | Berlin | Gendarmenmarkt | 5,925 | 61.2% | 38.8% |
| frankfurt1_maintower | Frankfurt | Maintower | 5,851 | 57.0% | 43.0% |
| frankfurt2_westendtower | Frankfurt | Westendtower | 3,575 | 73.4% | 26.6% |

Each epoch contains 6-18 visible satellites with measurements including pseudorange, C/N0, elevation, azimuth, and ground-truth NLOS labels.

---

## Model Architecture

### GAT-Based NLOS Perception

```
Input: (N, 11) node features per epoch
       |
       v
[Input Projection]: Linear(11 -> 128) + ReLU + Dropout(0.1)
       |
       v
[GAT Layer x2]: 128-dim, 8 heads, concat=False, residual + LayerNorm
       |
       v
[Output Projection]: Linear(128 -> 128) + ReLU
       |
       +---> p_los_head:   Sigmoid         -> p(LOS) in [0,1]
       +---> uncertainty_head: Linear      -> log_sigma
```

### 11-Dimensional Node Features

| Dim | Feature | Normalization | Physical Meaning |
|-----|---------|--------------|------------------|
| 0 | elevation | / 90 deg | Low elevation -> higher NLOS probability |
| 1 | azimuth | / 360 deg | Directional context |
| 2 | C/N0 | / 60 dBHz | Signal quality indicator |
| 3 | prStdev | / 5 m | Measurement uncertainty (receiver report) |
| 4 | prMes | / 3e8 m | Observed range magnitude |
| 5 | prInnovation | / 100 m | Pseudorange innovation |
| 6 | cos(elevation) | - | Geometric precision proxy |
| 7-10 | Constellation | one-hot | GPS / GLONASS / Galileo / BeiDou |

### Graph Construction

- Edge condition: azimuth difference < 90 degrees (bidirectional)
- Edge weight: |az_i - az_j| / 90 in [0, 1] (linear normalization)
- Self-loops automatically added when no valid edges exist

### Loss Function

```
L_total = lambda_bce * BCE(p_los, label)
          + lambda_unc * GaussianNLL(error | sigma)
          - lambda_entropy * H(p_los)
          + lambda_elevation * ElevationPhysicalPrior
```

---

## Key Findings (Module 1 Analysis)

1. **Node features dominate**: The model primarily relies on per-satellite features (elevation, C/N0, prStdev); GAT neighbor aggregation is underutilized.
2. **FN pattern**: False negatives (missed NLOS) occur on high-elevation, high-C/N0, low-prStdev satellites that visually resemble LOS.
3. **FP pattern**: False positives (misclassified LOS) occur on low-elevation, low-C/N0, high-prStdev satellites.
4. **Sigma failure**: In Frankfurt scenes, sigma(NLOS) is nearly equal to sigma(LOS), indicating the BCE+Uncertainty architecture fails to distinguish measurement quality between LOS and NLOS.
5. **Small graphs**: With only 8-20 nodes per epoch and batch_size=1, the GAT struggles to learn rich graph structural patterns.

---

## Quick Start

### Environment

```txt
torch >= 2.0.0 (CUDA recommended)
numpy, pandas, scipy
```

### Training

```bash
# Enter module directory
cd model/part1_GAT/RadioGAT-Multi-band-Radiomap-Reconstruction

# Single dataset training
python run_full_training.py --exp-name exp_001 --dataset berlin1_potsdamer_platz

# Serial 4-city training (with auto-analysis)
python run_serial.py

# Parallel training (4 separate CMD windows, same directory)
python run_full_training.py --exp-name exp_001 --dataset berlin1_potsdamer_platz
python run_full_training.py --exp-name exp_002 --dataset berlin2_gendarmenmarkt
python run_full_training.py --exp-name exp_003 --dataset frankfurt1_maintower
python run_full_training.py --exp-name exp_004 --dataset frankfurt2_westendtower
```

### Analysis

```bash
python generate_report.py --exp exp_001 --dataset berlin1_potsdamer_platz
```

### TensorBoard Monitoring

```bash
tensorboard --logdir=model/part1_GAT/result/exp_001/tensorboard
```

Training metrics tracked: Loss, BCE, Uncertainty, Accuracy, F1, Precision, Recall, p_LOS distribution, sigma distribution, gradient norms, learning rate.

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| LEARNING_RATE | 1e-4 | Initial learning rate |
| NUM_EPOCHS | 100 | Maximum training epochs |
| BATCH_SIZE | 1 | Forced to 1 (variable satellites per epoch) |
| GRADIENT_ACCUMULATION | 8 | Effective batch size via accumulation |
| HIDDEN_FEATURES | 128 | GAT hidden dimension |
| NUM_HEADS | 8 | Attention heads per layer |
| NUM_LAYERS | 2 | GAT layers |
| DROPOUT | 0.1 | Dropout rate |
| AZIMUTH_THRESHOLD | 90 deg | Graph edge threshold |
| LAMBDA_BCE | 0.6 | BCE loss weight |
| LAMBDA_ENTROPY | 0.03 | Entropy regularization weight |
| LAMBDA_ELEVATION_PRIOR | 0.1 | Elevation physical prior weight |
| EARLY_STOPPING_PATIENCE | 20 | Early stopping patience |
| POS_WEIGHT | 1.07 | NLOS class weight |

---

## Results Structure

Training outputs saved per experiment:

```
result/exp_001/
+-- best_model.pth              # Best validation model (with optimizer state)
+-- final_model.pth             # Final model at training end
+-- checkpoints/                # Per-epoch checkpoint files
+-- tensorboard/                # TensorBoard event files
+-- result.md                   # Full analysis report
'-- env.md                      # Environment configuration snapshot
```

---

## Reference

This project adapts the RadioGAT architecture for GNSS NLOS perception:

> X. Li et al., "RadioGAT: A Joint Model-Based and Data-Driven Framework for Multi-Band Radiomap Reconstruction via Graph Attention Networks," IEEE Transactions on Wireless Communications, vol. 23, no. 11, pp. 17777-17792, Nov. 2024.

Research framework document: `file/Research_Framework.pdf`
