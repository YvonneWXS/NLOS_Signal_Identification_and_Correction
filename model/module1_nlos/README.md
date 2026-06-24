# Module 1: NLOS Perception & Error Modeling

## 1. Module Overview

### 1.1 Function
GAT-based NLOS detection with Mixture-of-Gaussians output. Detects per-satellite LOS/NLOS probability (p_los), NLOS bias (mu_nlos), LOS uncertainty (sigma_los), and NLOS uncertainty (sigma_nlos).

### 1.2 Core Flow
Raw GNSS observations -> 11-dim feature extraction -> Graph construction (azimuth threshold) -> GAT (8-head, 2-layer) -> MoG output (p_los, mu, sigma_los, sigma_nlos)

### 1.3 Input/Output
- Input: Processed pickle (per-epoch: gnss_id, sv_id, pr_mes, elevation, azimuth, cno, nlos_label)\n- Output: MoG outputs (p_los, p_los_sharp, mu_nlos, sigma_los, sigma_nlos) per epoch

## 2. Architecture

### 2.1 Internal Structure
`
module1_nlos/
+-- data_loader.py   -> GNSSDataset, collate_fn, dataloaders
+-- features.py      -> NodeFeatureGenerator, GraphBuilder
+-- model.py          -> GATLayer, GATMoG network
+-- loss.py           -> MoGLoss (BCE->Blend->NLL 3-stage)
+-- trainer.py        -> Training loop, AMP, gradient accumulation
+-- inference.py      -> Load checkpoint, forward pass, analyze
+-- run.py            -> CLI entry point
+-- config.yaml       -> Module-specific parameters
`

### 2.2 Core Dependencies
- Depends on common/coordinate.py, common/sp3_reader.py\n- Output consumed by module2_localization and module3_adaptive

## 3. Configuration

### 3.1 config.yaml Parameters
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| dataset.name | str | berlin1 | berlin1/berlin2/frankfurt1/frankfurt2 | Dataset identifier |
| model.hidden_features | int | 128 | 64-256 | Hidden layer dimension |
| model.num_heads | int | 8 | 4-16 | Multi-head attention heads |
| training.learning_rate | float | 5.0e-6 | 1e-6 - 1e-4 | Adam learning rate |
| training.batch_size | int | 32 | 1-64 | Block-diagonal batch size |
| loss.pure_bce_epochs | int | 8 | 5-15 | Stage 1: pure BCE epochs |
| loss.blend_epochs | int | 25 | 15-40 | Stage 2: BCE->NLL transition |

## 4. Usage

### 4.1 Command Line
`ash
# Training
python -m model.module1_nlos.run --dataset berlin1 --mode train

# Inference (from checkpoint)
python -m model.module1_nlos.run --dataset berlin1 --mode inference --checkpoint best_model.pth
`

### 4.2 Python API
`python
from module1_nlos.inference import InferenceEngine
engine = InferenceEngine('best_model.pth')
outputs = engine.predict(epoch_data)  # -> {p_los, mu_nlos, sigma_los, sigma_nlos}
`

## 5. Core API Reference

### NLOSGAT (model.py)
`python
class NLOSGAT(nn.Module):
    def forward(x, edge_index) -> (p_los, log_sigma)
`
### MoGLoss (loss.py)
`python
class MoGLoss(nn.Module):
    def forward(p_los, log_sigma, errors, labels) -> (loss, components)
`

## 6. Dependencies

### 6.1 Internal
- common.coordinate: LLA<->ECEF transforms\n- common.sp3_reader: SP3 precise ephemeris

### 6.2 External
- torch>=1.9.0\n- torch_geometric>=2.0\n- numpy, scipy

## 7. Testing

`ash
pytest model/module1_nlos/tests/ -v
`

## 8. FAQ

### Q: Why 3-stage training?
Stage 1 (BCE) stabilizes p_los. Stage 2 (blend) introduces NLL gradually. Stage 3 (pure NLL) optimizes full MoG.

### Q: Checkpoint compatibility?
Use checkpoints from model_2/part1_GAT/result/exp_001-004/best_model.pth.
