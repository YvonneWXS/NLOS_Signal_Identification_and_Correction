# Module 1: NLOS Perception & Error Modeling (Part1 GAT)

## 1. Module Overview

### 1.1 Function
NLOS classification and pseudorange error modeling using Graph Attention Networks (GAT) with Mixture of Gaussians (MoG) output. Role in PI-PEM: **Perception** — predicts per-satellite LOS probability and error distribution parameters.

### 1.2 Core Flow
`
Raw GNSS epochs -> Node features (11-dim) -> Azimuth graph -> GAT encoder -> MoG heads -> (p_los, mu_nlos, sigma_los, sigma_nlos)
`

### 1.3 Input & Output
- **Input**: Processed epoch data from data/processedData/{dataset}_processed.pkl
- **Output**: Per-satellite predictions:
  - p_los: LOS probability [0, 1]
  - mu_nlos: NLOS pseudorange bias (km)
  - sigma_los: LOS uncertainty std (km)
  - sigma_nlos: NLOS uncertainty std (km)

## 2. Architecture

### 2.1 File Structure
`
part1_GAT/model/
├── run.py              -> CLI entry point
├── config.py           -> Centralized configuration
├── GAT_V2025.py        -> Model (GATLayer + NLOSGAT) + MoG loss + training loop
├── Data_read.py        -> Dataset loading, SP3 parsing, caching
├── NodeFeature_Generate.py -> 11-dim feature extraction
├── Depth_Adj_Generate.py   -> Azimuth graph construction
├── Radio_Depth_Generate.py -> Radio depth computation
├── sp3_reader.py       -> SP3 precise ephemeris reader
├── New_axis40.txt      -> Reference station coordinates
├── stations_position.txt -> Station positions
└── results/            -> Experiment outputs
`

### 2.2 Core Classes
| Class | File | Function |
|-------|------|----------|
| GATLayer | GAT_V2025.py | Vectorized multi-head GAT with block-diagonal batching |
| NLOSGAT | GAT_V2025.py | 4-head MoG model (p_los, mu_nlos, sigma_los, sigma_nlos) |
| NLOSLoss | GAT_V2025.py | BCE + heteroscedastic uncertainty (warmup phase) |
| MoGNLLLoss | GAT_V2025.py | Full MoG NLL (post-warmup phase) |
| Config | config.py | All training, model, loss, and data parameters |

### 2.3 Data Flow
1. Data_read.load_and_process_dataset() loads cached epoch data
2. NodeFeature_Generate.extract_node_features() computes 11-dim features per satellite
3. Depth_Adj_Generate.build_azimuth_graph() constructs edges (azimuth diff < 90 deg)
4. GNSDataset wraps epochs into PyTorch Dataset with block-diagonal collation
5. 	rain_epoch() runs 3-phase training: BCE warmup -> blend -> MoG NLL

## 3. Configuration

### 3.1 Key Parameters (config.py)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| LEARNING_RATE | float | 5e-5 | Adam learning rate |
| NUM_EPOCHS | int | 100 | Total training epochs |
| BATCH_SIZE | int | 32 | Block-diagonal batch size |
| HIDDEN_FEATURES | int | 128 | GAT hidden dimension |
| NUM_HEADS | int | 8 | Multi-head attention heads |
| NUM_LAYERS | int | 2 | GAT layer count |
| LAMBDA_BCE | float | 0.6 | BCE loss weight |
| LAMBDA_ENTROPY | float | 0.03 | Entropy regularization |
| MOG_PURE_BCE_EPOCHS | int | 8 | Phase 1: BCE-only warmup |
| MOG_BLEND_EPOCHS | int | 25 | Phase 2: BCE-to-NLL transition |
| USE_BLOCK_DIAGONAL | bool | True | Enable fast batched training |
| USE_AMP | bool | True | Automatic mixed precision |
| USE_TENSORBOARD | bool | True | TensorBoard logging |

## 4. Usage

### 4.1 CLI
`ash
cd part1_GAT/model
python run.py --dataset berlin1_potsdamer_platz --exp-name exp_001
python run.py --dataset berlin2_gendarmenmarkt --exp-name exp_002 --epochs 50
`

### 4.2 Output
Results saved to part1_GAT/results/{exp_name}/:
- checkpoints/best_model.pth — Best validation model
- checkpoints/checkpoint_epoch_*.pth — Periodic checkpoints
- 	ensorboard/ — TensorBoard event files
- predictions.json — Final epoch predictions

## 5. API Reference

### GAT_V2025.main()
`python
def main(resume_from=None, num_epochs=None, dataset_name=None, exp_name=None)
`
- esume_from: Checkpoint path to resume training
- 
um_epochs: Override config.NUM_EPOCHS
- dataset_name: Single dataset name
- exp_name: Experiment folder name under results/

## 6. Dependencies

- torch>=1.9, numpy, scipy
- Internal: config, Data_read, NodeFeature_Generate, Depth_Adj_Generate

## 7. Tests

`ash
# Quick smoke test (2 epochs)
cd part1_GAT/model
python run.py --dataset berlin1_potsdamer_platz --exp-name test --epochs 2

# Monitor with TensorBoard
tensorboard --logdir=../results/test/tensorboard
`

## 8. FAQ

**Q: How long does training take?**
A: ~52 min for 100 epochs on berlin1 (1377 epochs, bs=32, RTX 5060).

**Q: GAT_V2025 vs GAT_V2026?**
A: V2025 is the production model with full MoG output (4 heads). V2026 removed.

**Q: Block-diagonal batching?**
A: Multiple epochs stacked into one large block-diagonal graph. Enables bs=32 with 2.7x speedup vs bs=1.
