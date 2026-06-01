Goal: Implement Module 2 (PI-PEM fusion layer) with two sub-modules:
2A (Motion-Geometry Joint State Predictor) and 2B (Factor Graph 
Soft-Information Fusion Positioner), using MoG Fix6 outputs as input.

================================================================
MODULE 2A: Motion-Geometry Joint State Predictor
================================================================

File: fusion/motion_geometry_predictor.py

PURPOSE:
Predict next-epoch NLOS prior probabilities for each satellite 
using historical trajectory + satellite geometry sequences.
Output serves as a prior injected into Module 1's p_los at 
inference time (Bayesian update: p_prior * p_gat → normalized).

INPUT (sliding window of last T=10 epochs):
  - receiver position history: (T, 3) ECEF km
  - receiver velocity (finite difference): (T, 3) km/s  
  - satellite geometry matrix per epoch: (T, MAX_SV, 3)
    where 3 = [elevation/90, azimuth/360, p_los from Module1]
  - visible satellite mask: (T, MAX_SV) binary
  MAX_SV = 20 (pad with zeros if fewer satellites)

OUTPUT per epoch:
  - p_nlos_prior: (MAX_SV,) predicted NLOS probability for 
    next epoch, for each satellite slot
  - confidence: (MAX_SV,) how confident the prediction is

ARCHITECTURE: Temporal Convolutional Network (TCN)
  - Input projection: Linear(MAX_SV*3 + 3, 128) per timestep
    (flatten satellite geometry + append velocity)
  - TCN body: 4 dilated causal conv1d layers
    dilation = [1, 2, 4, 8], kernel_size=3, channels=128
    Each layer: Conv1d + LayerNorm + GELU + residual
  - Output head: Linear(128, MAX_SV*2) → reshape to (MAX_SV, 2)
    apply Sigmoid for p_nlos_prior and confidence separately

TRAINING:
  - Supervised: use Module1 p_los outputs as soft labels
    (target = 1 - p_los from MoG Fix6 best_model)
  - Loss: BCE(p_nlos_prior, 1-p_los_module1) weighted by confidence
  - Optimizer: Adam lr=1e-4, weight_decay=1e-5
  - Epochs: 50, batch_size=32 sequences
  - Split: per-dataset, 80/20 train/val, no cross-epoch leakage

BAYESIAN PRIOR INJECTION at inference:
  If confidence_i > 0.6 for satellite i:
    p_los_fused_i = p_los_gat_i * (1 - p_nlos_prior_i) / Z
    where Z is normalizing constant
  Else: use p_los_gat_i unchanged

EVALUATION METRICS for 2A:
  - Prior accuracy: how often p_nlos_prior > 0.5 matches 
    ground truth NLOS label
  - AUC-ROC of p_nlos_prior vs ground truth
  - Report improvement in p_los gap when prior injection is used
    vs not used (delta_gap = gap_with_prior - gap_without_prior)

================================================================
MODULE 2B: Factor Graph Soft-Information Fusion Positioner
================================================================

File: fusion/factor_graph_fusion.py

PURPOSE:
Given per-satellite MoG parameters (p_los, mu_nlos, sigma_los, 
sigma_nlos) from Module 1 (optionally updated by 2A prior), 
compute optimal receiver position via factor graph optimization 
with MoG likelihood as observation noise model.

PHYSICAL MODEL:
State vector: x = [x_ecef, y_ecef, z_ecef, clk_bias] (4-DOF, km)

For satellite i with position sv_i (ECEF, km):
  predicted_pr_i = ||x[0:3] - sv_i|| + x[3]
  residual_i = pr_mes_i - predicted_pr_i

Log-likelihood (MoG observation factor):
  log p_i = logsumexp([
    log(p_los_i)   + logN(residual_i; 0,        sigma_los_i),
    log(1-p_los_i) + logN(residual_i; mu_nlos_i, sigma_nlos_i)
  ])
  where logN(r; mu, sigma) = -0.5*(r-mu)^2/sigma^2 - log(sigma)

Optimization objective:
  x* = argmax_x [ sum_i log p_i(residual_i | x) ]
  Solved via scipy.optimize.minimize(method='L-BFGS-B')
  Analytical Jacobian required (derive via chain rule through 
  predicted_pr geometry).

INITIALIZATION STRATEGY:
  Use WLS solution as warm start for L-BFGS-B.
  WLS weight: w_i = p_los_i / sigma_los_i^2
  Iterative linearized LS, max 5 iterations.

IMPLEMENTATION DETAILS:
  class MoGObservationModel:
    def log_likelihood(residual, p_los, mu_nlos, sigma_los, sigma_nlos)
    def jacobian(residual, x, sv_pos, p_los, mu_nlos, sigma_los, sigma_nlos)
      # returns d(log_p)/d(x) analytically
  
  class FactorGraphPositioner:
    def solve_epoch(observations, sv_positions) -> (x_ecef, convergence_info)
    def batch_solve(all_epochs_data) -> position_series

BASELINES (implement in fusion/baselines.py):
  1. Standard LS: uniform weights, all satellites
  2. WLS-elevation: weight = sin(elevation)^2
  3. WLS-MoG: weight = p_los_i / sigma_los_i^2 (uses Module1 output)
  4. Hard-threshold: exclude p_los < 0.5, then LS on remainder

================================================================
EVALUATION (fusion/evaluate_fusion.py)
================================================================

Load Module 1 MoG Fix6 best_model.pth for each of the 4 datasets.
Run inference to get (p_los, mu_nlos, sigma_los, sigma_nlos) per 
satellite per epoch. Then run all positioning methods.

COORDINATE CONVERSION:
  Reuse existing Radio_Depth_Generate.py ECEF↔LLA functions.
  Ground truth from NAV-POSLLH.csv → convert to ECEF for comparison.

METRICS (report per dataset, all epochs):
  - CEP50: median 2D horizontal error (meters)
  - CEP95: 95th percentile 2D error
  - Mean 2D error
  - RMSE 3D
  - % epochs with error < 5m / < 10m / < 20m

REPORT TABLE FORMAT:
  Method          | berlin1 CEP50 | berlin2 CEP50 | frk1 CEP50 | frk2 CEP50
  Standard LS     |
  WLS-elevation   |
  WLS-MoG         |
  Hard-threshold  |
  FactorGraph-MoG |
  FactorGraph-MoG+2A |  (with prior injection from 2A)

EXPECTED RESULTS (target, not guaranteed):
  FactorGraph-MoG should outperform WLS-MoG by >10% on CEP50
  in at least 3/4 datasets (theoretical advantage from using 
  full MoG distribution vs. only variance weighting).
  2A prior injection should further improve CEP50 by >5% in 
  berlin2 and frankfurt1 (higher NLOS ratio scenes).

================================================================
FILE STRUCTURE
================================================================
fusion/
  __init__.py
  motion_geometry_predictor.py  — Module 2A: TCN predictor
  factor_graph_fusion.py        — Module 2B: MoG factor graph
  baselines.py                  — 4 baseline methods
  evaluate_fusion.py            — end-to-end eval + table
  utils.py                      — coordinate transforms (reuse existing)

================================================================
CONSTRAINTS
================================================================
- No GTSAM/g2o required; use scipy.optimize only
- Reuse Radio_Depth_Generate.py for ECEF/LLA conversion
- Do NOT modify any Module 1 files (GAT_V2025.py, config.py etc.)
- Clock bias must be 4th DOF in state vector
- All 4 datasets must be evaluated
- Module 2A is optional enhancement; 2B must work standalone