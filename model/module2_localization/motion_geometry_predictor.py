"""
fusion/motion_geometry_predictor.py — Module 2A: TCN NLOS Prior Predictor
==========================================================================
Predicts next-epoch NLOS prior probabilities using historical trajectory
and satellite geometry sequences. Output serves as Bayesian prior for
Module 1 p_los at inference time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DilatedCausalConv1d(nn.Module):
    """Dilated causal 1D convolution with residual connection."""
    
    def __init__(self, channels, dilation, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,  # causal padding
        )
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T)  [batch, channels, time]
        Returns:
            out: (B, C, T)
        """
        residual = x
        out = self.conv(x)
        out = out[..., :x.size(-1)]  # trim causal padding
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out + residual)
        return out


class TCNPriorPredictor(nn.Module):
    """Temporal Convolutional Network for NLOS prior prediction.
    
    Input: sliding window of T epochs:
      - receiver position: (T, 3)
      - receiver velocity: (T, 3)
      - satellite geometry: (T, MAX_SV, 3) [elevation/90, azimuth/360, p_los]
      - visible mask: (T, MAX_SV)
    
    Output per epoch:
      - p_nlos_prior: (MAX_SV,) predicted NLOS probability
      - confidence: (MAX_SV,) prediction confidence
    """
    
    def __init__(self, max_sv=20, in_features_per_sv=3, pos_vel_features=6, 
                 hidden=128, tcn_layers=4, kernel_size=3):
        super().__init__()
        self.max_sv = max_sv
        self.hidden = hidden
        
        # Input projection per timestep
        flat_in = max_sv * in_features_per_sv + pos_vel_features  # 20*3 + 6 = 66
        self.input_proj = nn.Sequential(
            nn.Linear(flat_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        
        # TCN body
        dilations = [2**i for i in range(tcn_layers)]  # [1, 2, 4, 8]
        self.tcn_layers = nn.ModuleList([
            DilatedCausalConv1d(hidden, d, kernel_size)
            for d in dilations
        ])
        
        # Output heads
        self.p_nlos_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, max_sv),
            nn.Sigmoid(),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, max_sv),
            nn.Sigmoid(),
        )
    
    def forward(self, positions, velocities, sat_geometry, sat_mask):
        """
        Args:
            positions: (B, T, 3) receiver positions in km
            velocities: (B, T, 3) receiver velocities in km/s
            sat_geometry: (B, T, MAX_SV, 3) satellite geometry features
            sat_mask: (B, T, MAX_SV) binary visibility mask
        
        Returns:
            p_nlos_prior: (B, MAX_SV) predicted NLOS probabilities
            confidence: (B, MAX_SV) prediction confidence
        """
        B, T, _, _ = sat_geometry.shape
        
        # Flatten satellite features and concatenate position/velocity
        sat_flat = sat_geometry.reshape(B, T, -1)  # (B, T, MAX_SV*3)
        pos_vel = torch.cat([positions, velocities], dim=-1)  # (B, T, 6)
        combined = torch.cat([sat_flat, pos_vel], dim=-1)  # (B, T, MAX_SV*3+6)
        
        # Input projection
        x = self.input_proj(combined)  # (B, T, hidden)
        
        # TCN (need (B, C, T) format)
        x = x.transpose(1, 2)  # (B, hidden, T)
        for layer in self.tcn_layers:
            x = layer(x)
        x = x.transpose(1, 2)  # (B, T, hidden)
        
        # Take last timestep output
        x_last = x[:, -1, :]  # (B, hidden)
        
        # Output heads
        p_nlos_prior = self.p_nlos_head(x_last)
        confidence = self.confidence_head(x_last)
        
        return p_nlos_prior, confidence


class MotionGeometryPredictor:
    """Wrapper around TCNPriorPredictor for training and inference."""
    
    def __init__(self, max_sv=20, device='cpu'):
        self.max_sv = max_sv
        self.device = device
        self.model = TCNPriorPredictor(max_sv=max_sv).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    def prepare_sequences(self, all_epochs_data, mog_outputs, window=10):
        """Prepare training sequences from epoch data.
        
        Args:
            all_epochs_data: list of epoch dicts
            mog_outputs: list of MoG inference outputs
            window: sliding window length T=10
        
        Returns:
            sequences: list of (positions, velocities, sat_geometry, sat_mask, targets)
        """
        sequences = []
        
        for t in range(window, len(all_epochs_data)):
            seq_epochs = all_epochs_data[t - window:t]
            seq_mog = mog_outputs[t - window:t]
            next_mog = mog_outputs[t]
            
            # Positions (T, 3)
            pos = np.array([ep['gt_ecef'] for ep in seq_epochs])
            
            # Velocities (T, 3) via finite difference
            vel = np.zeros_like(pos)
            vel[1:] = pos[1:] - pos[:-1]  # km per epoch (~0.2s interval)
            vel[0] = vel[1] if window > 1 else 0
            
            # Satellite geometry (T, MAX_SV, 3)
            geo = np.zeros((window, self.max_sv, 3))
            mask = np.zeros((window, self.max_sv))
            
            for i, mog in enumerate(seq_mog):
                if mog is None:
                    continue
                n_sv = min(len(mog['p_los']), self.max_sv)
                geo[i, :n_sv, 0] = mog['elevation_deg'][:n_sv] / 90.0
                geo[i, :n_sv, 1] = mog['azimuth_deg'][:n_sv] / 360.0
                geo[i, :n_sv, 2] = mog['p_los'][:n_sv]
                mask[i, :n_sv] = 1.0
            
            # Target: NLOS probability for next epoch
            if next_mog is not None:
                n_sv = min(len(next_mog['p_los']), self.max_sv)
                target = np.zeros(self.max_sv)
                target[:n_sv] = 1.0 - next_mog['p_los'][:n_sv]  # p_nlos
                tmask = np.zeros(self.max_sv)
                tmask[:n_sv] = 1.0
            else:
                target = np.zeros(self.max_sv)
                tmask = np.zeros(self.max_sv)
            
            sequences.append((pos, vel, geo, mask, target, tmask))
        
        return sequences
    
    def train_step(self, batch_pos, batch_vel, batch_geo, batch_mask, batch_target, batch_tmask):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        p_nlos_prior, confidence = self.model(batch_pos, batch_vel, batch_geo, batch_mask)
        
        # Weighted BCE loss
        bce = F.binary_cross_entropy(p_nlos_prior, batch_target, reduction='none')
        weighted_bce = (bce * confidence * batch_tmask).sum() / max(batch_tmask.sum(), 1)
        
        loss = weighted_bce
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, positions, velocities, sat_geometry, sat_mask):
        """Predict NLOS prior for next epoch.
        
        Returns:
            p_nlos_prior: (MAX_SV,) numpy array
            confidence: (MAX_SV,) numpy array
        """
        self.model.eval()
        with torch.no_grad():
            pos_t = torch.tensor(positions, dtype=torch.float32, device=self.device).unsqueeze(0)
            vel_t = torch.tensor(velocities, dtype=torch.float32, device=self.device).unsqueeze(0)
            geo_t = torch.tensor(sat_geometry, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(sat_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            p_nlos_prior, confidence = self.model(pos_t, vel_t, geo_t, mask_t)
        
        return p_nlos_prior.squeeze(0).cpu().numpy(), confidence.squeeze(0).cpu().numpy()


def bayesian_prior_injection(p_los_gat, p_nlos_prior, confidence, conf_threshold=0.6):
    """Apply Bayesian prior injection to Module 1 p_los.
    
    p_los_fused = p_los_gat * (1 - p_nlos_prior) / Z
    where Z = p_los_gat * (1 - p_nlos_prior) + (1 - p_los_gat) * p_nlos_prior
    
    Only applied when confidence > conf_threshold.
    """
    p_los_fused = p_los_gat.copy()
    for i in range(len(p_los_gat)):
        if confidence[i] > conf_threshold and i < len(p_nlos_prior):
            p_los_gat_i = p_los_gat[i]
            p_nlos_prior_i = p_nlos_prior[i]
            num = p_los_gat_i * (1.0 - p_nlos_prior_i)
            den = num + (1.0 - p_los_gat_i) * p_nlos_prior_i
            if den > 1e-8:
                p_los_fused[i] = num / den
    return p_los_fused


print("fusion/motion_geometry_predictor.py loaded successfully")