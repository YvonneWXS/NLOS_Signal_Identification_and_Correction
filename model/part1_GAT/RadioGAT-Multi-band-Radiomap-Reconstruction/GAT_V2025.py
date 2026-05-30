# -*- coding: utf-8 -*-
"""
GAT_V2025.py -- NLOS Perception GAT Main File
==============================================
Part 1: Model Definition -- GATLayer + NLOSGAT
Part 2: Loss Function    -- NLOSLoss (BCE + Uncertainty + Entropy)
Part 3: Training         -- Dataset + train_epoch + evaluate
Part 4: Main Entry       -- main() training orchestration

Current mode: BCE + Uncertainty (USE_MIXTURE_GAUSSIAN=False)
"""

import os
import sys
import math
import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, List, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config, get_config
from NodeFeature_Generate import (
    extract_node_features, extract_labels, extract_pseudorange_errors,
    FEATURE_DIM
)
from Depth_Adj_Generate import build_azimuth_graph


def _safe_torch_save(obj, path):
    """Save using BytesIO buffer to avoid sandbox issues with torch.save."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    with open(path, "wb") as f:
        f.write(buf.getvalue())


# ============================================================
# Part 1: Model Definition
# ============================================================


class GATLayer(nn.Module):
    """Graph Attention Layer (preserves original RadioGAT custom style)"""

    def __init__(self, in_features: int, out_features: int, heads: int = 4,
                 dropout: float = 0.1, concat: bool = True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, heads * out_features)))
        self.att = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.att.data)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Vectorized forward (v2026 block-diagonal batching compatible)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        N = x.size(0)
        h = self.heads
        out_dim = self.out_features

        x_hat = torch.mm(x, self.W).view(N, h, out_dim)      # (N, H, D)
        out = torch.zeros(N, h, out_dim, device=x.device, dtype=x_hat.dtype)

        if edge_index.size(1) > 0:
            att_weight = F.softmax(self.att.view(-1), dim=0) # (2*D,)

            src = edge_index[0].long()                       # (E,)
            dst = edge_index[1].long()                       # (E,)
            mask = (src < N) & (dst < N)
            src, dst = src[mask], dst[mask]

            for head_idx in range(h):
                msgs = x_hat[src, head_idx, :] * att_weight[head_idx]
                out[:, head_idx, :].index_add_(0, dst, msgs)
        else:
            out = x_hat

        if self.concat:
            out = out.view(N, -1)
        else:
            out = torch.mean(out, dim=1)
        return out


class NLOSGAT(nn.Module):
    """
    NLOS Perception GAT Network

    Output:
      p_los:      (N, 1) LOS probability in [0, 1]      (Sigmoid)
      log_sigma_nlos:  (N, 1) predicted uncertainty log-std   (unactivated)

    Architecture:
      Linear(11->128) -> 2xGATLayer(128, heads=8, concat=False)
      -> LayerNorm -> Output -> 2 heads
    """

    def __init__(self, in_features: int = FEATURE_DIM,
                 hidden_features: int = 128, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        super(NLOSGAT, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(GATLayer(
                hidden_features, hidden_features, heads=num_heads,
                dropout=dropout, concat=False
            ))
            self.norm_layers.append(nn.LayerNorm(hidden_features))

        self.output = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.p_los_head = nn.Linear(hidden_features, 1)
        self.mu_nlos_head = nn.Linear(hidden_features, 1)
        self.log_sigma_los_head = nn.Linear(hidden_features, 1)
        self.log_sigma_nlos_head = nn.Linear(hidden_features, 1)

        self._init_output_layers()

    def _init_output_layers(self):
        """Initialize output heads to avoid activation saturation."""
        nn.init.constant_(self.p_los_head.bias, -0.5)
        nn.init.normal_(self.p_los_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.mu_nlos_head.bias, -2.0)  # softplus(-2.0) ≈ 0.127 km
        nn.init.normal_(self.mu_nlos_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_sigma_los_head.bias, -2.0)  # exp(-2.0)?0.135 km, was -6.0?0.0025
        nn.init.normal_(self.log_sigma_los_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_sigma_nlos_head.bias, -3.0)
        nn.init.normal_(self.log_sigma_nlos_head.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = x.size(0)
        device = x.device

        if N == 0:
            zeros = torch.zeros(1, 1, device=device)
            return zeros, zeros, zeros, zeros

        if edge_index.dim() != 2 or edge_index.size(1) == 0 or \
           (edge_index.size(1) > 0 and edge_index.max() >= N):
            edge_index = torch.tensor(
                [[i for i in range(N)], [i for i in range(N)]],
                device=device, dtype=torch.long
            )
        else:
            edge_index = edge_index.to(torch.long)

        h = self.input_proj(x)
        if h.dim() == 1:
            h = h.unsqueeze(0)

        for gat_layer, norm_layer in zip(self.gat_layers, self.norm_layers):
            h_res = h
            h = gat_layer(h, edge_index)
            h = F.elu(h)
            h = norm_layer(h)
            h = h + h_res
            if self.training:
                h = F.dropout(h, p=0.1)

        h = self.output(h)

        p_los = torch.sigmoid(self.p_los_head(h))
        mu_nlos_raw = self.mu_nlos_head(h)
        mu_nlos = F.softplus(mu_nlos_raw)
        mu_nlos = torch.clamp(mu_nlos, 0.0, 500.0)  # match MU_NLOS_MAX
        log_sigma_los = self.log_sigma_los_head(h)
        log_sigma_nlos = self.log_sigma_nlos_head(h)

        return p_los, mu_nlos, log_sigma_los, log_sigma_nlos


# ============================================================
# Part 2: Loss Function
# ============================================================


class NLOSLoss(nn.Module):
    """
    BCE + Heteroscedastic Uncertainty + Entropy regularization

    L_total = lambda_bce * L_BCE + L_uncertainty - lambda_entropy * H(p_los)
    """

    def __init__(self, pos_weight: float = 1.07,
                 label_smoothing: float = 0.05,
                 lambda_bce: float = 0.6,
                 p_los_smoothing: float = 0.2,
                 lambda_entropy: float = 0.03,
                 lambda_unc: float = 0.08,
                 lambda_elevation_prior: float = 0.1):
        super(NLOSLoss, self).__init__()
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.lambda_bce = lambda_bce
        self.p_los_smoothing = p_los_smoothing
        self.lambda_entropy = lambda_entropy
        self.lambda_unc = lambda_unc
        self.lambda_elevation_prior = lambda_elevation_prior
        self.eps = 1e-6

    def forward(self, p_los: torch.Tensor, log_sigma_nlos: torch.Tensor,
                pseudorange_error: torch.Tensor,
                nlos_label: torch.Tensor,
                elevation: Optional[torch.Tensor] = None,
                return_components: bool = False):
        N = p_los.size(0)
        if N == 0:
            loss = torch.tensor(0.0, device=p_los.device)
            return (loss, {}) if return_components else loss

        p_los = p_los.squeeze()
        log_sigma_nlos = log_sigma_nlos.squeeze()
        pseudorange_error = pseudorange_error.squeeze()
        nlos_label = nlos_label.squeeze()

        weights = torch.where(nlos_label == 1,
                              torch.tensor(self.pos_weight, device=p_los.device),
                              torch.tensor(1.0, device=p_los.device))

        target_los = 1.0 - nlos_label
        if self.label_smoothing > 0:
            target_los = target_los * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        p_los_smooth = p_los * (1.0 - self.p_los_smoothing) + 0.5 * self.p_los_smoothing
        bce = F.binary_cross_entropy(
            torch.clamp(p_los_smooth, self.eps, 1.0 - self.eps),
            target_los,
            weight=weights.detach()
        )

        sigma = torch.exp(log_sigma_nlos)
        sigma = torch.clamp(sigma, self.eps, 1e6)
        per_sample_unc = 0.5 * torch.log(sigma ** 2) + \
                         0.5 * (pseudorange_error ** 2) / (sigma ** 2 + self.eps)
        uncertainty_loss = (per_sample_unc * weights.detach()).mean()

        total_loss = self.lambda_bce * bce + self.lambda_unc * uncertainty_loss

        entropy = None
        if self.lambda_entropy > 0:
            p_clamped = torch.clamp(p_los, self.eps, 1.0 - self.eps)
            entropy = -(p_clamped * torch.log(p_clamped) + (1.0 - p_clamped) * torch.log(1.0 - p_clamped))
            total_loss = total_loss - self.lambda_entropy * entropy.mean()

        log_sigma_nlos_reg = 0.001 * (log_sigma_nlos ** 2).mean()
        total_loss = total_loss + log_sigma_nlos_reg

        elev_neg_count = 0
        elev_neg_p_los_mean = 0.0
        if elevation is not None and self.lambda_elevation_prior > 0:
            below_horizon = (elevation < 0).float()
            if below_horizon.sum() > 0:
                prior_penalty = torch.relu(p_los.squeeze() - 0.3) * below_horizon
                prior_loss = prior_penalty.mean()
                total_loss = total_loss + self.lambda_elevation_prior * prior_loss
                elev_neg_count = int(below_horizon.sum().item())
                elev_neg_p_los_mean = p_los.squeeze()[below_horizon.bool()].mean().item()

        if return_components:
            los_mask = (nlos_label == 0)
            nlos_mask = (nlos_label == 1)
            p_los_los_avg = p_los[los_mask].mean().item() if los_mask.any() else 0.0
            p_los_nlos_avg = p_los[nlos_mask].mean().item() if nlos_mask.any() else 0.0

            components = {
                'bce': bce.item(),
                'uncertainty': uncertainty_loss.item(),
                'p_los_mean': p_los.mean().item(),
                'p_los_min': p_los.min().item(),
                'p_los_max': p_los.max().item(),
                'p_los_los_avg': p_los_los_avg,
                'p_los_nlos_avg': p_los_nlos_avg,
                'sigma_mean': sigma.mean().item(),
                'log_sigma_nlos_mean': log_sigma_nlos.mean().item(),
            }
            if entropy is not None:
                components['entropy'] = entropy.mean().item()
            if elevation is not None and elev_neg_count > 0:
                components['elev_neg_count'] = elev_neg_count
                components['elev_neg_p_los_mean'] = elev_neg_p_los_mean
            return total_loss, components
        return total_loss


# ============================================================
# Part 3: Training & Evaluation
# ============================================================


class MoGNLLLoss(nn.Module):
    """Mixture of Gaussians NLL Loss."""
    def __init__(self, lambda_entropy=0.03, lambda_elevation_prior=0.1,
                 lambda_mu_reg=0.001, lambda_sigma_reg=0.001, sigma_gap_target=0.5, lambda_sigma_sep=0.1,
                 mu_target=0.15):
        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.lambda_elevation_prior = lambda_elevation_prior
        self.lambda_mu_reg = lambda_mu_reg
        self.lambda_sigma_reg = lambda_sigma_reg
        self.sigma_gap_target = sigma_gap_target
        self.lambda_sigma_sep = lambda_sigma_sep
        self.mu_target = mu_target
        self.eps = 1e-6

    def forward(self, p_los, mu_nlos, log_sigma_los, log_sigma_nlos, pseudorange_error,
                nlos_label, elevation=None, return_components=False):
        import math as _m
        N = p_los.size(0)
        if N == 0:
            loss = torch.tensor(0.0, device=p_los.device)
            return (loss, {}) if return_components else loss
        p_los = p_los.squeeze()
        mu_nlos = mu_nlos.squeeze()
        log_sigma_los = log_sigma_los.squeeze()
        log_sigma_nlos = log_sigma_nlos.squeeze()
        err = pseudorange_error.squeeze()
        p_safe = torch.clamp(p_los, self.eps, 1.0-self.eps)
        sigma_los = torch.clamp(torch.exp(log_sigma_los), 0.01, 50.0)  # widened: was [0.001,5.0]
        sigma_nlos = torch.clamp(torch.exp(log_sigma_nlos), 0.05, 150.0)  # widened: was [0.01,5.0]
        log_prob_los = -0.5*(err/(sigma_los+self.eps))**2 - torch.log(sigma_los+self.eps) - 0.5*_m.log(2*_m.pi)
        log_prob_nlos = -0.5*((err-mu_nlos)/(sigma_nlos+self.eps))**2 - torch.log(sigma_nlos+self.eps) - 0.5*_m.log(2*_m.pi)
        lp_los = torch.log(p_safe) + log_prob_los
        lp_nlos = torch.log(1-p_safe) + log_prob_nlos
        max_log = torch.max(lp_los, lp_nlos)
        log_mix = max_log + torch.log(torch.exp(lp_los-max_log) + torch.exp(lp_nlos-max_log) + self.eps)
        nll = -log_mix.mean()
        total_loss = nll
        if self.lambda_entropy > 0:
            ent = -(p_safe*torch.log(p_safe) + (1-p_safe)*torch.log(1-p_safe))
            total_loss -= self.lambda_entropy * ent.mean()
        total_loss += self.lambda_mu_reg * ((mu_nlos - self.mu_target)**2).mean()
        total_loss += self.lambda_sigma_reg * (log_sigma_nlos**2).mean()
        if elevation is not None and self.lambda_elevation_prior > 0:
            below = (elevation.squeeze() < 0).float()
            if below.sum() > 0:
                total_loss += self.lambda_elevation_prior * torch.relu(p_los-0.3).mul(below).mean()
        # Sigma separation loss: push sigma_nlos(NLOS) >> sigma_nlos(LOS)
        if self.lambda_sigma_sep > 0:
            lm = (nlos_label.squeeze()==0); nm = (nlos_label.squeeze()==1)
            if lm.any() and nm.any():
                gap = sigma_nlos[nm].mean() - sigma_nlos[lm].mean()
                total_loss += self.lambda_sigma_sep * torch.relu(self.sigma_gap_target - gap)
        # Sigma centering: soft pull toward physical ranges
        sigma_center_loss = ((sigma_los - 0.3).pow(2).mean() * 0.10 + (sigma_nlos - 1.5).pow(2).mean() * 0.01)
        total_loss = total_loss + sigma_center_loss

        if return_components:
            lm = (nlos_label.squeeze()==0); nm = (nlos_label.squeeze()==1)
            return total_loss, {
                'nll':nll.item(),'p_los_mean':p_los.mean().item(),
                'p_los_los_avg':p_los[lm].mean().item() if lm.any() else 0,
                'p_los_nlos_avg':p_los[nm].mean().item() if nm.any() else 0,
                'mu_nlos_mean':mu_nlos.mean().item(),'sigma_nlos_mean':sigma_nlos.mean().item()}
        return total_loss

class SupervisedComponentNLLLoss(nn.Module):
    """Train sigma/mu components using ground-truth LOS/NLOS labels.

    Decouples classification (p_los) from distribution fitting.
    LOS samples -> fit zero-mean Gaussian with sigma_los.
    NLOS samples -> fit Gaussian(mu_nlos, sigma_nlos).
    """
    def __init__(self, lambda_mu_reg=0.001, lambda_sigma_reg=0.001,
                 sigma_gap_target=0.3, lambda_sigma_sep=1.0, mu_target=0.15):
        super().__init__()
        self.lambda_mu_reg = lambda_mu_reg
        self.lambda_sigma_reg = lambda_sigma_reg
        self.sigma_gap_target = sigma_gap_target
        self.lambda_sigma_sep = lambda_sigma_sep
        self.mu_target = mu_target
        self.eps = 1e-6

    def forward(self, mu_nlos, log_sigma_los, log_sigma_nlos,
                pseudorange_error, nlos_label, return_components=False):
        import math as _m
        mu_nlos = mu_nlos.squeeze()
        log_sigma_los = log_sigma_los.squeeze()
        log_sigma_nlos = log_sigma_nlos.squeeze()
        err = pseudorange_error.squeeze()
        nl = nlos_label.squeeze()

        sigma_los = torch.clamp(torch.exp(log_sigma_los), 0.01, 50.0)  # widened: was [0.001,5.0]
        sigma_nlos = torch.clamp(torch.exp(log_sigma_nlos), 0.05, 150.0)  # widened: was [0.01,5.0]

        los_mask = (nl == 0)
        nlos_mask = (nl == 1)

        total_loss = torch.tensor(0.0, device=err.device)
        nll_los_val = torch.tensor(0.0, device=err.device)
        nll_nlos_val = torch.tensor(0.0, device=err.device)

        if los_mask.any():
            nll_los_val = (0.5 * (err[los_mask] / (sigma_los[los_mask] + self.eps))**2
                           + torch.log(sigma_los[los_mask] + self.eps)
                           + 0.5 * _m.log(2 * _m.pi)).mean()
            total_loss = total_loss + nll_los_val

        if nlos_mask.any():
            nll_nlos_val = (0.5 * ((err[nlos_mask] - mu_nlos[nlos_mask])
                                   / (sigma_nlos[nlos_mask] + self.eps))**2
                            + torch.log(sigma_nlos[nlos_mask] + self.eps)
                            + 0.5 * _m.log(2 * _m.pi)).mean()
            total_loss = total_loss + nll_nlos_val

        total_loss = total_loss + self.lambda_mu_reg * ((mu_nlos - self.mu_target)**2).mean()
        total_loss = total_loss + self.lambda_sigma_reg * (log_sigma_nlos**2).mean()

        if self.lambda_sigma_sep > 0 and los_mask.any() and nlos_mask.any():
            gap = sigma_nlos[nlos_mask].mean() - sigma_los[los_mask].mean()
            total_loss = total_loss + self.lambda_sigma_sep * torch.relu(self.sigma_gap_target - gap)

        if return_components:
            return total_loss, {
                'nll_los': nll_los_val.item(),
                'nll_nlos': nll_nlos_val.item(),
                'sigma_los_mean': sigma_los[los_mask].mean().item() if los_mask.any() else 0,
                'sigma_nlos_mean': sigma_nlos[nlos_mask].mean().item() if nlos_mask.any() else 0,
                'mu_nlos_mean': mu_nlos[nlos_mask].mean().item() if nlos_mask.any() else 0,
            }
        return total_loss


def _extract_elevation(node_features: torch.Tensor) -> torch.Tensor:
    """Extract raw elevation (degrees) from node features -- dim 0 is elevation/90"""
    return node_features[..., 0] * 90.0



def batch_collate_fn(batch):
    """Block-diagonal collate for variable-size GNSS graphs."""
    nfs, eis, eas, pes, lbs = [], [], [], [], []
    offset = 0
    for nf, ei, ea, pe, lb in batch:
        N = nf.size(0)
        if N == 0:
            continue
        nfs.append(nf)
        eis.append(ei + offset)
        eas.append(ea)
        pes.append(pe)
        lbs.append(lb)
        offset += N
    if len(nfs) == 0:
        return (torch.zeros(0, 11), torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0), torch.zeros(0), torch.zeros(0))
    return (torch.cat(nfs, dim=0), torch.cat(eis, dim=1),
            torch.cat(eas, dim=0), torch.cat(pes, dim=0), torch.cat(lbs, dim=0))

class GNSDataset(Dataset):
    """GNSS Graph Dataset -- wraps EpochData list into PyTorch Dataset"""

    def __init__(self, epochs_data: List, config: Config):
        self.graph_data = []
        threshold = config.AZIMUTH_THRESHOLD

        for epoch in epochs_data:
            if len(epoch.observations) == 0:
                continue

            node_features = extract_node_features(epoch)
            edge_index, edge_attr = build_azimuth_graph(epoch, threshold)
            pr_errors = extract_pseudorange_errors(epoch)
            nlos_labels = extract_labels(epoch)

            assert node_features.shape[1] == FEATURE_DIM, \
                f"Feature dim mismatch: {node_features.shape[1]} != {FEATURE_DIM}"

            self.graph_data.append({
                'node_features': node_features.astype(np.float32),
                'edge_index': edge_index.astype(np.int64),
                'edge_attr': edge_attr.astype(np.float32),
                'pseudorange_error': pr_errors.astype(np.float32),
                'nlos_label': nlos_labels.astype(np.float32),
            })

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        data = self.graph_data[idx]
        return (
            torch.tensor(data['node_features'], dtype=torch.float32),
            torch.tensor(data['edge_index'], dtype=torch.long),
            torch.tensor(data['edge_attr'], dtype=torch.float32),
            torch.tensor(data['pseudorange_error'], dtype=torch.float32),
            torch.tensor(data['nlos_label'], dtype=torch.float32),
        )


def train_epoch(model: nn.Module, dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: Optional[ReduceLROnPlateau],
                loss_fn: nn.Module, device: torch.device,
                epoch: int, gradient_clip: float = 2.0,
                gradient_accumulation: int = 4,
                log_interval: int = 10,
                writer: Optional[SummaryWriter] = None,
                global_step: int = 0,
                scaler: Optional[GradScaler] = None,
                use_amp: bool = False,
                mog_loss_fn: Optional[nn.Module] = None,
                nlos_loss_bce: Optional[nn.Module] = None,
                mog_pure_bce_epochs: int = 20,
                mog_blend_epochs: int = 15,
                 sup_loss_fn: Optional[nn.Module] = None,
                 bce_only_loss: Optional[nn.Module] = None) -> Dict[str, float]:
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_uncertainty = 0.0
    num_batches = 0
    nan_batches = 0
    grad_norms_before = []
    grad_norms_after = []

    use_mog_loss_fn = mog_loss_fn is not None and nlos_loss_bce is not None
    is_pure_bce = use_mog_loss_fn and (epoch < mog_pure_bce_epochs)
    is_blend = use_mog_loss_fn and (mog_pure_bce_epochs <= epoch < mog_pure_bce_epochs + mog_blend_epochs)
    # All heads remain trainable throughout training (no freeze for mu_nlos/sigma heads)
    lam = 0.5 * (1 + math.cos(math.pi * (epoch - mog_pure_bce_epochs) / max(mog_blend_epochs, 1))) if is_blend else 0.0  # cosine schedule
    all_p_los = []
    all_log_sigma = []

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        node_features, edge_index, edge_attr, pseudorange_errors, nlos_labels = batch

        if node_features.size(0) == 0:
            continue

        if node_features.dim() == 3 and node_features.size(0) == 1:
            node_features = node_features.squeeze(0)
        if edge_index.dim() == 3 and edge_index.size(0) == 1:
            edge_index = edge_index.squeeze(0)

        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        pseudorange_errors = pseudorange_errors.to(device)
        nlos_labels = nlos_labels.to(device)

        if edge_index.size(1) == 0:
            N = node_features.size(0)
            edge_index = torch.tensor(
                [[i for i in range(N)], [i for i in range(N)]],
                device=device
            )

        # Forward (AMP)
        if use_amp and scaler is not None:
            with autocast('cuda'):
                p_los, mu_nlos, log_sigma_los, log_sigma_nlos = model(node_features, edge_index)
            p_los = p_los.float()
            mu_nlos = mu_nlos.float()
            log_sigma_nlos = log_sigma_nlos.float()
        else:
            p_los, mu_nlos, log_sigma_los, log_sigma_nlos = model(node_features, edge_index)

        if torch.isnan(p_los).any() or torch.isnan(log_sigma_nlos).any():
            nan_batches += 1
            continue

        elevation_deg = _extract_elevation(node_features)

        if use_mog_loss_fn:
            if is_pure_bce:
                loss, components = nlos_loss_bce(p_los, log_sigma_nlos, pseudorange_errors,
                                    nlos_labels, elevation=elevation_deg, return_components=True)
                # L2 supervision: pull mu_nlos toward 0.15 km + sigma centering during BCE warmup
                mu_reg = 0.05 * (mu_nlos - 0.15).pow(2).mean()
                sigma_warmup_reg = 0.01 * ((torch.exp(log_sigma_los) - 0.3).pow(2).mean()
                                          + (torch.exp(log_sigma_nlos) - 1.5).pow(2).mean())
                loss = loss + mu_reg + sigma_warmup_reg
                components['mu_reg'] = mu_reg.item()
            elif is_blend:
                bce_loss_fn = bce_only_loss if bce_only_loss is not None else nlos_loss_bce
                loss_bce, comps_bce = bce_loss_fn(p_los, log_sigma_nlos, pseudorange_errors,
                                    nlos_labels, elevation=elevation_deg, return_components=True)
                loss_comp, comps_comp = sup_loss_fn(mu_nlos, log_sigma_los, log_sigma_nlos,
                                    pseudorange_errors, nlos_labels, return_components=True)
                # lam: 1.0 -> 0.0, comp_weight: 0.0 -> 1.0 over blend epochs
                comp_weight = 1.0 - lam
                loss = lam * loss_bce + comp_weight * loss_comp
                components = {**comps_bce,
                              'sigma_los': comps_comp.get('sigma_los_mean', 0),
                              'sigma_nlos': comps_comp.get('sigma_nlos_mean', 0),
                              'mu_nlos': comps_comp.get('mu_nlos_mean', 0)}
            else:
                # Pure NLL: detach p_los from NLL (only train sigmas via NLL)
                # p_los is trained separately via BCE
                p_los_detached = p_los.detach()
                loss_nll, components = mog_loss_fn(p_los_detached, mu_nlos, log_sigma_los, log_sigma_nlos,
                                    pseudorange_errors, nlos_labels, elevation=elevation_deg, return_components=True)
                target_los = 1.0 - nlos_labels.squeeze()
                loss_bce = F.binary_cross_entropy(p_los.squeeze(), target_los, reduction='mean')
                loss = loss_nll * 0.5 + loss_bce * 1.5  # BCE 3x weight over NLL
        else:
            try:
                loss, components = loss_fn(p_los, log_sigma_nlos, pseudorange_errors,
                                nlos_labels, elevation=elevation_deg, return_components=True)
            except (TypeError, ValueError):
                loss = loss_fn(p_los, log_sigma_nlos, pseudorange_errors, nlos_labels, elevation=elevation_deg)
                components = {}

        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            continue

        # Gradient accumulation (AMP)
        loss = loss / gradient_accumulation
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % gradient_accumulation == 0:
            # Unscale for AMP before gradient ops
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)

            total_norm_before = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            grad_norms_before.append(total_norm_before)

            for param in model.parameters():
                if param.grad is not None:
                    param.grad.add_(torch.randn_like(param.grad) * 1e-5)

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            total_norm_after = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            grad_norms_after.append(total_norm_after)

            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation
        if components:
            total_bce += components.get('bce', 0.0) * gradient_accumulation
            total_uncertainty += components.get('uncertainty', 0.0) * gradient_accumulation
        num_batches += 1

        all_p_los.append(p_los.detach().cpu().mean().item())
        all_log_sigma.append(log_sigma_nlos.detach().cpu().mean().item())

        if epoch < 1 and batch_idx == 0:
            bce_val = components.get('bce', 0.0) if components else 0.0
            unc_val = components.get('uncertainty', 0.0) if components else 0.0
            sigma = torch.exp(log_sigma_nlos)
            print(f"  Epoch {epoch}, Batch {batch_idx}: "
                  f"loss={loss.item() * gradient_accumulation:.4f} "
                  f"(BCE={bce_val * gradient_accumulation:.4f}, Unc={unc_val * gradient_accumulation:.4f}), "
                  f"p_los=[{p_los.min().item():.3f}, {p_los.max().item():.3f}], "
                  f"sigma=[{sigma.min().item():.3f}, {sigma.max().item():.3f}]")

    # Handle remaining gradients (AMP-aware)
    if (batch_idx + 1) % gradient_accumulation != 0 and num_batches > 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    metrics = {'loss': total_loss / max(num_batches, 1)}
    if total_bce > 0:
        metrics['bce'] = total_bce / max(num_batches, 1)
        metrics['uncertainty'] = total_uncertainty / max(num_batches, 1)
    if grad_norms_before:
        metrics['grad_norm_before'] = np.mean(grad_norms_before)
        metrics['grad_norm_after'] = np.mean(grad_norms_after)
    if nan_batches > 0:
        metrics['nan_batches'] = nan_batches
    if all_p_los:
        metrics['p_los_avg'] = np.mean(all_p_los)
        metrics['log_sigma_nlos_avg'] = np.mean(all_log_sigma)

    if writer is not None:
        try:
            step = global_step + 1
            writer.add_scalar('Train/Loss', metrics['loss'], step)
            if 'bce' in metrics:
                writer.add_scalar('Train/BCE', metrics['bce'], step)
                writer.add_scalar('Train/Uncertainty', metrics['uncertainty'], step)
            if 'grad_norm_before' in metrics:
                writer.add_scalar('Train/GradNorm_Before', metrics['grad_norm_before'], step)
                writer.add_scalar('Train/GradNorm_After', metrics['grad_norm_after'], step)
            if 'p_los_avg' in metrics:
                writer.add_scalar('Train/p_LOS_avg', metrics['p_los_avg'], step)
                writer.add_scalar('Train/log_sigma_nlos_avg', metrics['log_sigma_nlos_avg'], step)
            if 'nan_batches' in metrics:
                writer.add_scalar('Train/NaN_Batches', metrics['nan_batches'], step)
        except Exception:
            pass

    if scheduler is not None:
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(metrics['loss'])
        else:
            scheduler.step()

    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader,
             loss_fn: nn.Module, device: torch.device) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    all_preds = []
    all_labels = []
    all_p_los_vals = []
    all_nlos_labels_vals = []
    all_log_sigma_vals = []
    all_mu_nlos_vals = []
    all_sigma_los_vals_from_model = []
    all_elevations = []
    num_batches = 0

    for batch in dataloader:
        node_features, edge_index, edge_attr, pseudorange_errors, nlos_labels = batch

        if node_features.size(0) == 0:
            continue

        if node_features.dim() == 3 and node_features.size(0) == 1:
            node_features = node_features.squeeze(0)
        if edge_index.dim() == 3 and edge_index.size(0) == 1:
            edge_index = edge_index.squeeze(0)

        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        pseudorange_errors = pseudorange_errors.to(device)
        nlos_labels = nlos_labels.to(device)

        if edge_index.size(1) == 0:
            N = node_features.size(0)
            edge_index = torch.tensor(
                [[i for i in range(N)], [i for i in range(N)]],
                device=device
            )

        p_los, mu_nlos, log_sigma_los, log_sigma_nlos = model(node_features, edge_index)

        if torch.isnan(p_los).any():
            continue

        elevation_deg = _extract_elevation(node_features)

        try:
            loss, comps = loss_fn(p_los, log_sigma_nlos, pseudorange_errors,
                                  nlos_labels, elevation=elevation_deg,
                                  return_components=True)
        except (TypeError, ValueError):
            loss = loss_fn(p_los, log_sigma_nlos, pseudorange_errors, nlos_labels,
                           elevation=elevation_deg)
            comps = {}
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        total_loss += loss.item()
        if comps:
            total_bce += comps.get('bce', 0.0)
        num_batches += 1

        p_los_np = p_los.squeeze().cpu().numpy()
        mu_nlos_np = mu_nlos.squeeze().cpu().numpy()
        log_sigma_nlos_np = log_sigma_nlos.squeeze().cpu().numpy()
        elev_np = elevation_deg.squeeze().cpu().numpy()
        p_nlos_pred = 1.0 - p_los_np
        nlos_lab = nlos_labels.squeeze().cpu().numpy()

        if p_nlos_pred.ndim == 0:
            all_preds.append(float(p_nlos_pred))
            all_labels.append(float(nlos_lab))
            all_p_los_vals.append(float(p_los_np))
            all_nlos_labels_vals.append(float(nlos_lab))
            all_log_sigma_vals.append(float(log_sigma_nlos_np))
            all_elevations.append(float(elev_np))
        else:
            all_preds.extend(p_nlos_pred.flatten().tolist())
            all_labels.extend(nlos_lab.flatten().tolist())
            all_p_los_vals.extend(p_los_np.flatten().tolist())
            all_nlos_labels_vals.extend(nlos_lab.flatten().tolist())
            all_log_sigma_vals.extend(log_sigma_nlos_np.flatten().tolist())
            all_mu_nlos_vals.extend(mu_nlos_np.flatten().tolist())
            all_sigma_los_vals_from_model.extend(torch.exp(log_sigma_los).squeeze().cpu().numpy().flatten().tolist())
            all_elevations.extend(elev_np.flatten().tolist())

    if all_preds and all_labels:
        preds = np.array(all_preds) > 0.5
        labels = np.array(all_labels)

        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))

        accuracy = (tp + tn) / max(len(labels), 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    else:
        accuracy = precision = recall = f1 = 0.0

    metrics = {
        'loss': total_loss / max(num_batches, 1),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    if total_bce > 0:
        metrics['bce'] = total_bce / max(num_batches, 1)

    if all_p_los_vals:
        p_los_arr = np.array(all_p_los_vals)
        nlos_lab_arr = np.array(all_nlos_labels_vals)
        los_mask = nlos_lab_arr == 0
        nlos_mask = nlos_lab_arr == 1
        metrics['p_los_mean'] = float(np.mean(p_los_arr))
        metrics['p_los_min'] = float(np.min(p_los_arr))
        metrics['p_los_max'] = float(np.max(p_los_arr))
        metrics['p_los_los_avg'] = float(np.mean(p_los_arr[los_mask])) if los_mask.any() else 0.0
        metrics['p_los_nlos_avg'] = float(np.mean(p_los_arr[nlos_mask])) if nlos_mask.any() else 0.0
    if all_log_sigma_vals:
        log_sigma_nlos_arr = np.array(all_log_sigma_vals)
        sigma_arr = np.exp(log_sigma_nlos_arr)
        metrics['log_sigma_nlos_mean'] = float(np.mean(log_sigma_nlos_arr))
        metrics['sigma_mean'] = float(np.mean(sigma_arr))
        if len(all_nlos_labels_vals) == len(all_log_sigma_vals):
            nlab = np.array(all_nlos_labels_vals)
            sl = sigma_arr[nlab == 0]; sn = sigma_arr[nlab == 1]
            if len(sl): metrics['sigma_nlos_los'] = float(np.mean(sl))
            if len(sn): metrics['sigma_nlos_nlos'] = float(np.mean(sn))
            if len(sl) and len(sn): metrics['sigma_nlos_gap'] = float(np.mean(sn) - np.mean(sl))
    if all_elevations:
        elev_arr = np.array(all_elevations)
        neg_mask = elev_arr < 0
        if neg_mask.any():
            p_los_arr_all = np.array(all_p_los_vals)
            metrics['elev_neg_count'] = int(neg_mask.sum())
            metrics['elev_neg_p_los_mean'] = float(np.mean(p_los_arr_all[neg_mask]))

    # MoG-specific metrics
    if all_mu_nlos_vals:
        mu_arr = np.array(all_mu_nlos_vals)
        metrics['mu_nlos_mean'] = float(np.mean(mu_arr))
        metrics['mu_nlos_std'] = float(np.std(mu_arr))
        metrics['mu_nlos_max'] = float(np.max(mu_arr))
    if all_sigma_los_vals_from_model:
        sl_arr = np.array(all_sigma_los_vals_from_model)
        sn_arr = np.array(sigma_arr)  # sigma_nlos from log_sigma_nlos head
        metrics['sigma_los_mean'] = float(np.mean(sl_arr))
        metrics['sigma_nlos_from_model_mean'] = float(np.mean(sigma_arr)) if len(all_log_sigma_vals) else 0
        if len(all_nlos_labels_vals) == len(all_sigma_los_vals_from_model):
            nlab = np.array(all_nlos_labels_vals)
            sl_los = sl_arr[nlab == 0]; sl_nlos = sl_arr[nlab == 1]
            if len(sl_los) and len(sl_nlos): 
                sn_nlos_gap = sn_arr[nlab == 1]
                metrics['sigma_sep'] = float(np.mean(sn_nlos_gap) - np.mean(sl_los))

    return metrics


def create_optimizer_and_scheduler(model: nn.Module, config: Config):
    """Create optimizer and learning rate scheduler"""
    # Separate param groups: p_los head gets higher LR for faster classification convergence
    p_los_params = list(model.p_los_head.parameters())
    mu_nlos_params = list(model.mu_nlos_head.parameters())
    sigma_los_params = list(model.log_sigma_los_head.parameters())
    sigma_nlos_params = list(model.log_sigma_nlos_head.parameters())
    excluded_ids = set(id(p) for p in (p_los_params + mu_nlos_params + sigma_los_params + sigma_nlos_params))
    other_params = [p for p in model.parameters() if id(p) not in excluded_ids]
    optimizer = torch.optim.AdamW([
        {'params': p_los_params, 'lr': config.LEARNING_RATE * 4},       # 2e-4
        {'params': mu_nlos_params, 'lr': config.LEARNING_RATE},         # 5e-5
        {'params': sigma_los_params, 'lr': config.LEARNING_RATE},       # 5e-5
        {'params': sigma_nlos_params, 'lr': config.LEARNING_RATE},      # 5e-5
        {'params': other_params, 'lr': config.LEARNING_RATE},           # 5e-5 backbone
    ], weight_decay=1e-3, eps=1e-8)
    scheduler = None
    if config.USE_LR_SCHEDULER:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE, min_lr=config.SCHEDULER_MIN_LR,
        )
    return optimizer, scheduler


# ============================================================
# Part 4: Main Entry
# ============================================================

def main(resume_from: str = None, num_epochs: int = None, dataset_name: str = None, exp_name: str = None):
    """Complete training pipeline entry point

    Args:
        resume_from: checkpoint path, continue training from this checkpoint.
        num_epochs:  override NUM_EPOCHS config.
        dataset_name: override DATASETS config.
    """
    config = get_config()
    if num_epochs is not None:
        config.NUM_EPOCHS = num_epochs
    if dataset_name is not None:
        config.DATASETS = [dataset_name]
    if exp_name is not None:
        config._exp_name = exp_name
    config.ensure_dirs()

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    device = config.get_device()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated, "
              f"{torch.cuda.memory_reserved()/1024**2:.1f} MB reserved")
    print(config)

    from Data_read import load_and_process_dataset
    all_epochs = []
    for ds_name in config.DATASETS:
        epochs = load_and_process_dataset(ds_name, config)
        if epochs:
            all_epochs.extend(epochs)
        print(f"  {ds_name}: {len(epochs)} epochs")

    if not all_epochs:
        print("ERROR: No data loaded!")
        return

    print(f"\nTotal epochs: {len(all_epochs)}")

    num_total = len(all_epochs)
    indices = np.random.permutation(num_total)
    split = int(num_total * (1 - config.VALIDATION_SPLIT))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_epochs_data = [all_epochs[i] for i in train_indices]
    val_epochs_data = [all_epochs[i] for i in val_indices]
    print(f"Train epochs: {len(train_epochs_data)}, Val epochs: {len(val_epochs_data)}")

    train_dataset = GNSDataset(train_epochs_data, config)
    val_dataset = GNSDataset(val_epochs_data, config)

    if config.USE_BLOCK_DIAGONAL:
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                  shuffle=True, num_workers=config.NUM_WORKERS,
                                  pin_memory=True, collate_fn=batch_collate_fn,
                                  drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE * 2,
                                shuffle=False, num_workers=config.VAL_NUM_WORKERS,
                                pin_memory=True, collate_fn=batch_collate_fn,
                                drop_last=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                  num_workers=0, collate_fn=lambda x: x[0])
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=0, collate_fn=lambda x: x[0])

    model = NLOSGAT(
        in_features=config.IN_FEATURES,
        hidden_features=config.HIDDEN_FEATURES,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    nlos_loss_bce = NLOSLoss(
        pos_weight=config.POS_WEIGHT, label_smoothing=config.LABEL_SMOOTHING,
        lambda_bce=config.LAMBDA_BCE, p_los_smoothing=config.P_LOS_SMOOTHING,
        lambda_entropy=config.LAMBDA_ENTROPY, lambda_unc=config.LAMBDA_UNC,
        lambda_elevation_prior=config.LAMBDA_ELEVATION_PRIOR,
    )
    bce_only_loss = NLOSLoss(
        pos_weight=config.POS_WEIGHT, label_smoothing=config.LABEL_SMOOTHING,
        lambda_bce=config.LAMBDA_BCE, p_los_smoothing=config.P_LOS_SMOOTHING,
        lambda_entropy=config.LAMBDA_ENTROPY, lambda_unc=0.0,
        lambda_elevation_prior=config.LAMBDA_ELEVATION_PRIOR,
    ) if config.USE_MIXTURE_GAUSSIAN else None
    loss_fn = nlos_loss_bce
    mog_loss_fn = MoGNLLLoss(
        lambda_entropy=config.LAMBDA_ENTROPY,
        lambda_elevation_prior=config.LAMBDA_ELEVATION_PRIOR,
        lambda_mu_reg=config.LAMBDA_MU_REG,
        lambda_sigma_reg=config.LAMBDA_SIGMA_REG,
        sigma_gap_target=config.SIGMA_GAP_TARGET,
        lambda_sigma_sep=config.LAMBDA_SIGMA_SEP,
        mu_target=config.MU_NLOS_TARGET,
    ) if config.USE_MIXTURE_GAUSSIAN else None
    sup_loss_fn = SupervisedComponentNLLLoss(
        lambda_mu_reg=config.LAMBDA_MU_REG,
        lambda_sigma_reg=config.LAMBDA_SIGMA_REG,
        sigma_gap_target=config.SIGMA_GAP_TARGET,
        lambda_sigma_sep=config.LAMBDA_SIGMA_SEP,
        mu_target=config.MU_NLOS_TARGET,
    ) if config.USE_MIXTURE_GAUSSIAN else None

    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    # AMP scaler
    scaler = GradScaler('cuda') if config.USE_AMP and device.type == 'cuda' else None
    if scaler is not None:
        print("AMP: enabled (automatic mixed precision)")

    # Handle experiment directory and checkpoint resume
    start_epoch = 0
    checkpoint_dir = None
    tensorboard_dir = None

    # Setup experiment directories
    if resume_from and os.path.exists(resume_from):
        exp_dir = os.path.dirname(os.path.dirname(resume_from))
        checkpoint_dir = os.path.dirname(resume_from)
        tensorboard_dir = config.TENSORBOARD_DIR or os.path.join(exp_dir, 'tensorboard')
    elif hasattr(config, '_exp_name') and config._exp_name:
        exp_dir = os.path.join(config.RESULT_DIR, config._exp_name)
        checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        tensorboard_dir = config.TENSORBOARD_DIR or os.path.join(exp_dir, 'tensorboard')
        os.makedirs(checkpoint_dir, exist_ok=True)
        if config.USE_TENSORBOARD:
            os.makedirs(tensorboard_dir, exist_ok=True)

        # Auto-resume from latest checkpoint if available
        if os.path.isdir(checkpoint_dir):
            ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')])
            if ckpts:
                resume_from = os.path.join(checkpoint_dir, ckpts[-1])
                print(f"AUTO-RESUME: found checkpoint {ckpts[-1]}")
    else:
        exp_dirs = sorted([d for d in os.listdir(config.RESULT_DIR)
                           if d.startswith('exp_') and os.path.isdir(os.path.join(config.RESULT_DIR, d))])
        exp_id = len(exp_dirs) + 1
        exp_dir = os.path.join(config.RESULT_DIR, f'exp_{exp_id:03d}')
        checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        tensorboard_dir = config.TENSORBOARD_DIR or os.path.join(exp_dir, 'tensorboard')
        os.makedirs(checkpoint_dir, exist_ok=True)
        if config.USE_TENSORBOARD:
            os.makedirs(tensorboard_dir, exist_ok=True)

    print(f"Experiment dir: {exp_dir}")
    if config.USE_TENSORBOARD:
        print(f"TensorBoard dir: {tensorboard_dir}")

    # Resume from checkpoint (either explicit or auto-detected)
    if resume_from and os.path.exists(resume_from):
        print(f"\n{'='*60}")
        print(f"Resuming from checkpoint: {resume_from}")
        print(f"{'='*60}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"  Loaded checkpoint: epoch={checkpoint['epoch']} (resuming from epoch {start_epoch+1})")
        if 'val_loss' in checkpoint:
            print(f"  Checkpoint val_loss: {checkpoint['val_loss']:.4f}")
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        print(f"  Experiment dir: {exp_dir}")

    writer = None
    if config.USE_TENSORBOARD:
        writer = SummaryWriter(log_dir=tensorboard_dir)
        if start_epoch == 0:
            dummy_x = torch.randn(5, config.IN_FEATURES, device=device)
            dummy_edge = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]], device=device)
            try:
                writer.add_graph(model, (dummy_x, dummy_edge))
            except Exception:
                pass

    best_val_loss = float('inf')
    patience_counter = 0
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']

    total_epochs = config.NUM_EPOCHS
    print(f"\n{'='*60}")
    print(f"Starting training (epoch {start_epoch+1}->{total_epochs}, "
          f"grad_accum={config.GRADIENT_ACCUMULATION})")
    if config.USE_TENSORBOARD:
        print(f"TensorBoard: tensorboard --logdir={tensorboard_dir}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, total_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            epoch=epoch, gradient_clip=config.GRADIENT_CLIP,
            gradient_accumulation=config.GRADIENT_ACCUMULATION,
            log_interval=config.LOG_INTERVAL,
            writer=writer, global_step=epoch,
            scaler=scaler, use_amp=config.USE_AMP,
            mog_loss_fn=mog_loss_fn, nlos_loss_bce=nlos_loss_bce,
            mog_pure_bce_epochs=config.MOG_PURE_BCE_EPOCHS,
            mog_blend_epochs=config.MOG_BLEND_EPOCHS,
            sup_loss_fn=sup_loss_fn,
            bce_only_loss=bce_only_loss,
        )

        val_metrics = evaluate(model, val_loader, loss_fn, device)

        lr = optimizer.param_groups[0]['lr']
        train_loss = train_metrics['loss']
        train_bce = train_metrics.get('bce', 0.0)
        train_unc = train_metrics.get('uncertainty', 0.0)
        val_bce = val_metrics.get('bce', 0.0)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
              f"Train: Loss={train_loss:.3f} BCE={train_bce:.4f} Unc={train_unc:.4f} | "
              f"Val: Loss={val_metrics['loss']:.3f} | "
              f"Acc={val_metrics['accuracy']:.3f} F1={val_metrics['f1']:.3f} | "
              f"LR: {lr:.2e}")
        if 'p_los_mean' in val_metrics:
            sigma_info = ""
            if 'sigma_mean' in val_metrics:
                sigma_info = f" sigma={val_metrics['sigma_mean']:.3f}"
            print(f"  p_los: mean={val_metrics['p_los_mean']:.3f} min={val_metrics['p_los_min']:.3f} max={val_metrics['p_los_max']:.3f} | "
                  f"LOS_avg={val_metrics['p_los_los_avg']:.3f} NLOS_avg={val_metrics['p_los_nlos_avg']:.3f}{sigma_info}")
            if 'elev_neg_count' in val_metrics:
                print(f"  elev<0: n={val_metrics['elev_neg_count']} p_los={val_metrics['elev_neg_p_los_mean']:.3f}")
            if val_metrics['p_los_min'] < 0.15:
                print(f"  *** WARNING: p_los min={val_metrics['p_los_min']:.3f} < 0.15, "
                      f"model may be collapsing toward NLOS ***")
            if val_metrics['p_los_max'] > 0.85:
                print(f"  *** WARNING: p_los max={val_metrics['p_los_max']:.3f} > 0.85, "
                      f"model may be collapsing toward LOS ***")
            if 'mu_nlos_mean' in val_metrics:
                mog_parts = [f"mu_nlos=[{val_metrics['mu_nlos_mean']:.4f}"]
                if 'sigma_los_mean' in val_metrics:
                    mog_parts.append(f"sigma_los={val_metrics['sigma_los_mean']:.4f}")
                if 'sigma_sep' in val_metrics:
                    mog_parts.append(f"sigma_sep={val_metrics['sigma_sep']:.4f}")
                mog_parts.append("]")
                print(f"  MoG: " + " ".join(mog_parts))

        if writer is not None:
            try:
                step = epoch + 1
                writer.add_scalar('Val/Loss', val_metrics['loss'], step)
                if 'bce' in val_metrics:
                    writer.add_scalar('Val/BCE', val_metrics['bce'], step)
                writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], step)
                writer.add_scalar('Val/F1', val_metrics['f1'], step)
                writer.add_scalar('Val/Precision', val_metrics['precision'], step)
                writer.add_scalar('Val/Recall', val_metrics['recall'], step)
                if 'p_los_mean' in val_metrics:
                    writer.add_scalar('Val/p_LOS_mean', val_metrics['p_los_mean'], step)
                    writer.add_scalar('Val/p_LOS_min', val_metrics['p_los_min'], step)
                    writer.add_scalar('Val/p_LOS_LOS_avg', val_metrics['p_los_los_avg'], step)
                    writer.add_scalar('Val/p_LOS_NLOS_avg', val_metrics['p_los_nlos_avg'], step)
                if 'sigma_mean' in val_metrics:
                    writer.add_scalar('Val/sigma_mean', val_metrics['sigma_mean'], step)
                if 'elev_neg_p_los_mean' in val_metrics:
                    writer.add_scalar('Val/elev_neg_p_los_mean', val_metrics['elev_neg_p_los_mean'], step)
                    if 'mu_nlos_mean' in val_metrics:
                        writer.add_scalar('Val/mu_nlos_mean', val_metrics['mu_nlos_mean'], step)
                    if 'sigma_los_mean' in val_metrics:
                        writer.add_scalar('Val/sigma_los_mean', val_metrics['sigma_los_mean'], step)
                    if 'sigma_sep' in val_metrics:
                        writer.add_scalar('Val/sigma_sep', val_metrics['sigma_sep'], step)
                writer.add_scalar('LR', lr, step)

                if (epoch + 1) % config.LOG_HISTOGRAM_EPOCHS == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, step)
                        writer.add_histogram(f'Weights/{name}', param, step)
            except Exception:
                pass

        # R3: Composite metric (F1*0.7 + sigma_sep*0.3) for MoG best_model selection
        if config.USE_MIXTURE_GAUSSIAN:
            sigma_sep = val_metrics.get('sigma_sep', 0)
            val_metric_for_best = -val_metrics['f1']  # P0 fix: F1-only selection (was composite F1*0.7+sigma_sep*0.3)
        else:
            val_metric_for_best = val_metrics['loss']
        if val_metric_for_best < best_val_loss:
            best_val_loss = val_metric_for_best
            patience_counter = 0
            _safe_torch_save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
            }, os.path.join(exp_dir, 'best_model.pth'))
        else:
            patience_counter += 1

        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            _safe_torch_save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        mog_patience = max(config.EARLY_STOPPING_PATIENCE, 60) if config.USE_MIXTURE_GAUSSIAN else config.EARLY_STOPPING_PATIENCE
        if patience_counter >= mog_patience:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(no improvement for {config.EARLY_STOPPING_PATIENCE} epochs)")
            break

    _safe_torch_save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, os.path.join(exp_dir, 'final_model.pth'))

    if writer is not None:
        writer.close()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Results saved to: {exp_dir}")
    if config.USE_TENSORBOARD:
        print(f"TensorBoard logs: {tensorboard_dir}")
        print(f"  Run: tensorboard --logdir={tensorboard_dir}")


if __name__ == '__main__':
    main()

