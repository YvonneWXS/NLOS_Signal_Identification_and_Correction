"""fusion/prnc.py ? Pseudorange Residual NLOS Correction (v5)
==============================================================
Core concept: correct NLOS pseudorange bias via residuals and
Module 1 p_los gating, while keeping ALL satellites at uniform
weight in the final LS solve (no DOP degradation).

Methods:
  solve_basic    ? residual-based correction (gate: p_los<0.6, res>2*sigma)
  solve_mu       ? direct mu_nlos correction (simplest, most robust)
  solve_adaptive ? two-stage (hard + soft) with CNO-aware noise floor
  solve_with_tcn ? adaptive + TCN prior blending
"""

import numpy as np
from baselines import solve_standard_ls


class PRNCPositioner:
    """Pseudorange Residual NLOS Correction ? uniform weights, DOP-preserving."""

    def solve_basic(self, pr_mes, sv_positions, p_los, sigma_los, max_iters=5):
        """Basic residual-based correction. Gate: p_los < 0.6 AND residual > 2*sigma."""
        p_nlos = 1.0 - p_los
        nf = 2.0 * sigma_los
        x = solve_standard_ls(sv_positions, pr_mes)

        for _ in range(max_iters):
            dists = np.linalg.norm(sv_positions - x[:3], axis=1)
            residuals = pr_mes - dists - x[3]
            gate = (p_los < 0.6) & (residuals > nf)
            corr = np.where(gate, p_nlos * (residuals - nf), 0.0)
            pr_c = pr_mes - corr
            x_n = solve_standard_ls(sv_positions, pr_c)
            if np.linalg.norm((x_n - x)[:3]) * 1000 < 1.0:
                break
            x = x_n

        diag = {"num_corrected": int(gate.sum()),
                "mean_corr_km": float(corr[corr > 0].mean()) if (corr > 0).any() else 0.0}
        return x, diag

    def solve_mu(self, pr_mes, sv_positions, p_los, mu_nlos):
        """Direct mu_nlos correction: pr_corrected = pr - (1-p_los) * mu_nlos."""
        p_nlos = 1.0 - np.clip(p_los, 0.0, 1.0)
        corr = p_nlos * mu_nlos
        x = solve_standard_ls(sv_positions, pr_mes - corr)
        diag = {"mean_corr_km": float(np.mean(corr))}
        return x, diag

    def solve_adaptive(self, pr_mes, sv_positions, p_los, sigma_los,
                       mu_nlos=None, cno=None, max_iters=7):
        """Adaptive two-stage PRNC with CNO-aware noise floor."""
        p_nlos = 1.0 - np.clip(p_los, 0.0, 1.0)
        if cno is not None:
            cn = np.clip(np.array(cno) / 45.0, 0.3, 1.0)
            nf = sigma_los * (1.0 + 2.0 * (1.0 - cn))
        else:
            nf = 2.0 * sigma_los

        x = solve_standard_ls(sv_positions, pr_mes)
        for _ in range(max_iters):
            dists = np.linalg.norm(sv_positions - x[:3], axis=1)
            res = pr_mes - dists - x[3]

            # Stage 1: high-confidence NLOS
            gate_h = (p_los < 0.3) & (res > nf)
            corr_h = np.where(gate_h, res - nf, 0.0)

            # Stage 2: soft correction for ambiguous
            gate_s = (p_los >= 0.3) & (p_los < 0.6) & (res > nf)
            sw = (0.6 - p_los) / 0.3
            corr_s = np.where(gate_s, sw * (res - nf) * 0.5, 0.0)

            total = corr_h + corr_s
            if mu_nlos is not None:
                total += p_nlos * mu_nlos * 0.3

            x_n = solve_standard_ls(sv_positions, pr_mes - total)
            if np.linalg.norm((x_n - x)[:3]) * 1000 < 0.5:
                break
            x = x_n

        return x, {}

    def solve_with_tcn(self, pr_mes, sv_positions, p_los, sigma_los, mu_nlos,
                        tcn_p_nlos, cno=None):
        """PRNC-adaptive with TCN prior blending."""
        conf = 2.0 * np.abs(tcn_p_nlos[:len(p_los)] - 0.5)
        alpha = np.clip(conf * np.abs(tcn_p_nlos[:len(p_los)] - 0.5) * 2, 0, 0.25)
        pb = (1 - alpha) * np.clip(p_los, 0, 1) + alpha * (1 - tcn_p_nlos[:len(p_los)])
        disagree = ((tcn_p_nlos[:len(p_los)] > 0.6) & (p_los > 0.5)) |                    ((tcn_p_nlos[:len(p_los)] < 0.4) & (p_los < 0.5))
        pf = np.where(disagree, pb, p_los)
        return self.solve_adaptive(pr_mes, sv_positions, pf, sigma_los, mu_nlos, cno)


print("fusion/prnc.py loaded (PRNC ? uniform weights, DOP-preserving)")


import numpy as np

def solve_prnc_mu_corrected(pr_mes, sv_positions, p_los, sigma_los, mu_nlos, max_iter=7):
    """[v7] PRNC-mu with direction-corrected mu_nlos from exp_044-047.

    Safety gate: if mu_nlos direction is still wrong, fall back to Standard LS.
    Otherwise, subtract p_nlos * mu_nlos and run WLS with corrected measurements.
    """
    p_nlos = 1.0 - p_los

    # Safety gate: check mu_nlos direction
    los_mask = p_los > 0.7
    nlos_mask = p_los < 0.3
    if los_mask.sum() > 0 and nlos_mask.sum() > 0:
        if mu_nlos[los_mask].mean() > mu_nlos[nlos_mask].mean():
            from baselines import solve_standard_ls
            return solve_standard_ls(sv_positions, pr_mes)

    correction = p_nlos * mu_nlos
    pr_corrected = pr_mes - correction

    x = np.zeros(4)
    x[2] = 6371.0
    x[3] = np.median(pr_corrected - np.linalg.norm(sv_positions - x[:3], axis=1))

    for _ in range(max_iter):
        dist = np.linalg.norm(sv_positions - x[:3], axis=1)
        h = dist + x[3]
        residuals = pr_corrected - h
        los_vecs = (sv_positions - x[:3]) / dist[:, np.newaxis]
        H = np.hstack([-los_vecs, np.ones((len(pr_mes), 1))])
        try:
            delta = np.linalg.lstsq(H, residuals, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        x += delta
        if np.linalg.norm(delta[:3]) * 1000 < 0.1:
            break

    return x

