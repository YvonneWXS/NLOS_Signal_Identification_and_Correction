# module2_localization/platt_calibration.py — Platt scaling for p_los calibration
import numpy as np
from scipy.optimize import minimize
from .base import LocalizationBase
from .factory import LocalizationFactory


def fit_platt(p_raw, labels):
    """Fit Platt scaling: p_cal = sigmoid(A * logit(p_raw) + B)."""
    p = np.clip(np.asarray(p_raw).flatten(), 1e-8, 1 - 1e-8)
    y = np.asarray(labels).flatten()
    logits = np.log(p / (1 - p))
    
    def bce(params):
        A, B = params
        z = A * logits + B
        pred = 1.0 / (1.0 + np.exp(-z))
        return -np.mean(y * np.log(pred + 1e-8) + (1 - y) * np.log(1 - pred + 1e-8))
    
    result = minimize(bce, [1.0, 0.0], method="Nelder-Mead")
    return {"A": float(result.x[0]), "B": float(result.x[1])}


def apply_platt(p_raw, params):
    """Apply Platt scaling."""
    p = np.clip(np.asarray(p_raw).flatten(), 1e-8, 1 - 1e-8)
    logits = np.log(p / (1 - p))
    z = params["A"] * logits + params["B"]
    return 1.0 / (1.0 + np.exp(-z))
