# common/metrics.py — Evaluation metrics for GNSS positioning
import numpy as np
from scipy import stats as scipy_stats


def cep(errors, pct=50):
    """Circular Error Probable: radius containing pct% of errors (km)"""
    errors = np.asarray(errors).flatten()
    return float(np.percentile(errors, pct))


def cep50(errors):
    return cep(errors, 50)


def cep95(errors):
    return cep(errors, 95)


def rmse(errors):
    errors = np.asarray(errors).flatten()
    return float(np.sqrt(np.mean(errors**2)))


def mae(errors):
    errors = np.asarray(errors).flatten()
    return float(np.mean(np.abs(errors)))


def median_error(errors):
    return float(np.median(np.asarray(errors).flatten()))


def std_error(errors):
    return float(np.std(np.asarray(errors).flatten()))


def compute_horizontal_error(gt_ecef_km, pred_ecef_km):
    """Horizontal error in km (2D distance ignoring height)"""
    from .coordinate import ecef_to_enu
    enu = ecef_to_enu(gt_ecef_km, pred_ecef_km)
    if enu.ndim == 1:
        return float(np.sqrt(enu[0]**2 + enu[1]**2))
    return np.sqrt(enu[:, 0]**2 + enu[:, 1]**2)


def compute_3d_error(gt_ecef_km, pred_ecef_km):
    """3D error in km"""
    return np.linalg.norm(np.asarray(pred_ecef_km) - np.asarray(gt_ecef_km), axis=-1)


def all_metrics(errors):
    """Return dict of all standard metrics"""
    errors = np.asarray(errors).flatten()
    return {
        'cep50': float(np.percentile(errors, 50)),
        'cep95': float(np.percentile(errors, 95)),
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'mae': float(np.mean(np.abs(errors))),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'mean': float(np.mean(errors)),
        'n_samples': len(errors),
    }


def wilcoxon_test(errors_a, errors_b):
    """Wilcoxon signed-rank test between two sets of errors.
    Returns p-value: p < 0.05 means statistically significant difference."""
    _, p = scipy_stats.wilcoxon(
        np.asarray(errors_a).flatten(),
        np.asarray(errors_b).flatten()
    )
    return float(p)
