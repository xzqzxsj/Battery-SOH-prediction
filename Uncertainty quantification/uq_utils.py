import math
from statistics import NormalDist
from typing import Dict, Optional
import numpy as np

from MAgent_UQ.methods import (
    trailing_window,
    aci,
    aci_clipped,
    OGD,
    SF_OGD,
    decay_OGD,
    quantile,
    quantile_integrator_log,
    quantile_integrator_log_scorecaster,
    ECI,
    ECI_cutoff,
    ECI_integral,
    full_smoothed_eci,
    smoothed_ogd,
)


NORMAL = NormalDist(mu=0.0, sigma=1.0)

# 绝对残差得分
def absolute_residual_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y_true) - np.asarray(y_pred))


METHODS = {
    "Trail": trailing_window,
    "ACI": aci,
    "ACI_clipped": aci_clipped,
    "OGD": OGD,
    "SF_OGD": SF_OGD,
    "decay_OGD": decay_OGD,
    "Quantile": quantile,
    "Quantile+Integrator(log)": quantile_integrator_log,
    "Quantile+Integrator(log)+Scorecaster": quantile_integrator_log_scorecaster,
    "ECI": ECI,
    "ECI_cutoff": ECI_cutoff,
    "ECI_integral": ECI_integral,
    "full_smoothed_eci": full_smoothed_eci,
    "smoothed_ogd": smoothed_ogd,
}


def run_uq_method(
    scores: np.ndarray,
    method_name: str = "Quantile+Integrator(log)",
    alpha: float = 0.1,
    lr: float = 0.01,
    ahead: int = 1,
    T_burnin: int = 30,
    method_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    调用 ICLR 代码中的在线 UQ 方法，输出每一步的 q_t（区间半宽）。
    """
    if method_kwargs is None:
        method_kwargs = {}

    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) == 0:
        raise ValueError("scores 为空，无法运行 UQ 方法")

    if method_name not in METHODS:
        raise ValueError(f"Unsupported UQ method: {method_name}")

    fn = METHODS[method_name]

    if method_name == "Trail":
        params = dict(
            scores=scores,
            alpha=alpha,
            lr=None,
            weight_length=method_kwargs.get("weight_length", max(30, T_burnin)),
            ahead=ahead,
        )
        result = fn(**params)
    elif method_name in ["ACI", "ACI_clipped"]:
        params = dict(
            scores=scores,
            alpha=alpha,
            lr=lr,
            window_length=method_kwargs.get("window_length", max(50, T_burnin)),
            T_burnin=T_burnin,
            ahead=ahead,
        )
        result = fn(**params)
    elif method_name in ["ECI", "ECI_cutoff", "ECI_integral", "full_smoothed_eci", "smoothed_ogd"]:
        params = dict(
            scores=scores,
            alpha=alpha,
            lr=lr,
            T_burnin=T_burnin,
            ahead=ahead,
            proportional_lr=method_kwargs.get("proportional_lr", True),
        )
        result = fn(**params)
    elif method_name in ["OGD", "SF_OGD", "decay_OGD", "Quantile"]:
        params = dict(
            scores=scores,
            alpha=alpha,
            lr=lr,
            ahead=ahead,
            T_burnin=T_burnin,
            proportional_lr=method_kwargs.get("proportional_lr", True),
        )
        result = fn(**params)
    elif method_name == "Quantile+Integrator(log)":
        params = dict(
            scores=scores,
            alpha=alpha,
            lr=lr,
            Csat=method_kwargs.get("Csat", 3.0),
            KI=method_kwargs.get("KI", 1.0),
            ahead=ahead,
            T_burnin=T_burnin,
            proportional_lr=method_kwargs.get("proportional_lr", True),
        )
        result = fn(**params)
    elif method_name == "Quantile+Integrator(log)+Scorecaster":
        params = dict(
            scores=scores,
            alpha=alpha,
            lr=lr,
            data=method_kwargs.get("data", None),
            T_burnin=T_burnin,
            Csat=method_kwargs.get("Csat", 3.0),
            KI=method_kwargs.get("KI", 1.0),
            upper=method_kwargs.get("upper", True),
            ahead=ahead,
            integrate=method_kwargs.get("integrate", True),
            proportional_lr=method_kwargs.get("proportional_lr", True),
            scorecast=method_kwargs.get("scorecast", False),
            config_name=method_kwargs.get("config_name", "battery_uq"),
            seasonal_period=method_kwargs.get("seasonal_period", 1),
        )
        result = fn(**params)
    else:
        raise ValueError(f"Unsupported UQ method: {method_name}")

    q = np.asarray(result["q"], dtype=float).reshape(-1)
    finite_q = q[np.isfinite(q)]
    if len(finite_q) == 0:
        fallback = float(np.quantile(scores, 1 - alpha))
    else:
        fallback = float(np.median(finite_q))
    q[~np.isfinite(q)] = fallback
    q = np.maximum(q, 0.0)
    result["q"] = q
    return result


# =========================
# Interval / distribution metrics
# =========================

def interval_width(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.asarray(upper) - np.asarray(lower)



def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    covered = ((y_true >= lower) & (y_true <= upper)).astype(float)
    return float(covered.mean())



def coverage_percent(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return 100.0 * picp(y_true, lower, upper)



def average_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(interval_width(lower, upper)))



def median_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.median(interval_width(lower, upper)))



def mpicd(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    center = 0.5 * (np.asarray(lower) + np.asarray(upper))
    return float(np.mean(np.abs(center - np.asarray(y_true))))



def gaussian_sigma_from_q(q: np.ndarray, alpha: float) -> np.ndarray:
    z = NORMAL.inv_cdf(1 - alpha / 2.0)
    z = max(z, 1e-8)
    sigma = np.asarray(q, dtype=float) / z
    sigma = np.maximum(sigma, 1e-8)
    return sigma



def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    闭式 Gaussian CRPS。
    CRPS(N(mu, sigma), y) = sigma * [ z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi) ]
    """
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-8)
    z = (y_true - mu) / sigma

    cdf = np.vectorize(NORMAL.cdf)(z)
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))



def average_crps_from_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q: np.ndarray,
    alpha: float,
) -> float:
    sigma = gaussian_sigma_from_q(q, alpha)
    crps = crps_gaussian(y_true, y_pred, sigma)
    return float(np.mean(crps))



def compute_uq_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    q: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    metrics = {
        "Coverage(%)": coverage_percent(y_true, lower, upper),
        "Average width": average_width(lower, upper),
        "Median width": median_width(lower, upper),
        "CRPS": average_crps_from_interval(y_true, y_pred, q, alpha),
        "PICP": picp(y_true, lower, upper),
        "MPICD": mpicd(y_true, lower, upper),
    }
    return metrics
