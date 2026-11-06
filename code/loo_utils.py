# =============================================================================
# PSIS-LOO (Vehtari et al., 2017) on a saved pointwise log-likelihood (chains x draws x observations).
# =============================================================================
import warnings
from types import SimpleNamespace

import arviz as az
import numpy as np
from scipy.special import logsumexp

warnings.filterwarnings("ignore")


def loo_safe(ll):
    """PSIS-LOO with ArviZ. Observations whose log-likelihood is constant across the lowest draws
    (explained by the contaminant only: flat importance ratios, which PSIS cannot handle) contribute
    their plain importance-sampling estimate, with k = 0. Returns elpd_i, pareto_k, elpd, se, p (p_loo)."""
    flat = ll.reshape(-1, ll.shape[-1])
    top = np.sort(flat, axis=0)[:max(5, flat.shape[0] // 5)]
    const = (top.max(axis=0) - top.min(axis=0)) < 1e-6
    elpd_i = np.empty(ll.shape[-1])
    k = np.zeros(ll.shape[-1])
    elpd_i[const] = -(logsumexp(-flat[:, const], axis=0) - np.log(flat.shape[0]))
    # infinitesimal noise (1e-6) breaks ties in the tail; negligible effect on elpd
    sub = ll[..., ~const].astype(np.float64) + 1e-6 * np.random.default_rng(0).standard_normal(ll[..., ~const].shape)
    lo = az.loo(az.from_dict({"posterior": {"mu": np.zeros(sub.shape[:2])}, "log_likelihood": {"y": sub}}), pointwise=True)
    elpd_i[~const] = np.asarray(lo.elpd_i).reshape(-1)
    k[~const] = np.asarray(lo.pareto_k).reshape(-1)
    return SimpleNamespace(elpd_i=elpd_i, pareto_k=k, good_k=float(lo.good_k), elpd=float(elpd_i.sum()),
                           se=float(np.sqrt(len(elpd_i) * np.var(elpd_i))), p=float(lo.p), n_const=int(const.sum()))


def diff(a, b):
    """elpd difference a - b and its standard error (paired over observations)."""
    d = a.elpd_i - b.elpd_i
    return float(d.sum()), float(np.sqrt(len(d) * np.var(d)))
