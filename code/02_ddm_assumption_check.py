# =============================================================================
# Step 2. Can a standard DDM reproduce the response-time distribution? (Method: "Checks of DDM assumptions";
# Results: "A standard DDM fits only the late responses").
#
# Maximum-likelihood fit of the DDM (Navarro & Fuss, 2009; accuracy coding; z = a/2) to each session
# (child x level), in two ways:
#   all      all responses, shifted by +500 ms so that all are positive (the shift is absorbed by t)
#   cut150   responses >= 150 ms after the cue only, with a 2% uniform contaminant over 150-500 ms and,
#            as is common practice, no correction for the removed responses
# Sessions with fewer than 40 responses or fewer than 3 errors are skipped. For each session, 20,000
# responses are simulated from the fitted parameters, those inside the window of the observed data are
# kept, and the .1-.9 quantiles of correct responses are compared.
#
#   python code/02_ddm_assumption_check.py
# Output: results/ddm_check_all.csv, results/ddm_check_cut150.csv, results/ddm_check_summary.json
# =============================================================================
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skew

from config import CUT, RESULTS
from data_prep import load_trials

RT_SHIFT = 0.5
K = np.arange(1, 15)
KS = np.arange(-7, 8)
QS = [0.1, 0.3, 0.5, 0.7, 0.9]


def fstd(u, w):
    """Standardised first-passage density at the lower boundary."""
    u = np.maximum(u, 1e-10)
    small = (np.sum((w + 2 * KS[:, None]) * np.exp(-(w + 2 * KS[:, None]) ** 2 / (2 * u)), 0)
             / np.sqrt(2 * np.pi * u ** 3))
    large = np.pi * np.sum(K[:, None] * np.exp(-(K[:, None] ** 2) * np.pi ** 2 * u / 2)
                           * np.sin(K[:, None] * np.pi * w), 0)
    return np.where(u < 1, small, large)


def ddm_pdf(rt, resp, v, a, t, w=0.5):
    dt = rt - t
    out = np.zeros_like(rt)
    ok = dt > 0
    vv = np.where(resp == 1, -v, v)          # upper boundary (correct): v -> -v, w -> 1 - w
    ww = np.where(resp == 1, 1 - w, w)
    d = dt[ok]
    out[ok] = (np.exp(-vv[ok] * a * ww[ok] - vv[ok] ** 2 * d / 2) / a ** 2) * fstd(d / a ** 2, ww[ok])
    return np.maximum(out, 0)


def fit_run(rt, resp, lo, hi):
    """Maximum-likelihood v, a, t (Nelder-Mead from 6 starting points), 2% uniform contaminant on [lo, hi]."""
    def nll(p):
        v, a, t = p[0], np.exp(p[1]), p[2]
        if t < 0 or t >= rt.min() + 0.05:
            return 1e10
        dens = 0.98 * ddm_pdf(rt, resp, v, a, t) + 0.02 * 0.5 / (hi - lo)
        return -np.sum(np.log(dens + 1e-300))
    best = None
    for v0 in (1.0, 3.0, 6.0):
        for a0 in (np.log(0.8), np.log(1.5)):
            r = minimize(nll, [v0, a0, max(rt.min() - 0.08, 0.0)], method="Nelder-Mead",
                         options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-4})
            if best is None or r.fun < best.fun:
                best = r
    return best.x[0], np.exp(best.x[1]), best.x[2]


def simulate(v, a, t, n=20000, dt=0.0005, tmax=3.0, rng=np.random.default_rng(1)):
    """Euler simulation of the diffusion process (the default generator is shared across calls)."""
    x = np.full(n, a / 2); rt = np.full(n, np.nan); resp = np.zeros(n)
    alive = np.ones(n, bool); s = 0
    while alive.any() and s * dt < tmax:
        s += 1
        x[alive] += v * dt + np.sqrt(dt) * rng.standard_normal(alive.sum())
        up = alive & (x >= a); lo = alive & (x <= 0)
        rt[up | lo] = s * dt + t; resp[up] = 1; resp[lo] = -1
        alive &= ~(up | lo)
    return rt, resp


def check(df, label, lo, hi):
    rows = []
    for (s, sess), g in df.groupby(["subj", "sessione"]):
        rt, resp = g["rt"].to_numpy(), g["response"].to_numpy()
        if len(rt) < 40 or (resp == -1).sum() < 3:
            continue
        v, a, t = fit_run(rt, resp, lo, hi)
        srt, sresp = simulate(v, a, t)
        keep = (srt >= lo) & (srt <= hi)       # same window as the observed data
        srt, sresp = srt[keep], sresp[keep]
        oq = np.quantile(rt[resp == 1], QS); pq = np.quantile(srt[sresp == 1], QS)
        rows.append({"subj": s, "sess": sess, "v": v, "a": a, "t": t,
                     "acc_obs": (resp == 1).mean(), "acc_pred": (sresp == 1).mean(),
                     "skew_obs": skew(rt[resp == 1]), "skew_pred": skew(srt[sresp == 1]),
                     **{f"q{int(q*100)}_err_ms": 1000 * (p - o) for q, o, p in zip(QS, oq, pq)}})
    r = pd.DataFrame(rows)
    print(f"\n=== {label}: {len(r)} sessions ===")
    print(r[["v", "a", "t", "acc_obs", "acc_pred", "skew_obs", "skew_pred"]].describe().round(3).loc[["mean", "50%", "std"]])
    qcols = [c for c in r if c.endswith("_err_ms")]
    print("Error on the quantiles of correct RTs (predicted - observed, ms): mean | median |error|")
    print(pd.DataFrame({"mean": r[qcols].mean(), "median_abs": r[qcols].abs().median()}).round(1))
    r.to_csv(RESULTS / f"ddm_check_{label}.csv", index=False)
    return {"sessions": len(r), "skew_obs_mean": float(r.skew_obs.mean()), "skew_pred_mean": float(r.skew_pred.mean()),
            "median_abs_quantile_error_ms": {c[:-7]: float(r[c].abs().median()) for c in qcols}}


if __name__ == "__main__":
    df = load_trials()
    d = df[df.omission == 0].copy()
    d["response"] = np.where(d.correct == 1, 1, -1)
    A = d.assign(rt=d.rt_raw + RT_SHIFT)
    B = d[d.rt_raw >= CUT].assign(rt=lambda x: x.rt_raw)
    out = {"all": check(A, "all", lo=-np.inf, hi=np.inf),
           "cut150": check(B, "cut150", lo=CUT, hi=0.5)}
    json.dump(out, open(RESULTS / "ddm_check_summary.json", "w"), indent=1)
