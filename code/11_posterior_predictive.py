# =============================================================================
# Step 11. Analytic posterior predictive checks by condition (obstacle type x level x session) for
# Models 1-3 (Results: Model 1 paragraph on predicted accuracy and quantiles; Figure 4).
#
# For each model and each of 40 posterior draws, the joint density p(rt, correct) is computed on a 5-ms
# grid for every combination child x session x cell x obstacle type; predictions for a condition weight
# the combinations by their number of observed responses. Model 3 uses the chains of its best mode
# (from results/model_comparison.json, written by 09_model_comparison.py).
#
#   python code/11_posterior_predictive.py
# Output: results/ppc_conditions_<model>.csv, results/ppc_bins_<model>.csv, results/ppc_summary.json,
#         results/fits/ppc_density_<model>.npy, figures/figure4_model_comparison.png
# =============================================================================
import json

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax.scipy.special import expit, ndtr

import mixture as MD
import models as C
from config import FIGURES, FITS, RESULTS, RT_HI, RT_LO

GRID = np.round(np.arange(RT_LO, RT_HI + 1e-9, 0.005), 4)          # 161 points
EDGES = np.linspace(RT_LO, RT_HI, 33)                               # 25-ms bins
QS = (0.1, 0.3, 0.5, 0.7, 0.9)
N_DRAWS = 40
MODELS = [("model1", "Model 1\ntimed-diffusion mixture"), ("model2", "Model 2\nfast-guess mixture"),
          ("model3", "Model 3\ntwo-stage mixture")]


def joint_density(m, S, D, lapse=C.LAPSE):
    """(n_combinations, n_grid, 2): density of incorrect (0) and correct (1) responses on the grid."""
    P = C.params(m, {k: jnp.asarray(v) for k, v in S.items()}, D)
    g = jnp.asarray(GRID)[None, :]
    n = P["v"].shape[0]
    out = []
    for cor in (0, 1):
        cc = jnp.full((n, GRID.size), cor)
        v, a, t = (jnp.broadcast_to(P[k][:, None], (n, GRID.size)) for k in ("v", "a", "t"))
        mu, sd = P["mu"][:, None], P["sdA"][:, None]
        pA = expit(P["pA"])[:, None]
        pi = expit(P["pi"])[:, None]
        if m in ("model1", "model2"):
            Z = jnp.maximum(MD.ddm_mass_window(P["v"], P["a"], P["t"]), 1e-12)[:, None]
            fR = MD.ddm_pdf((g - t).reshape(-1), cc.reshape(-1), v.reshape(-1), a.reshape(-1)).reshape(n, -1) / Z
            tn = jnp.exp(-0.5 * ((g - mu) / sd) ** 2) / (sd * jnp.sqrt(2 * jnp.pi)) / (
                ndtr((RT_HI - mu) / sd) - ndtr((RT_LO - mu) / sd))
            f = pi * tn * (pA if cor else 1 - pA) + (1 - pi) * fR
        else:  # model3
            p1w = C.stage1_profile(P["v1"], P["a"], P["tau"]) * (P["a"][:, None] * C.GLX_W[None, :])
            Z = jnp.maximum(C.twostage_mass_q(P["v1"], P["v"], P["a"], P["tau"], P["t"], p1w), 1e-12)[:, None]
            rep = lambda z: jnp.repeat(z[:, None], GRID.size, axis=1).reshape(-1)
            repk = lambda z: jnp.repeat(z[:, None, :], GRID.size, axis=1).reshape(n * GRID.size, -1)
            fD = C.twostage_pdf_q(jnp.tile(jnp.asarray(GRID), n), cc.reshape(-1), rep(P["v1"]), rep(P["v"]),
                                  rep(P["a"]), rep(P["tau"]), rep(P["t"]), repk(p1w)).reshape(n, -1) / Z
            tn = jnp.exp(-0.5 * ((g - mu) / sd) ** 2) / (sd * jnp.sqrt(2 * jnp.pi)) / jnp.exp(
                C.log_tn_mass((RT_LO - mu) / sd, (RT_HI - mu) / sd))
            f = pi * tn * (pA if cor else 1 - pA) + (1 - pi) * fD
        out.append(lapse * 0.5 / C.WIN + (1 - lapse) * f)
    return np.stack([np.asarray(o) for o in out], axis=-1)


def predicted(w, dens):
    """Joint density (n_grid, 2) for a condition, normalised, for one draw."""
    jd = np.tensordot(w, dens, axes=(0, 0)) / w.sum()
    return jd / (jd.sum() * 0.005)


def summarise(d, combos, dens_draws):
    """Per condition: accuracy per bin, quantiles of correct responses, overall accuracy."""
    d = d.copy()
    d["cond"] = d.obstacle.str.lower() + "_" + d.versione.map({"breve": "L1", "lunga": "L2"}) + "_s" + d.sessione.astype(str)
    rows, bins = [], []
    centers = (EDGES[:-1] + EDGES[1:]) / 2
    gb = np.clip(np.digitize(GRID, EDGES) - 1, 0, len(centers) - 1)
    groups = list(d.groupby("cond")) + [("jump", d[d.obstacle == "JUMP"]), ("slide", d[d.obstacle == "SLIDE"]), ("all", d)]
    for cond, g in groups:
        w = np.bincount(g.c.values, minlength=len(combos)).astype(float)
        obs_bin = np.clip(np.digitize(g.rt_raw.values, EDGES) - 1, 0, len(centers) - 1)
        n_obs = np.bincount(obs_bin, minlength=len(centers))
        acc_obs = np.bincount(obs_bin, weights=g.correct.values, minlength=len(centers)) / np.maximum(n_obs, 1)
        q_obs = np.quantile(g.rt_raw[g.correct == 1], QS)
        acc_pred, q_pred, accT_pred, num_b, den_b = [], [], [], [], []
        for dens in dens_draws:
            jd = predicted(w, dens)
            num = np.bincount(gb, weights=jd[:, 1], minlength=len(centers))
            den = np.bincount(gb, weights=jd.sum(1), minlength=len(centers))
            num_b.append(num)
            den_b.append(den)
            acc_pred.append(num / np.maximum(den, 1e-12))
            cdf = np.cumsum(jd[:, 1]) / jd[:, 1].sum()
            q_pred.append(np.interp(QS, cdf, GRID))
            accT_pred.append(jd[:, 1].sum() / jd.sum())
        acc_pred, q_pred = np.array(acc_pred), np.array(q_pred)
        ok = n_obs >= 50
        rmse = np.sqrt(np.average((acc_pred.mean(0)[ok] - acc_obs[ok]) ** 2, weights=n_obs[ok]))
        rows.append({"condition": cond, "n": len(g), "acc_obs": g.correct.mean(), "acc_pred": np.mean(accT_pred),
                     "rmse_acc_bins": rmse,
                     "mean_abs_quantile_error_ms": float(np.mean(np.abs(q_pred.mean(0) - q_obs)) * 1000),
                     "max_abs_quantile_error_ms": float(np.max(np.abs(q_pred.mean(0) - q_obs)) * 1000),
                     **{f"q{int(q*100)}_obs": q_obs[i] * 1000 for i, q in enumerate(QS)},
                     **{f"q{int(q*100)}_pred": q_pred.mean(0)[i] * 1000 for i, q in enumerate(QS)}})
        for i, cbin in enumerate(centers):
            bins.append({"condition": cond, "bin_centre_ms": cbin * 1000, "n": n_obs[i],
                         "correct_obs": float(acc_obs[i] * n_obs[i]), "acc_obs": acc_obs[i],
                         "acc_pred": acc_pred.mean(0)[i], "acc_pred_lo": np.quantile(acc_pred[:, i], .055),
                         "acc_pred_hi": np.quantile(acc_pred[:, i], .945),
                         "mass_correct_pred": np.mean(num_b, 0)[i], "mass_pred": np.mean(den_b, 0)[i]})
    return pd.DataFrame(rows), pd.DataFrame(bins)


def range_accuracy(bins, conds, lo_ms, hi_ms):
    """Observed and predicted accuracy over a range of response times: the predicted accuracy of each
    25-ms bin of each condition in `conds`, averaged with the observed number of responses as weights."""
    b = bins[bins.condition.isin(conds) & (bins.bin_centre_ms > lo_ms) & (bins.bin_centre_ms < hi_ms)]
    return {"n": int(b.n.sum()), "acc_obs": float(b.correct_obs.sum() / b.n.sum()),
            "acc_pred": float(np.average(b.acc_pred, weights=b.n))}


def run(m, d, combos, D, chains=None):
    P = C.load_draws(m, chains)
    n = next(iter(P.values())).shape[0]
    idx = np.random.default_rng(0).choice(n, N_DRAWS, replace=False)
    dens = [joint_density(m, {k: v[i] for k, v in P.items()}, D) for i in idx]
    rows, bins = summarise(d, combos, dens)
    print(f"=== {m}")
    print(rows[["condition", "n", "acc_obs", "acc_pred", "rmse_acc_bins", "mean_abs_quantile_error_ms"]].round(3).to_string(index=False))
    rows.to_csv(RESULTS / f"ppc_conditions_{m}.csv", index=False)
    bins.to_csv(RESULTS / f"ppc_bins_{m}.csv", index=False)
    w = np.bincount(d.c.values, minlength=len(combos)).astype(float)
    jd = np.mean([np.tensordot(w, x, axes=(0, 0)) / w.sum() for x in dens], axis=0)
    np.save(FITS / f"ppc_density_{m}.npy", jd)
    return rows, bins


def figure4(d):
    edges = np.linspace(RT_LO, RT_HI, 33)
    nb = len(edges) - 1
    mids = (edges[:-1] + edges[1:]) / 2 * 1000
    binof = lambda x: np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    nb_obs = np.bincount(binof(d.rt_raw.values), minlength=nb)
    obs_rt = nb_obs / len(d) / 0.025
    obs_acc = np.bincount(binof(d.rt_raw.values), weights=d.correct.values, minlength=nb) / np.maximum(nb_obs, 1)
    ok = nb_obs >= 50
    fig, axes = plt.subplots(2, len(MODELS), figsize=(2.6 * len(MODELS), 4.8), sharex=True, sharey="row")
    for j, (m, name) in enumerate(MODELS):
        jd = np.load(FITS / f"ppc_density_{m}.npy")                     # (n_grid, 2): incorrect, correct
        gb = np.clip(np.digitize(GRID, edges) - 1, 0, nb - 1)
        dens = jd.sum(1) / (jd.sum() * 0.005)
        pred_rt = np.bincount(gb, weights=dens, minlength=nb) / np.bincount(gb, minlength=nb)
        pred_acc = np.bincount(gb, weights=jd[:, 1], minlength=nb) / np.maximum(
            np.bincount(gb, weights=jd.sum(1), minlength=nb), 1e-12)
        ax = axes[0, j]
        ax.plot(mids, obs_rt, "o", color="#141a21", ms=2.5, label="Observed")
        ax.plot(mids, pred_rt, color="#2a78d6", lw=2, label="Predicted")
        ax.set_title(name, fontsize=9)
        ax = axes[1, j]
        ax.plot(mids[ok], obs_acc[ok] * 100, "o", color="#141a21", ms=2.5)
        ax.plot(mids[ok], pred_acc[ok] * 100, color="#2a78d6", lw=2)
        ax.set_xlabel("RT from cue (ms)", fontsize=8)
        for a_ in axes[:, j]:
            a_.axvline(0, color="#4b5562", lw=0.6)
            a_.grid(color="#d8dde3", lw=0.5)
            a_.set_axisbelow(True)
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)
            a_.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("Density (1/s)", fontsize=8)
    axes[1, 0].set_ylabel("Accuracy (%)", fontsize=8)
    axes[1, 0].set_ylim(50, 100)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    d, combos, D = C.data_for()
    D.pop("n_subj")
    best = json.load(open(RESULTS / "model_comparison.json"))["Model 3"]["best_mode"]
    out = {}
    for m, _ in MODELS:
        rows, bins = run(m, d, combos, D, chains=best if m == "model3" else None)
        cond = rows[~rows.condition.isin(["jump", "slide", "all"])]
        out[m] = {"max |acc_pred - acc_obs| over conditions (points)": float(100 * (cond.acc_pred - cond.acc_obs).abs().max()),
                  "rmse of accuracy in 25-ms bins, range over conditions (points)": [float(100 * cond.rmse_acc_bins.min()),
                                                                                   float(100 * cond.rmse_acc_bins.max())],
                  "mean |quantile error| (ms), range over conditions": [float(cond.mean_abs_quantile_error_ms.min()),
                                                                       float(cond.mean_abs_quantile_error_ms.max())],
                  "mean |quantile error| (ms), all responses": float(rows.loc[rows.condition == "all", "mean_abs_quantile_error_ms"].iloc[0]),
                  "max |quantile error| (ms) over conditions": float(cond.max_abs_quantile_error_ms.max()),
                  "jumps 100-200 ms before the cue": range_accuracy(
                      bins, [c for c in cond.condition if c.startswith("jump_")], -200, -100),
                  "all responses > 400 ms after the cue": range_accuracy(bins, ["all"], 400, 500)}
    if "model3" in out:
        out["model3_chains_used"] = best
    print(json.dumps(out, indent=1))
    json.dump(out, open(RESULTS / "ppc_summary.json", "w"), indent=1)
    figure4(d)
