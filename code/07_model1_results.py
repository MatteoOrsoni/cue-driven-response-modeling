# =============================================================================
# Step 7. Results of Model 1 (Results: "A mixture of timed and diffusion responses (Model 1)"; Model 1 rows
# of Tables 3, 4A and 5; Figure 3).
#
#   - convergence diagnostics
#   - parameter estimates quoted in the text (share of timed responses, centre, spread and accuracy of timed
#     responses, per-child spread and its correlation with the raw SD of correct responses, drift, boundary,
#     non-decision time)
#   - contrasts for RQ2-RQ4 (draw by draw)
#   - posterior predictive simulation of the full dataset from 20 draws (Figure 3A-B)
#   - posterior probability that each response is timed (Figure 3C-D) and comparison with the 150-ms cut-off
#
#   python code/07_model1_results.py
# Output: results/model1_summary.json, results/model1_contrasts.csv, results/model1_p_timed_by_bin.csv,
#         figures/figure3_model1.png
# =============================================================================
import json

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr, truncnorm

import mixture as M
import model1_tools as T
from config import CELLS, CUT, FIGURES, RESULTS, RT_HI, RT_LO, summarise

rng = np.random.default_rng(2026)


# -----------------------------------------------------------------------------
# Posterior predictive simulation
# -----------------------------------------------------------------------------
def simulate_ddm(v, a, t, dt=0.001):
    """Euler simulation; returns rt (nan beyond the window) and correctness."""
    n = len(v)
    x = a / 2
    rt = np.full(n, np.nan)
    cor = np.zeros(n, int)
    alive = np.ones(n, bool)
    steps = int((RT_HI - t.min()) / dt) + 1
    for s in range(1, steps + 1):
        x = np.where(alive, x + v * dt + np.sqrt(dt) * rng.standard_normal(n), x)
        tt = t + s * dt
        up, lo = alive & (x >= a), alive & (x <= 0)
        rt[up | lo] = tt[up | lo]
        cor[up] = 1
        alive &= ~(up | lo) & (tt < RT_HI)
    return rt, cor


def ppc(P, d, combos, n_draws=20):
    c = d.c.values
    sims = []
    draws = rng.choice(P["b_v"].shape[0], n_draws, replace=False)
    for i in draws:
        p = T.combo_params(P, combos, i)
        q = {k: v[c] for k, v in p.items()}
        n = len(c)
        timed = rng.random(n) < q["pi"]
        lo, hi = (RT_LO - q["mu"]) / q["sdA"], (RT_HI - q["mu"]) / q["sdA"]
        rtA = truncnorm.rvs(lo, hi, loc=q["mu"], scale=q["sdA"], random_state=rng)
        corA = (rng.random(n) < q["pA"]).astype(int)
        # diffusion responses: simulated again until they fall within the window
        rtR, corR = np.full(n, np.nan), np.zeros(n, int)
        todo = np.where(~timed)[0]
        for _ in range(20):
            if len(todo) == 0:
                break
            r_, c_ = simulate_ddm(q["v"][todo], q["a"][todo], q["t"][todo])
            ok = ~np.isnan(r_)
            rtR[todo[ok]], corR[todo[ok]] = r_[ok], c_[ok]
            todo = todo[~ok]
        rt = np.where(timed, rtA, rtR)
        cor = np.where(timed, corA, corR)
        sims.append(pd.DataFrame({"draw": i, "rt": rt, "correct": cor, "obstacle": d.obstacle.values,
                                  "timed": timed}).dropna())
    return pd.concat(sims)


def figure3(d, sim, pt):
    edges = np.linspace(-0.300, 0.500, 33)                 # 32 bins of 25 ms
    nb = len(edges) - 1
    mids = (edges[:-1] + edges[1:]) / 2 * 1000
    binof = lambda x: np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    INK2, GRID = "#4b5562", "#d8dde3"
    for ax in axes.flat:
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axvline(0, color="#141a21", lw=0.8)
    # A: response-time distribution
    ax = axes[0, 0]
    obs = np.bincount(binof(d.rt_raw.values), minlength=nb) / len(d)
    pred = np.array([np.bincount(binof(g.rt.values), minlength=nb) / len(g) for _, g in sim.groupby("draw")])
    ax.fill_between(mids, np.quantile(pred, .055, 0) * 100, np.quantile(pred, .945, 0) * 100, color="#eb6834", alpha=.3, lw=0)
    ax.plot(mids, pred.mean(0) * 100, color="#eb6834", lw=2, label="Model")
    ax.plot(mids, obs * 100, "o", color="#141a21", ms=3, label="Observed")
    ax.set_ylabel("Responses (%)")
    ax.set_title("A", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    # B: accuracy by obstacle type
    ax = axes[0, 1]
    for ob, col, lab in [("JUMP", "#1baf7a", "Jump"), ("SLIDE", "#4a3aa7", "Slide")]:
        o = d[d.obstacle == ob]
        bo = binof(o.rt_raw.values)
        n_obs = np.bincount(bo, minlength=nb)
        acc_obs = np.bincount(bo, weights=o.correct.values, minlength=nb) / np.maximum(n_obs, 1)
        s = sim[sim.obstacle == ob]
        pr = []
        for _, g in s.groupby("draw"):
            bg = binof(g.rt.values)
            pr.append(np.bincount(bg, weights=g.correct.values, minlength=nb) / np.maximum(np.bincount(bg, minlength=nb), 1))
        pr = np.array(pr)
        okb = n_obs >= 50
        ax.fill_between(mids[okb], np.quantile(pr, .055, 0)[okb] * 100, np.quantile(pr, .945, 0)[okb] * 100, color=col, alpha=.25, lw=0)
        ax.plot(mids[okb], pr.mean(0)[okb] * 100, color=col, lw=2, label=f"{lab}: model")
        ax.plot(mids[okb], acc_obs[okb] * 100, "o", color=col, ms=3, mfc="white", label=f"{lab}: observed")
    ax.set_ylim(40, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("B", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center")
    # C: probability of the timed component by response time
    ax = axes[1, 0]
    dd = d.assign(pt=pt, b=binof(d.rt_raw.values))
    m = dd.groupby("b").pt.mean().reindex(range(nb)).values
    ax.plot(mids, m * 100, color="#4a3aa7", lw=2)
    ax.axvline(150, color="#141a21", lw=0.8, ls="--")
    ax.text(155, 90, "150-ms cut-off", fontsize=8, color=INK2)
    ax.set_ylabel("P(timed component) (%)")
    ax.set_xlabel("Response time relative to the cue (ms)")
    ax.set_title("C", loc="left", fontweight="bold")
    # D: per child and session
    ax = axes[1, 1]
    run = dd.groupby(["s", "sessione"]).pt.mean()
    cut = d.assign(x=d.rt_raw < CUT).groupby(["s", "sessione"]).x.mean()
    ax.scatter(cut * 100, run * 100, s=10, color="#2a78d6", alpha=.7, lw=0)
    ax.plot([0, 100], [0, 100], color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Responses < 150 ms (%)")
    ax.set_ylabel("Timed component, estimated (%)")
    ax.set_title("D", loc="left", fontweight="bold")
    for ax in axes[0]:
        ax.set_xlabel("Response time relative to the cue (ms)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_model1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return dd, run, cut, pd.DataFrame({"bin_centre_ms": mids, "n": np.bincount(dd.b, minlength=nb),
                                       "mean_p_timed": m})


# -----------------------------------------------------------------------------
# Estimates quoted in the text
# -----------------------------------------------------------------------------
def estimates(P, d):
    E = {}
    E["timed share by cell (median)"] = {c: float(np.median(expit(P["b_p"][:, i]))) for i, c in enumerate(CELLS)}
    E["timed centre (ms)"] = summarise(1000 * P["m0"])
    E["timed SD (ms)"] = summarise(1000 * np.exp(P["g0"]))
    E["timed accuracy"] = summarise(expit(P["r0"]))
    E["timed accuracy, slides"] = summarise(expit(P["r0"] - 0.5 * P["r_jump"]))
    E["timed accuracy, jumps"] = summarise(expit(P["r0"] + 0.5 * P["r_jump"]))
    E["boundary a by cell (median)"] = {c: float(np.median(np.exp(P["b_a"][:, i]))) for i, c in enumerate(CELLS)}
    E["drift v by cell (median)"] = {c: float(np.median(P["b_v"][:, i])) for i, c in enumerate(CELLS)}
    E["drift v, jump - slide"] = summarise(P["b_v_jump"])
    E["non-decision time (ms)"] = summarise(T.tms(P["c_t"]))
    # per-child spread of the timed component (average level and session) and the raw data
    sig_child = np.exp(P["g0"][:, None] + P["sd_g"][:, None] * P["z_g"]).mean(0) * 1000
    E["timed SD per child (ms)"] = {"min": float(sig_child.min()), "median": float(np.median(sig_child)),
                                    "max": float(sig_child.max())}
    for lab, sel in (("all correct responses", d.correct == 1),
                     ("correct responses < 150 ms", (d.correct == 1) & (d.rt_raw < CUT))):
        sd_obs = d[sel].groupby("s").rt_raw.std() * 1000
        E[f"r(timed SD per child, SD of {lab})"] = float(pearsonr(sig_child[sd_obs.index.values], sd_obs.values)[0])
    return E


def classification(d, dd, run, cut):
    pt, rt = dd.pt.values, d.rt_raw.values
    late = rt >= CUT
    return {
        "mean P(timed), all responses": float(pt.mean()),
        "mean P(timed), before the cue": float(pt[rt < 0].mean()),
        "mean P(timed), 0-100 ms": float(pt[(rt >= 0) & (rt < 0.1)].mean()),
        "mean P(timed), 400-500 ms": float(pt[rt >= 0.4].mean()),
        "responses >= 150 ms": int(late.sum()),
        "mean P(timed), responses >= 150 ms": float(pt[late].mean()),
        "mean P(timed), responses < 150 ms": float(pt[~late].mean()),
        "mean P(timed), responses 0-150 ms": float(pt[(rt >= 0) & ~late].mean()),
        "sessions": int(len(run)),
        "sessions with estimated timed share > share < 150 ms": int((run.values > cut.values).sum()),
        "mean difference (percentage points)": float(100 * (run.values - cut.values).mean()),
        "r(timed share, share < 150 ms) across sessions": float(np.corrcoef(cut, run)[0, 1]),
    }


if __name__ == "__main__":
    idata, P, d, combos = T.load("model1")
    s = az.summary(idata, var_names=["~z_"], filter_vars="like")
    diag = {"rhat_max": float(pd.to_numeric(s["r_hat"]).max()), "ess_bulk_min": float(pd.to_numeric(s["ess_bulk"]).min()),
            "divergences": int(np.asarray(idata["sample_stats"]["diverging"]).sum()),
            "draws": int(P["b_v"].shape[0])}
    print("diagnostics:", diag)
    C = T.contrasts(P)
    tab = pd.DataFrame({k: summarise(v) for k, v in C.items()}).T
    tab.index.names = ["block", "contrast"]
    print(tab.round(3).to_string())
    tab.to_csv(RESULTS / "model1_contrasts.csv")
    E = estimates(P, d)
    sim = ppc(P, d, combos)
    pt = T.p_timed_trial(P, d, combos)
    dd, run, cut, bins = figure3(d, sim, pt)
    bins.to_csv(RESULTS / "model1_p_timed_by_bin.csv", index=False)
    K = classification(d, dd, run, cut)
    out = {"diagnostics": diag, "estimates": E, "classification": K}
    print(json.dumps({"estimates": E, "classification": K}, indent=1, ensure_ascii=False))
    json.dump(out, open(RESULTS / "model1_summary.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)
