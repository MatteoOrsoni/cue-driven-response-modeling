# =============================================================================
# Step 12. Parameter recovery for Model 1 (Method: "Parameter recovery and reliability"; Results, end of
# the Model 1 section).
#
#   python code/12_parameter_recovery.py simulate
#       one dataset with exactly the structure of the real one (same children, sessions, levels and
#       obstacles), simulated from the posterior medians of Model 1: each response is timed with the
#       probability given by the model; diffusion responses are simulated in steps of 0.5 ms and redrawn
#       when they fall outside the window; 1% of responses are replaced by lapses
#       -> results/fits/sim_recovery.csv, results/fits/sim_recovery_truth.npz
#   python code/06_fit_model1.py recovery        (refit, GPU)
#   python code/12_parameter_recovery.py evaluate
#       coverage of the 45 population parameters by the recovered 89% CIs; bias of drift, boundary and
#       non-decision time; correlation of true and recovered child effects; recovery of the RQ2-RQ4
#       contrasts -> results/recovery_population.csv, results/recovery_children.csv,
#       results/recovery_contrasts.csv, results/recovery_summary.json
# =============================================================================
import json
import sys

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import truncnorm

import mixture as M
import model1_tools as T
from config import FITS, RESULTS, RT_HI, RT_LO, summarise

RE = T.RE
rng = np.random.default_rng(7)


def posterior(tag):
    idata = az.from_netcdf(FITS / f"idata_{tag}.nc")
    post = idata["posterior"]
    return idata, {k: post[k].values.reshape(-1, *post[k].shape[2:]) for k in post.data_vars}


def params(Tr, combos):
    """Parameters per combination from point values Tr (population + child effects u)."""
    cs, cell = combos.s.values, combos.cella_i.values
    sess, ver, jump = combos.sess.values, combos.ver.values, combos.jump.values
    u = Tr["u"]
    v = Tr["b_v"][cell] + Tr["b_v_sess"] * sess + Tr["b_v_jump"] * jump + u["v0"][cs] + u["vver"][cs] * ver
    a = np.exp(Tr["b_a"][cell] + Tr["b_a_sess"] * sess + u["a0"][cs] + u["aver"][cs] * ver)
    t = M.T_MIN + (M.T_MAX - M.T_MIN) * expit(Tr["c_t"] + Tr["c_t_ver"] * ver + Tr["c_t_sess"] * sess + u["t"][cs])
    pi = expit(Tr["b_p"][cell] + Tr["b_p_sess"] * sess + Tr["b_p_jump"] * jump + u["p0"][cs] + u["pver"][cs] * ver)
    mu = Tr["m0"] + Tr["m_ver"] * ver + Tr["m_sess"] * sess + u["m"][cs]
    sdA = np.exp(Tr["g0"] + Tr["g_ver"] * ver + Tr["g_sess"] * sess + u["g"][cs])
    pA = expit(Tr["r0"] + Tr["r_jump"] * jump + Tr["r_ver"] * ver + Tr["r_sess"] * sess + u["r"][cs])
    return dict(v=v, a=a, t=t, pi=pi, mu=mu, sdA=sdA, pA=pA)


def simulate_ddm(v, a, t, dt=0.0005):
    n = len(v)
    x = a / 2
    rt, cor = np.full(n, np.nan), np.zeros(n, int)
    alive = np.ones(n, bool)
    for s in range(1, int(RT_HI / dt) + 1):
        x = np.where(alive, x + v * dt + np.sqrt(dt) * rng.standard_normal(n), x)
        tt = t + s * dt
        up, lo = alive & (x >= a), alive & (x <= 0)
        rt[up | lo] = tt[up | lo]
        cor[up] = 1
        alive &= ~(up | lo) & (tt < RT_HI)
    return rt, cor


def simulate():
    _, P = posterior("model1")
    d = pd.read_csv(FITS / "trials_model1.csv")
    combos = pd.read_csv(FITS / "combos_model1.csv")
    U = T.child_effects(P)
    Tr = {k: np.median(v, 0) for k, v in P.items() if not k.startswith(("z_", "sd_"))}
    Tr["u"] = {k: np.median(U[k], 0) for k in RE}
    p = params(Tr, combos)
    c = d.c.values
    q = {k: v[c] for k, v in p.items()}
    n = len(d)
    timed = rng.random(n) < q["pi"]
    lo, hi = (RT_LO - q["mu"]) / q["sdA"], (RT_HI - q["mu"]) / q["sdA"]
    rtA = truncnorm.rvs(lo, hi, loc=q["mu"], scale=q["sdA"], random_state=rng)
    corA = (rng.random(n) < q["pA"]).astype(int)
    rtR, corR = np.full(n, np.nan), np.zeros(n, int)
    todo = np.where(~timed)[0]
    while len(todo):
        r_, c_ = simulate_ddm(q["v"][todo], q["a"][todo], q["t"][todo])
        ok = ~np.isnan(r_)
        rtR[todo[ok]], corR[todo[ok]] = r_[ok], c_[ok]
        todo = todo[~ok]
    sim = d[["subj", "sessione", "versione", "sess", "ver", "cella", "obstacle", "trial_n", "omission"]].copy()
    sim["rt_raw"] = np.where(timed, rtA, rtR)
    sim["correct"] = np.where(timed, corA, corR)
    lapse = rng.random(n) < M.LAPSE           # contaminant: uniform in the window, correct at chance
    sim.loc[lapse, "rt_raw"] = rng.uniform(RT_LO, RT_HI, lapse.sum())
    sim.loc[lapse, "correct"] = (rng.random(lapse.sum()) < 0.5).astype(int)
    sim["timed_true"] = timed.astype(int)
    sim.to_csv(FITS / "sim_recovery.csv", index=False)
    np.savez(FITS / "sim_recovery_truth.npz",
             **{f"pop_{k}": np.atleast_1d(v) for k, v in Tr.items() if k != "u"},
             **{f"u_{k}": v for k, v in Tr["u"].items()})
    print(f"simulated {n} responses; timed share {timed.mean():.3f}; accuracy {sim.correct.mean():.3f} "
          f"(observed {d.correct.mean():.3f})")


def evaluate():
    pd.set_option("display.width", 200)
    truth = np.load(FITS / "sim_recovery_truth.npz")
    idata, P = posterior("model1_recovery")
    s = az.summary(idata, var_names=["~z_"], filter_vars="like")
    out = {"rhat_max": float(pd.to_numeric(s["r_hat"]).max()), "ess_min": float(pd.to_numeric(s["ess_bulk"]).min()),
           "divergences": int(np.asarray(idata["sample_stats"]["diverging"]).sum())}
    # population parameters
    rows = []
    for k in [k for k in P if not k.startswith(("z_", "sd_"))]:
        tv, draws = truth[f"pop_{k}"], P[k].reshape(P[k].shape[0], -1)
        for j in range(draws.shape[1]):
            lo, hi = np.quantile(draws[:, j], [0.055, 0.945])
            rows.append({"param": k if draws.shape[1] == 1 else f"{k}[{j}]", "true": float(tv[j]),
                         "recovered": float(np.median(draws[:, j])), "lo": lo, "hi": hi,
                         "covered": bool(lo <= tv[j] <= hi)})
    pop = pd.DataFrame(rows)
    out["population"] = {"n": int(len(pop)), "covered": int(pop.covered.sum()),
                         "r_true_recovered": float(np.corrcoef(pop.true, pop.recovered)[0, 1]),
                         "misses": pop.loc[~pop.covered, "param"].tolist()}
    print(f"\npopulation parameters: {pop.covered.sum()}/{len(pop)} true values inside the 89% CI; "
          f"r(true, recovered) = {out['population']['r_true_recovered']:.3f}")
    print(pop[~pop.covered].round(3).to_string(index=False))
    # bias of the diffusion parameters
    rv = lambda k: np.median(P[k], 0)
    out["bias"] = {"drift, mean over cells (recovered - true)": float((rv("b_v") - truth["pop_b_v"]).mean()),
                   "boundary, mean ratio recovered / true - 1": float((np.exp(rv("b_a")) / np.exp(truth["pop_b_a"])).mean() - 1),
                   "non-decision time true (ms)": float(T.tms(truth["pop_c_t"][0])),
                   "non-decision time recovered (ms)": float(np.median(T.tms(P["c_t"])))}
    # child effects
    U = T.child_effects(P)
    rec = pd.DataFrame({"r_true_recovered": {k: float(np.corrcoef(truth[f"u_{k}"], U[k].mean(0))[0, 1]) for k in RE}})
    out["children"] = rec.r_true_recovered.to_dict()
    print("\n", rec.round(3).to_string())
    # contrasts of RQ2-RQ4
    Tr = {k[4:]: (v[None] if v.size > 1 else v) for k, v in truth.items() if k.startswith("pop_")}
    ct, cr = T.contrasts(Tr), T.contrasts(P)
    crow = []
    for k in ct:
        sm = summarise(cr[k])
        tv = float(np.asarray(ct[k]).reshape(-1)[0])
        crow.append({"block": k[0], "contrast": k[1], "true": tv, "recovered": sm["median"], "lo": sm["lo"],
                     "hi": sm["hi"], "covered": sm["lo"] <= tv <= sm["hi"],
                     "ratio": sm["median"] / tv if abs(tv) > 1e-9 else np.nan})
    crow = pd.DataFrame(crow)
    drift_rq3 = crow[(crow.block == "RQ3") & crow.contrast.str.startswith("v:")]
    out["contrasts"] = {"n": int(len(crow)), "covered": int(crow.covered.sum()),
                        "drift minute contrasts: median recovered / true": float(drift_rq3.ratio.median())}
    print("\n", crow.round(3).to_string(index=False))
    pop.to_csv(RESULTS / "recovery_population.csv", index=False)
    rec.to_csv(RESULTS / "recovery_children.csv")
    crow.to_csv(RESULTS / "recovery_contrasts.csv", index=False)
    json.dump(out, open(RESULTS / "recovery_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "population"}, indent=1))


if __name__ == "__main__":
    {"simulate": simulate, "evaluate": evaluate}[sys.argv[1]]()
