# =============================================================================
# Post-processing helpers for Model 1 (posterior of mixture.run): parameters per combination,
# contrasts for RQ2-RQ4, posterior probability that each response is timed, child effects.
# =============================================================================
import arviz as az
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import truncnorm

import mixture as M
from config import CELLS, FITS, RT_HI, RT_LO

RE = [k for k, _ in M.RANEF]


def load(tag="model1"):
    """Posterior draws (chains pooled) and the data as indexed by the model."""
    idata = az.from_netcdf(FITS / f"idata_{tag}.nc")
    post = idata["posterior"]
    P = {k: post[k].values.reshape(-1, *post[k].shape[2:]) for k in post.data_vars}
    d = pd.read_csv(FITS / f"trials_{tag}.csv")
    combos = pd.read_csv(FITS / f"combos_{tag}.csv")
    return idata, P, d, combos


def tms(eta):
    """Non-decision time in ms from its linear predictor."""
    return 1000 * (M.T_MIN + (M.T_MAX - M.T_MIN) * expit(eta))


def child_effects(P):
    """Child effects u = sd * z, per draw: {name: (n_draws, n_children)}."""
    return {k: P[f"sd_{k}"][:, None] * P[f"z_{k}"] for k in RE}


def combo_params(P, combos, i):
    """Parameters per combination for draw i."""
    cs, cell = combos.s.values, combos.cella_i.values
    sess, ver, jump = combos.sess.values, combos.ver.values, combos.jump.values
    u = {k: P[f"sd_{k}"][i] * P[f"z_{k}"][i] for k in RE}
    v = P["b_v"][i][cell] + P["b_v_sess"][i] * sess + P["b_v_jump"][i] * jump + u["v0"][cs] + u["vver"][cs] * ver
    a = np.exp(P["b_a"][i][cell] + P["b_a_sess"][i] * sess + u["a0"][cs] + u["aver"][cs] * ver)
    t = M.T_MIN + (M.T_MAX - M.T_MIN) * expit(P["c_t"][i] + P["c_t_ver"][i] * ver + P["c_t_sess"][i] * sess + u["t"][cs])
    lp = P["b_p"][i][cell] + P["b_p_sess"][i] * sess + P["b_p_jump"][i] * jump + u["p0"][cs] + u["pver"][cs] * ver
    mu = P["m0"][i] + P["m_ver"][i] * ver + P["m_sess"][i] * sess + u["m"][cs]
    sdA = np.exp(P["g0"][i] + P["g_ver"][i] * ver + P["g_sess"][i] * sess + u["g"][cs])
    pA = expit(P["r0"][i] + P["r_jump"][i] * jump + P["r_ver"][i] * ver + P["r_sess"][i] * sess + u["r"][cs])
    return dict(v=v, a=a, t=t, pi=expit(lp), mu=mu, sdA=sdA, pA=pA)


def contrasts(P):
    """Population-level contrasts (draw by draw). Keys: (block, contrast).
    RQ2 = practice (session 2 - 1), RQ3 = minutes within a level, RQ4 = Level 2 - Level 1."""
    b_v, b_a, b_p = P["b_v"], P["b_a"], P["b_p"]
    idx = {c: i for i, c in enumerate(CELLS)}
    C = {}
    # RQ2: practice
    C[("RQ2", "drift rate v")] = P["b_v_sess"]
    C[("RQ2", "log boundary a")] = P["b_a_sess"]
    C[("RQ2", "non-decision t (ms)")] = tms(P["c_t"] + 0.5 * P["c_t_sess"]) - tms(P["c_t"] - 0.5 * P["c_t_sess"])
    C[("RQ2", "logit P(timed)")] = P["b_p_sess"]
    C[("RQ2", "timed centre (ms)")] = 1000 * P["m_sess"]
    C[("RQ2", "log timed SD")] = P["g_sess"]
    C[("RQ2", "logit accuracy of timed responses")] = P["r_sess"]
    # RQ3: consecutive minutes
    for name, arr in [("v", b_v), ("log a", b_a), ("logit P(timed)", b_p)]:
        for lev, n in (("L1", 4), ("L2", 5)):
            code = "B" if lev == "L1" else "L"
            for m in range(2, n + 1):
                C[("RQ3", f"{name}: {lev} min {m} - {m - 1}")] = arr[:, idx[f"{code}{m}"]] - arr[:, idx[f"{code}{m - 1}"]]
    # RQ4: levels (minutes 1-4 and minute 1)
    for name, arr in [("v", b_v), ("log a", b_a), ("logit P(timed)", b_p)]:
        C[("RQ4", f"{name}: L2 - L1 (min 1-4)")] = np.mean(
            [arr[:, idx[f"L{m}"]] - arr[:, idx[f"B{m}"]] for m in range(1, 5)], 0)
        C[("RQ4", f"{name}: L2 - L1 (min 1)")] = arr[:, idx["L1"]] - arr[:, idx["B1"]]
    C[("RQ4", "non-decision t (ms)")] = tms(P["c_t"] + 0.5 * P["c_t_ver"]) - tms(P["c_t"] - 0.5 * P["c_t_ver"])
    C[("RQ4", "timed centre (ms)")] = 1000 * P["m_ver"]
    C[("RQ4", "log timed SD")] = P["g_ver"]
    C[("RQ4", "logit accuracy of timed responses")] = P["r_ver"]
    # obstacle type
    C[("Obstacle", "v: jump - slide")] = P["b_v_jump"]
    C[("Obstacle", "logit P(timed): jump - slide")] = P["b_p_jump"]
    C[("Obstacle", "logit accuracy of timed responses: jump - slide")] = P["r_jump"]
    return C


def p_timed_trial(P, d, combos):
    """Posterior probability that each response belongs to the timed component, at the posterior
    median of the parameters."""
    med = {k: np.median(v, 0) for k, v in P.items()}
    p = combo_params({k: v[None] for k, v in med.items()}, combos, 0)
    c = d.c.values
    q = {k: v[c] for k, v in p.items()}
    rt, cor = d.rt_raw.values, d.correct.values
    lA = (truncnorm.logpdf(rt, (RT_LO - q["mu"]) / q["sdA"], (RT_HI - q["mu"]) / q["sdA"],
                           loc=q["mu"], scale=q["sdA"])
          + np.where(cor == 1, np.log(q["pA"]), np.log1p(-q["pA"])))
    fR = np.asarray(M.ddm_pdf(jnp.asarray(rt - q["t"]), jnp.asarray(cor), jnp.asarray(q["v"]), jnp.asarray(q["a"])))
    Z = np.asarray(M.ddm_mass_window(jnp.asarray(p["v"]), jnp.asarray(p["a"]), jnp.asarray(p["t"])))[c]
    with np.errstate(divide="ignore"):      # Z = 0 in a few combinations: the response is then timed
        lR = np.log(np.maximum(fR, 1e-300)) - np.log(Z)
    num = np.log(q["pi"]) + lA
    den = np.logaddexp(num, np.log1p(-q["pi"]) + lR)
    return np.exp(num - den)


def child_level_scores(P):
    """Posterior means of the child-specific effects used for split-half reliability."""
    U = {k: v.mean(0) for k, v in child_effects(P).items()}
    return {
        "drift, Level 1": U["v0"] - 0.5 * U["vver"], "drift, Level 2": U["v0"] + 0.5 * U["vver"],
        "log a, Level 1": U["a0"] - 0.5 * U["aver"], "log a, Level 2": U["a0"] + 0.5 * U["aver"],
        "logit P(timed), Level 1": U["p0"] - 0.5 * U["pver"], "logit P(timed), Level 2": U["p0"] + 0.5 * U["pver"],
        "non-decision time (logit)": U["t"], "timed centre": U["m"], "timed accuracy": U["r"],
        "timed spread (log SD)": U["g"]}
