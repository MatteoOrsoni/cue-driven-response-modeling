# =============================================================================
# Models of the formal comparison, with a common hierarchical structure.
#
#   model1     timed-diffusion mixture (Model 1; fitted by mixture.py, its log-likelihood is computed here)
#   model2     fast-guess mixture (Model 2): as Model 1, timed responses correct with probability .5
#   model3     two-stage mixture (Model 3): as Model 1, but the diffusion starts tau s before the cue with
#              drift v1 (visual information) and switches to drift v at the cue
#   ddm_trunc  hierarchical truncated DDM: diffusion starting at the cue, fitted to responses >= 150 ms
#              with a likelihood conditioned on the 150-500 ms window (same structure for v, a, t)
#
# All models are conditioned on a response within the window, include a 1% uniform contaminant and
# average the density over a 10-ms window, so that PSIS-LOO values are comparable.
# In Model 3 the boundary passes through a smooth cap, a_eff = 5 tanh(a / 5), for numerical stability.
#
# Fits are run by 08_fit_alternative_models.py; settings can be changed with environment variables:
#   FIT_WARMUP, FIT_SAMPLES, FIT_CHAINS, FIT_DEPTH  sampler settings
#   FIT_REPARAM=1   exact reparameterisation separating each population intercept from the mean of the
#                   corresponding child effects (used for Model 3)
#   FIT_JITTER=x    starting values dispersed across chains (N(0, x * prior SD) in unconstrained space,
#                   3 x wider for c_tau and b_p; used for Model 3)
# =============================================================================
import functools
import json
import os
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import expit
from numpyro.infer import MCMC, NUTS

import mixture as MD  # diffusion density (Navarro & Fuss), data structure
from config import CUT, FITS, INITS, RT_HI, RT_LO

# Full float32 precision in matrix products. On recent GPUs the default is TF32 (~3 significant digits),
# which makes the gradient of the two-stage density inconsistent with the energy and forces NUTS into
# tiny steps.
jax.config.update("jax_default_matmul_precision", "highest")

WIN = RT_HI - RT_LO
N_GRID = 96
LAPSE = MD.LAPSE
H_RES = MD.H_RES
GL_X, GL_W, N_GL = MD.GL_X, MD.GL_W, MD.N_GL
binned = MD.binned
log_tn_mass = MD.log_tn_mass

# eigenfunction expansion for the position of the accumulator at the cue (Model 3)
JJ = jnp.arange(1, 31)
SIN_START = jnp.sin(JJ * jnp.pi / 2)                            # start at a/2
T_MAX = MD.T_MAX
TAU_MIN, TAU_MAX = 0.1, 1.6       # bounds of the onset of visual accumulation, s before the cue (Model 3)
A_CAP = 5.0


# -----------------------------------------------------------------------------
# Densities
# -----------------------------------------------------------------------------
def ddm_mass(v, a, lo, hi, n=N_GRID):
    """Integral of f_upper + f_lower over decision times in [lo, hi] (lo >= 0)."""
    lo = jnp.maximum(lo, 0.0)
    hi = jnp.maximum(hi, lo + 1e-6)
    g = jnp.linspace(0.0, 1.0, n + 1)
    tau = lo[:, None] + (hi - lo)[:, None] * g[None, :]
    vb, ab = v[:, None] * jnp.ones_like(tau), a[:, None] * jnp.ones_like(tau)
    f = MD.ddm_pdf(tau, jnp.ones_like(tau), vb, ab) + MD.ddm_pdf(tau, jnp.zeros_like(tau), vb, ab)
    return jnp.trapezoid(f, dx=((hi - lo) / n)[:, None], axis=1)


# ---- two-stage diffusion (Model 3) ----------------------------------------------------------------
# The position of the accumulator at the cue, p1(x), is evaluated on Gauss-Legendre nodes; the first
# passage after the cue uses Navarro & Fuss densities with an arbitrary starting point, in log scale.
NX = 16
GLX_X, GLX_W = np.polynomial.legendre.leggauss(NX)
GLX_X, GLX_W = jnp.asarray((GLX_X + 1) / 2), jnp.asarray(GLX_W / 2)          # nodes on [0, 1]
NU = 24
GLU_S, GLU_W = np.polynomial.legendre.leggauss(NU)
GLU_S, GLU_W = jnp.asarray((GLU_S + 1) / 2), jnp.asarray(GLU_W / 2)          # nodes on [0, 1]


def ddm_pdf_w(tau, correct, v, a, w):
    """First-passage density with relative starting point w (upper boundary = correct)."""
    vv = jnp.where(correct == 1, -v, v)
    ww = jnp.where(correct == 1, 1 - w, w)
    pos = tau > 0
    ts = jnp.where(pos, tau, 1.0)
    fs = jnp.maximum(MD._fstd(ts / a ** 2, ww), 1e-30)
    logd = -vv * a * ww - vv ** 2 * ts / 2 - 2 * jnp.log(a) + jnp.log(fs)
    return jnp.where(pos, jnp.exp(jnp.minimum(logd, 80.0)), 0.0)


def stage1_profile(v1, a, tau):
    """p1 on the nodes x = a * GLX_X, (n, NX): (sub-normalised) density of the position at the cue."""
    lam = JJ[None, :] ** 2 * jnp.pi ** 2 / (2 * a[:, None] ** 2)
    S = jnp.einsum("nj,jx->nx", SIN_START[None, :] * jnp.exp(-lam * tau[:, None]),
                   jnp.sin(JJ[:, None] * jnp.pi * GLX_X[None, :]))
    x = a[:, None] * GLX_X[None, :]
    expo = v1[:, None] * (x - a[:, None] / 2) - v1[:, None] ** 2 * tau[:, None] / 2
    return jnp.maximum((2 / a)[:, None] * jnp.exp(jnp.minimum(expo, 80.0)) * S, 0.0)


def stage2_flux_q(u, correct, v2, a, p1w):
    """Flux after the cue, per observation: sum_i W_i p1(x_i) f(u | x_i), with p1w = p1 * a * GLX_W."""
    n = u.shape[0]
    rep = lambda z: jnp.repeat(z[:, None], NX, axis=1).reshape(-1)
    f = ddm_pdf_w(rep(u), rep(correct), rep(v2), rep(a), jnp.tile(GLX_X, n)).reshape(n, NX)
    return jnp.sum(p1w * f, axis=1)


def twostage_pdf_q(rt, correct, v1, v2, a, tau, tnd, p1w, h=H_RES):
    """Mean over [rt - h/2, rt + h/2] of the two-stage density (rt relative to the cue)."""
    lo, hi = rt - h / 2, rt + h / 2
    b1 = jnp.minimum(hi, tnd)
    half1 = jnp.maximum(b1 - lo, 0.0) / 2
    pts = ((lo + b1) / 2)[:, None] + half1[:, None] * GL_X[None, :]
    s1 = pts + tau[:, None] - tnd[:, None]
    f1 = MD.ddm_pdf(jnp.where(s1 > 0, s1, 1.0).reshape(-1), jnp.repeat(correct, N_GL),
                    jnp.repeat(v1, N_GL), jnp.repeat(a, N_GL)).reshape(-1, N_GL)
    int1 = half1 * jnp.sum(jnp.where(s1 > 0, f1, 0.0) * GL_W[None, :], axis=1)
    ua, ub = jnp.maximum(lo - tnd, 0.0), jnp.maximum(hi - tnd, 0.0)
    half2 = (ub - ua) / 2
    int2 = 0.0
    for i in range(N_GL):
        uu = (ua + ub) / 2 + half2 * GL_X[i]
        int2 = int2 + GL_W[i] * stage2_flux_q(uu, correct, v2, a, p1w)
    int2 = half2 * int2
    return jnp.maximum((int1 + int2) / h, 0.0)


def twostage_mass_q(v1, v2, a, tau, tnd, p1w):
    """P(response within the window) per combination: stage 1 with the standard DDM, stage 2 by
    quadrature in u (nodes concentrated near the cue: u = U s^2)."""
    m1 = ddm_mass(v1, a, RT_LO + tau - tnd, tau)
    U = jnp.maximum(RT_HI - tnd, 1e-4)
    nc = a.shape[0]
    u = U[:, None] * GLU_S[None, :] ** 2                                  # (n, NU)
    du = (2 * U[:, None] * GLU_S[None, :]) * GLU_W[None, :]                # Jacobian * weights
    shp = (nc, NU, NX)
    U3 = jnp.broadcast_to(u[:, :, None], shp).reshape(-1)
    V3 = jnp.broadcast_to(v2[:, None, None], shp).reshape(-1)
    A3 = jnp.broadcast_to(a[:, None, None], shp).reshape(-1)
    W3 = jnp.broadcast_to(GLX_X[None, None, :], shp).reshape(-1)
    f = (ddm_pdf_w(U3, jnp.ones_like(U3), V3, A3, W3) + ddm_pdf_w(U3, jnp.zeros_like(U3), V3, A3, W3)).reshape(shp)
    m2 = jnp.sum(jnp.sum(f * p1w[:, None, :], axis=2) * du, axis=1)
    return m1 + m2


# -----------------------------------------------------------------------------
# Parameter structure. The order of the entries reproduces the order of the sample sites of the
# article's fits (it matters for the dispersed starting values of Model 3).
# -----------------------------------------------------------------------------
_RANEF_MIX = [("v0", 1.0), ("vver", 0.5), ("a0", 0.3), ("aver", 0.2), ("t", 0.5), ("p0", 1.0), ("pver", 0.5),
              ("m", 0.05), ("r", 1.0)]
RANEF = {
    "model1": _RANEF_MIX + [("g", 0.3)],
    "model2": [e for e in _RANEF_MIX if e[0] != "r"] + [("g", 0.3)],
    "model3": _RANEF_MIX + [("v1", 1.0), ("g", 0.3)],
    "ddm_trunc": [("v0", 1.0), ("vver", 0.5), ("a0", 0.3), ("aver", 0.2), ("t", 0.5)],
}
_COMMON = {"b_v": (9, 2.0, 1.5), "b_v_sess": (None, 0, 1), "b_v_jump": (None, 0, 1),           # name: (shape, mean, sd)
           "b_a": (9, float(np.log(0.9)), 0.5), "b_a_sess": (None, 0, 0.3),
           "c_t": (None, 0, 1), "c_t_ver": (None, 0, 0.5), "c_t_sess": (None, 0, 0.5)}
_TIMED = {"b_p": (9, 0, 1.5), "b_p_sess": (None, 0, 1), "b_p_jump": (None, 0, 1),
          "m0": (None, 0, 0.15), "m_ver": (None, 0, 0.1), "m_sess": (None, 0, 0.1),
          "g0": (None, float(np.log(0.1)), 0.5), "g_ver": (None, 0, 0.3), "g_sess": (None, 0, 0.3),
          "r0": (None, 2.0, 1.0), "r_jump": (None, 0, 1), "r_ver": (None, 0, 1), "r_sess": (None, 0, 1)}
_VISUAL = {"b_v1": (2, 1.0, 1.5), "b_v1_jump": (None, 0, 1), "b_v1_sess": (None, 0, 1), "c_tau": (2, 0, 1)}
FIXED = {
    "model1": _TIMED,
    "model2": {k: v for k, v in _TIMED.items() if k not in ("r0", "r_jump", "r_ver", "r_sess")},
    "model3": {**_TIMED, **_VISUAL},
    "ddm_trunc": {},
}
MIXTURES = ("model1", "model2", "model3")

REPARAM = os.environ.get("FIT_REPARAM", "0") == "1"
JITTER = float(os.environ.get("FIT_JITTER", "0"))
WIDE = ("c_tau", "c_tau_id", "b_p", "b_p_id")
VER_CELL = jnp.asarray([-0.5] * 4 + [0.5] * 5)   # level of the cells (B1-B4 = -0.5, L1-L5 = +0.5)
# child effect -> (population intercept that absorbs its mean, covariate of the effect)
ANCHOR = {"v0": ("b_v", "1"), "vver": ("b_v", "ver"), "a0": ("b_a", "1"), "aver": ("b_a", "ver"),
          "t": ("c_t", "1"), "p0": ("b_p", "1"), "pver": ("b_p", "ver"), "m": ("m0", "1"), "r": ("r0", "1"),
          "v1": ("b_v1", "1"), "g": ("g0", "1")}


@functools.lru_cache(maxsize=None)
def helmert(n):
    """Orthonormal basis (n x n-1) of the subspace orthogonal to the vector of ones."""
    H = np.zeros((n, n - 1))
    for j in range(1, n):
        H[:j, j - 1] = 1.0 / np.sqrt(j * (j + 1))
        H[j, j - 1] = -j / np.sqrt(j * (j + 1))
    return H


def sample_sites_reparam(m, n_subj):
    """Same model as sample_sites, in coordinates without ridges (FIT_REPARAM=1). Child effects
    z ~ N(0, I) are written as z = H w + zeta (H orthogonal to the vector of ones, zeta = mean of the
    effects), and each population intercept absorbs the mean of the effects added to it
    (b_id = b + sd * zeta * covariate). The original b and z are stored as deterministic sites."""
    H = jnp.asarray(helmert(n_subj))
    L, shift = {}, {}
    for k, s in RANEF[m]:
        L[f"sd_{k}"] = numpyro.sample(f"sd_{k}", dist.HalfNormal(s))
    for k, _ in RANEF[m]:
        w = numpyro.sample(f"w_{k}", dist.Normal(0.0, 1.0).expand([n_subj - 1]).to_event(1))
        zeta = numpyro.sample(f"zeta_{k}", dist.Normal(0.0, 1.0 / np.sqrt(n_subj)))
        L[f"z_{k}"] = H @ w
        numpyro.deterministic(f"z_{k}", L[f"z_{k}"] + zeta)
        name, cov = ANCHOR[k]
        shift[name] = shift.get(name, 0.0) + L[f"sd_{k}"] * zeta * (VER_CELL if cov == "ver" else 1.0)
    for name, (shape, mu, sd) in {**_COMMON, **FIXED[m]}.items():
        prior = dist.Normal(mu, sd)
        if name in shift:
            bid = numpyro.sample(f"{name}_id", dist.ImproperUniform(dist.constraints.real, (),
                                                                     (shape,) if shape else ()))
            b = bid - shift[name]
            numpyro.factor(f"prior_{name}", jnp.sum(prior.log_prob(b)))     # original prior on b (Jacobian 1)
            numpyro.deterministic(name, b)
            L[name] = bid
        else:
            L[name] = numpyro.sample(name, prior.expand([shape]) if shape else prior)
    return L


def sample_sites(m, n_subj):
    if REPARAM:
        return sample_sites_reparam(m, n_subj)
    S = {}
    for name, (shape, mu, sd) in {**_COMMON, **FIXED[m]}.items():
        d = dist.Normal(mu, sd)
        S[name] = numpyro.sample(name, d.expand([shape]) if shape else d)
    for k, s in RANEF[m]:
        S[f"sd_{k}"] = numpyro.sample(f"sd_{k}", dist.HalfNormal(s))
    with numpyro.plate("subj", n_subj):
        for k, _ in RANEF[m]:
            S[f"z_{k}"] = numpyro.sample(f"z_{k}", dist.Normal(0.0, 1.0))
    return S


def params(m, S, D):
    """Parameters per combination (child x session x cell x obstacle type)."""
    u = {k: S[f"sd_{k}"] * S[f"z_{k}"] for k, _ in RANEF[m]}
    cs, cell, sess, ver, jump, lev = D["cs"], D["ccell"], D["csess"], D["cver"], D["cjump"], D["clev"]
    P = {"v": S["b_v"][cell] + S["b_v_sess"] * sess + S["b_v_jump"] * jump + u["v0"][cs] + u["vver"][cs] * ver,
         "a": jnp.exp(S["b_a"][cell] + S["b_a_sess"] * sess + u["a0"][cs] + u["aver"][cs] * ver)}
    eta_t = S["c_t"] + S["c_t_ver"] * ver + S["c_t_sess"] * sess + u["t"][cs]
    P["t"] = MD.T_MIN + (T_MAX - MD.T_MIN) * expit(eta_t)
    if m in MIXTURES:
        P["pi"] = S["b_p"][cell] + S["b_p_sess"] * sess + S["b_p_jump"] * jump + u["p0"][cs] + u["pver"][cs] * ver
        P["mu"] = S["m0"] + S["m_ver"] * ver + S["m_sess"] * sess + u["m"][cs]
        P["sdA"] = jnp.exp(S["g0"] + S["g_ver"] * ver + S["g_sess"] * sess + u["g"][cs])
        # Model 2: logit 0 = timed responses correct with probability .5
        P["pA"] = (S["r0"] + S["r_jump"] * jump + S["r_ver"] * ver + S["r_sess"] * sess + u["r"][cs]
                   if "r" in u else jnp.zeros_like(P["mu"]))
    if m == "model3":
        P["v1"] = S["b_v1"][lev] + S["b_v1_jump"] * jump + S["b_v1_sess"] * sess + u["v1"][cs]
        P["tau"] = TAU_MIN + (TAU_MAX - TAU_MIN) * expit(S["c_tau"][lev])
        # smooth cap on the boundary (A_CAP * tanh(a / A_CAP)): avoids regions (a ~ 100) where the
        # exponentials of the second stage overflow; not negligible (a = 2 -> 1.90), so the boundary of
        # Model 3 is reported on this effective scale
        P["a"] = A_CAP * jnp.tanh(P["a"] / A_CAP)
    return P


def loglik(m, P, D):
    """Pointwise log-likelihood. Every density is averaged over +-5 ms in all models."""
    rt, cor, c = D["rt"], D["correct"], D["c"]
    v, a, t = P["v"], P["a"], P["t"]
    if m in ("model1", "model2"):
        return MD.m1_loglik(rt, cor, c, v, a, t, P["pi"], P["mu"], P["sdA"], P["pA"])
    if m == "ddm_trunc":
        # density conditioned on the window [CUT, RT_HI]: faster responses are excluded by construction
        Z = jnp.maximum(ddm_mass(v, a, CUT - t, RT_HI - t), 1e-12)
        f = binned(lambda x: MD.ddm_pdf(x - t[c], cor, v[c], a[c]), rt) / Z[c]
        return jnp.logaddexp(jnp.log(LAPSE * 0.5 / (RT_HI - CUT)), jnp.log1p(-LAPSE) + jnp.log(jnp.maximum(f, 1e-30)))
    if m == "model3":
        lapse = jnp.log(LAPSE * 0.5 / WIN)
        v1, tau = P["v1"], P["tau"]
        p1w = stage1_profile(v1, a, tau) * (a[:, None] * GLX_W[None, :])
        Z = jnp.maximum(twostage_mass_q(v1, v, a, tau, t, p1w), 1e-12)
        fD = twostage_pdf_q(rt, cor, v1[c], v[c], a[c], tau[c], t[c], p1w[c]) / Z[c]
        lD = jnp.log(jnp.maximum(fD, 1e-30))
        mu, sdA = P["mu"][c], P["sdA"][c]
        ltr = log_tn_mass((RT_LO - mu) / sdA, (RT_HI - mu) / sdA)
        tn = binned(lambda x: jnp.exp(-0.5 * ((x - mu) / sdA) ** 2) / (sdA * jnp.sqrt(2 * jnp.pi)), rt)
        lA = (jnp.log(jnp.maximum(tn, 1e-30)) - ltr
              + jnp.where(cor == 1, jax.nn.log_sigmoid(P["pA"][c]), jax.nn.log_sigmoid(-P["pA"][c])))
        ll = jnp.logaddexp(jax.nn.log_sigmoid(P["pi"][c]) + lA, jax.nn.log_sigmoid(-P["pi"][c]) + lD)
        return jnp.logaddexp(lapse, jnp.log1p(-LAPSE) + ll)
    raise ValueError(m)


def make_model(m):
    def model(D):
        S = sample_sites(m, D["n_subj"])
        numpyro.factor("ll", loglik(m, params(m, S, D), D).sum())
    return model


def data_for(n_subj=None, df=None):
    d, combos, data = MD.build_data(n_subj, df)
    data["clev"] = jnp.asarray((combos.ver.values > 0).astype(np.int32))
    return d, combos, data


# -----------------------------------------------------------------------------
# Starting values
# -----------------------------------------------------------------------------
def init_values(m, n_subj):
    """Starting values: posterior medians of a preliminary fit of the mixture model (stored in
    inits/start_values.json) for the shared parameters; child effects at zero."""
    from numpyro.infer import init_to_value
    start = json.load(open(INITS / "start_values.json"))
    vals = {}
    for k in ("b_v", "b_v_sess", "b_v_jump", "b_a", "b_a_sess", "c_t", "c_t_ver", "c_t_sess"):
        vals[k] = jnp.asarray(start[k])
    for k in ("v0", "vver", "a0", "aver", "t"):
        vals[f"sd_{k}"] = jnp.asarray(start[f"sd_{k}"])
    if m in MIXTURES:
        # timed component: without these values NumPyro draws them in (-2, 2) and the centre of the timed
        # responses can start outside the response window (chains stuck in warmup)
        for k in ("b_p", "b_p_sess", "b_p_jump", "m0", "m_ver", "m_sess", "g0", "g_ver", "g_sess",
                  "r0", "r_jump", "r_ver", "r_sess"):
            vals[k] = jnp.asarray(start[k])
        for k in ("p0", "pver", "m", "r"):
            vals[f"sd_{k}"] = jnp.asarray(start[f"sd_{k}"])
    names = {k for k, _ in RANEF[m]}
    if "g" in names:
        vals["sd_g"] = jnp.asarray(0.1)
    if m == "model3":
        vals.update({"b_v1": jnp.array([1.0, 1.0]), "c_tau": jnp.array([0.0, 0.0]),
                     "b_v1_jump": jnp.asarray(0.0), "b_v1_sess": jnp.asarray(0.0), "sd_v1": jnp.asarray(0.3)})
    vals = {k: v for k, v in vals.items() if k in {**_COMMON, **FIXED[m]} or
            (k.startswith("sd_") and k[3:] in names)}
    for k in names:
        vals[f"z_{k}"] = jnp.zeros(n_subj)
        vals[f"w_{k}"] = jnp.zeros(n_subj - 1)      # used only with FIT_REPARAM (zeta = 0: b_id = b)
        vals[f"zeta_{k}"] = jnp.asarray(0.0)
    for name in {ANCHOR[k][0] for k in names}:
        if name in vals:
            vals[f"{name}_id"] = vals[name]
    ftype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    vals = {k: jnp.asarray(v, dtype=ftype) for k, v in vals.items()}
    if JITTER > 0:
        sc = {}
        for name, (shape, mu, sd) in {**_COMMON, **FIXED[m]}.items():
            sc[name] = sc[f"{name}_id"] = sd
        for k in names:
            sc[f"zeta_{k}"] = 1.0 / np.sqrt(n_subj)
        return functools.partial(init_jitter, values=vals, scale=JITTER, scales=sc)
    return init_to_value(values=vals)


def init_jitter(site=None, values=None, scale=0.0, scales=None):
    """As init_to_value, plus a N(0, scale * prior scale) perturbation in unconstrained space, different
    for each chain (it uses the random key of the site); wider for the parameters in WIDE."""
    from numpyro.infer import init_to_uniform
    if site is None:
        return functools.partial(init_jitter, values=values, scale=scale, scales=scales)
    if site["type"] == "sample" and not site["is_observed"]:
        if site["name"] not in values:
            return init_to_uniform(site)
        v = values[site["name"]]
        tr = dist.biject_to(site["fn"].support)
        sc = scale * (scales or {}).get(site["name"], 1.0) * (3.0 if site["name"] in WIDE else 1.0)
        return tr(tr.inv(v) + sc * jax.random.normal(site["kwargs"]["rng_key"], jnp.shape(v), dtype=v.dtype))


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------
def fit(m, tag, n_subj=None, warmup=1500, samples=1000, chains=4, df=None, max_tree_depth=10):
    """Fits model m and saves results/fits/ll_{tag}.npy (pointwise log-likelihood, chains x draws x
    responses) and results/fits/post_{tag}.npz (draws by chain, plus diagnostics in `_diag`:
    R-hat max, ESS min, divergences, mean leapfrog steps, share of iterations at max tree depth)."""
    d, combos, D = data_for(n_subj, df)
    n_static = {"n_subj": D.pop("n_subj")}
    print(f"{tag}: responses {len(d)}, children {n_static['n_subj']}, combinations {len(combos)}; {jax.devices()}")
    method = "vectorized" if jax.default_backend() == "gpu" or chains == 1 else "parallel"
    print(f"{tag}: warmup {warmup}, draws {samples}, chains {chains}, max tree depth {max_tree_depth}, "
          f"reparameterisation {'yes' if REPARAM else 'no'}, jitter {JITTER}")
    model = make_model(m)
    kernel = NUTS(lambda D: model({**D, **n_static}), target_accept_prob=0.9, max_tree_depth=max_tree_depth,
                  init_strategy=init_values(m, n_static["n_subj"]))
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method=method, progress_bar=True)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(2026), D, extra_fields=("diverging", "num_steps"))
    minutes = (time.time() - t0) / 60
    samples_ = mcmc.get_samples(group_by_chain=True)
    from numpyro.diagnostics import summary
    sm = summary({k: v for k, v in samples_.items() if not k.startswith(("z_", "w_"))}, group_by_chain=True)
    rhat = max(float(np.nanmax(np.asarray(s["r_hat"]))) for s in sm.values())
    ess = min(float(np.nanmin(np.asarray(s["n_eff"]))) for s in sm.values())
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    steps = np.asarray(mcmc.get_extra_fields()["num_steps"])
    at_max = float((steps >= 2 ** max_tree_depth - 1).mean())
    print(f"{tag}: R-hat max {rhat:.3f}, ESS min {ess:.0f}, divergences {div}, mean steps {steps.mean():.0f}, "
          f"iterations at max tree depth {100 * at_max:.0f}%, time {minutes:.0f} min")
    if "c_tau" in samples_:
        tau_ms = 1000 * (TAU_MIN + (TAU_MAX - TAU_MIN) * expit(np.asarray(samples_["c_tau"]))).mean(axis=(0, 1))
        print(f"  tau (ms) {np.round(tau_ms).tolist()}, v1 {np.round(np.asarray(samples_['b_v1']).mean(axis=(0, 1)), 2).tolist()}")

    # pointwise log-likelihood for PSIS-LOO, in blocks of draws
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in samples_.items()}
    n_draw = next(iter(flat.values())).shape[0]
    ll_fn = jax.jit(jax.vmap(lambda S: loglik(m, params(m, S, D), D)))
    blocks = [np.asarray(ll_fn({k: v[i:i + 5] for k, v in flat.items()}), dtype=np.float32)
              for i in range(0, n_draw, 5)]
    ll = np.concatenate(blocks).reshape(chains, -1, len(d))
    np.save(FITS / f"ll_{tag}.npy", ll)
    np.savez(FITS / f"post_{tag}.npz", **{k: np.asarray(v) for k, v in samples_.items()},
             _diag=np.array([rhat, ess, div, steps.mean(), at_max]))
    return mcmc


# -----------------------------------------------------------------------------
# Helpers for saved fits
# -----------------------------------------------------------------------------
def good_chains(post):
    """Chains that moved: a chain stuck in warmup has zero variance."""
    v = post["b_v"].reshape(post["b_v"].shape[0], post["b_v"].shape[1], -1)
    s = np.median(v.std(axis=1), axis=1)
    return np.where(s > 0.01 * np.median(s))[0]


def diagnostics(post, chains):
    """R-hat max and bulk ESS min over population parameters and SDs, for the given chains."""
    from numpyro.diagnostics import summary
    sm = summary({k: post[k][chains] for k in post.files if not k.startswith(("z_", "u_", "w_")) and k != "_diag"},
                 group_by_chain=True)
    rhat = max(float(np.nanmax(np.asarray(x["r_hat"]))) for x in sm.values())
    ess = min(float(np.nanmin(np.asarray(x["n_eff"]))) for x in sm.values())
    return rhat, ess


def load_draws(tag, chains=None):
    """Draws of a saved fit, chains pooled: {name: (n_draws, ...)}. chains: indices to keep
    (default: good_chains)."""
    z = np.load(FITS / f"post_{tag}.npz")
    ok = good_chains(z) if chains is None else np.asarray(chains)
    return {k: z[k][ok].reshape(-1, *z[k].shape[2:]) for k in z.files if k != "_diag"}
