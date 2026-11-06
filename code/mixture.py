# =============================================================================
# Model 1 (timed-diffusion mixture): densities, data structure and NumPyro model.
#
# For each response (rt in s relative to the auditory cue, window [-0.3, 0.5]):
#   with probability pi      -> TIMED response, released at the visible obstacle:
#                               rt ~ truncated Normal(mu_T, sigma_T) on [-0.3, 0.5],
#                               P(correct) = p_T
#   with probability 1 - pi  -> DIFFUSION response: Wiener diffusion (accuracy coding, z = a/2)
#                               whose time origin is the cue, conditioned on a response within 0.5 s
#   plus a fixed uniform contaminant (1%), and a likelihood averaged over a 10-ms window
#   (Equation 1 of the article).
#
# Structure (s = child; cell = level x minute, 9 cells; sess and ver = +-0.5; jump = +-0.5):
#   v           = b_v[cell] + b_v_sess*sess + b_v_jump*jump + u_v0[s] + u_vver[s]*ver
#   log a       = b_a[cell] + b_a_sess*sess + u_a0[s] + u_aver[s]*ver
#   t           = 0.35 * sigmoid(c_t + c_t_ver*ver + c_t_sess*sess + u_t[s])        (0-350 ms)
#   logit pi    = b_p[cell] + b_p_sess*sess + b_p_jump*jump + u_p0[s] + u_pver[s]*ver
#   mu_T        = m0 + m_ver*ver + m_sess*sess + u_m[s]
#   log sigma_T = g0 + g_ver*ver + g_sess*sess + u_g[s]
#   logit p_T   = r0 + r_jump*jump + r_ver*ver + r_sess*sess + u_r[s]
# Child effects u = sd * z, z ~ N(0, 1), independent (non-centred). Priors: Table A2 of the article.
#
# The model is fitted by 06_fit_model1.py. Parameter names of the posterior are those of the saved
# model outputs.
# =============================================================================
import os

# on CPU, several devices are needed to run the chains in parallel (must be set before importing jax)
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import expit
from numpyro.infer import MCMC, NUTS, init_to_median

from config import CELLS, FITS, RT_HI, RT_LO
from data_prep import load_trials

# Precision. The article's fits used float32 on GPU (MIX_X64=0, the default here). MIX_X64=1 switches
# to float64 (recommended for small CPU tests only: ~64 times slower on consumer GPUs).
if os.environ.get("MIX_X64", "0") == "1":
    jax.config.update("jax_enable_x64", True)

T_MIN, T_MAX = 0.0, 0.35  # bounds of the non-decision time (s)
N_GRID = 96               # grid points for the normalising integral of the diffusion density
LAPSE = 0.01              # uniform contaminant (Ratcliff & Tuerlinckx, 2002)
H_RES = 0.010             # temporal resolution: likelihood = mean density over +-5 ms
N_GL = 3                  # Gauss-Legendre points for that mean
GL_X, GL_W = np.polynomial.legendre.leggauss(N_GL)
GL_X, GL_W = jnp.asarray(GL_X), jnp.asarray(GL_W)
WIN = RT_HI - RT_LO       # width of the response window (0.8 s)

# child effects of Model 1 (name, HalfNormal scale of their SD)
RANEF = [("v0", 1.0), ("vver", 0.5), ("a0", 0.3), ("aver", 0.2), ("t", 0.5),
         ("p0", 1.0), ("pver", 0.5), ("m", 0.05), ("r", 1.0), ("g", 0.3)]

KS_SMALL = jnp.arange(-6, 7)
KS_LARGE = jnp.arange(1, 15)


# ----------------------------------------------------------------------------
# Wiener first-passage density (Navarro & Fuss, 2009)
# ----------------------------------------------------------------------------
def _fstd(u, w):
    """Standardised first-passage density at the lower boundary, u = tau / a^2 (small- and
    large-time series)."""
    u = jnp.maximum(u, 1e-8)
    wk = w[..., None] + 2 * KS_SMALL
    small = jnp.sum(wk * jnp.exp(-wk ** 2 / (2 * u[..., None])), -1) / jnp.sqrt(2 * jnp.pi * u ** 3)
    k = KS_LARGE
    large = jnp.pi * jnp.sum(k * jnp.exp(-(k ** 2) * jnp.pi ** 2 * u[..., None] / 2)
                             * jnp.sin(k * jnp.pi * w[..., None]), -1)
    return jnp.where(u < 1.0, small, large)


def ddm_pdf(tau, correct, v, a):
    """Density of a correct (upper boundary) or incorrect response at decision time tau, z = a/2."""
    w = jnp.full_like(tau, 0.5)
    vv = jnp.where(correct == 1, -v, v)     # upper boundary: v -> -v (w is symmetric)
    pos = tau > 0
    tau_s = jnp.where(pos, tau, 1.0)
    expo = jnp.exp(-vv * a * w - vv ** 2 * tau_s / 2)
    dens = expo / a ** 2 * _fstd(tau_s / a ** 2, w)
    return jnp.where(pos, jnp.maximum(dens, 0.0), 0.0)


def ddm_mass_window(v, a, t):
    """Z: probability that the diffusion process responds before RT_HI (trapezoidal rule)."""
    span = jnp.maximum(RT_HI - t, 1e-4)
    g = jnp.linspace(0.0, 1.0, N_GRID + 1)[1:]
    tau = span[:, None] * g[None, :]
    vb, ab = v[:, None] * jnp.ones_like(tau), a[:, None] * jnp.ones_like(tau)
    f = ddm_pdf(tau, jnp.ones_like(tau), vb, ab) + ddm_pdf(tau, jnp.zeros_like(tau), vb, ab)
    f = jnp.concatenate([jnp.zeros((f.shape[0], 1)), f], 1)        # f(0) = 0
    return jnp.trapezoid(f, dx=span[:, None] / N_GRID, axis=1).squeeze()


def binned(pdf_fn, rt, h=H_RES):
    """Mean of a smooth density over [rt - h/2, rt + h/2] (Gauss-Legendre)."""
    pts = rt[:, None] + (h / 2) * GL_X[None, :]
    vals = jnp.stack([pdf_fn(pts[:, i]) for i in range(N_GL)], axis=1)
    return 0.5 * jnp.sum(vals * GL_W[None, :], axis=1)


def log_tn_mass(lo_z, hi_z):
    """log(Phi(hi_z) - Phi(lo_z)), stable (hi_z > lo_z): lower tail by symmetry and, for tiny
    intervals, a Taylor expansion that avoids cancellation."""
    flip = lo_z > 0
    a_ = jnp.where(flip, -hi_z, lo_z)
    b_ = jnp.where(flip, -lo_z, hi_z)
    la, lb = jax.scipy.special.log_ndtr(a_), jax.scipy.special.log_ndtr(b_)
    d = jnp.minimum(la - lb, -1e-12)
    exact = lb + jnp.log(-jnp.expm1(d))
    w = hi_z - lo_z
    mid = (hi_z + lo_z) / 2
    taylor = (jnp.log(jnp.maximum(w, 1e-30)) - 0.5 * mid ** 2 - 0.5 * jnp.log(2 * jnp.pi)
              + jnp.log1p(w ** 2 * (mid ** 2 - 1) / 24))
    return jnp.where(w < 1e-2, taylor, exact)


def m1_loglik(rt, correct, c, v, a, t, logit_pi, mu_A, sd_A, logit_pA):
    """Pointwise log-likelihood of Equation 1. Parameters are given per combination
    (child x session x cell x obstacle type) and indexed by c."""
    Z = jnp.maximum(ddm_mass_window(v, a, t), 1e-12)
    fR = binned(lambda x: ddm_pdf(x - t[c], correct, v[c], a[c]), rt) / Z[c]
    lR = jnp.log(jnp.maximum(fR, 1e-30))
    mu, sdA = mu_A[c], sd_A[c]
    ltr = log_tn_mass((RT_LO - mu) / sdA, (RT_HI - mu) / sdA)
    tn = binned(lambda x: jnp.exp(-0.5 * ((x - mu) / sdA) ** 2) / (sdA * jnp.sqrt(2 * jnp.pi)), rt)
    lA = (jnp.log(jnp.maximum(tn, 1e-30)) - ltr
          + jnp.where(correct == 1, jax.nn.log_sigmoid(logit_pA[c]), jax.nn.log_sigmoid(-logit_pA[c])))
    ll = jnp.logaddexp(jax.nn.log_sigmoid(logit_pi[c]) + lA, jax.nn.log_sigmoid(-logit_pi[c]) + lR)
    return jnp.logaddexp(jnp.log(LAPSE * 0.5 / WIN), jnp.log1p(-LAPSE) + ll)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def build_data(n_subj=None, df=None):
    """Responses (omissions removed) and the combinations child x session x cell x obstacle type.
    df: obstacles (default: all obstacles of load_trials()); n_subj: first n children only (tests)."""
    if df is None:
        df = load_trials()
    d = df[df.omission == 0].copy()
    if n_subj:
        keep = d.subj.drop_duplicates().iloc[:n_subj]
        d = d[d.subj.isin(keep)].copy()
    d["s"] = d.subj.astype("category").cat.codes
    d["jump"] = np.where(d.obstacle == "JUMP", 0.5, -0.5)
    d["cella_i"] = d.cella.map({c: i for i, c in enumerate(CELLS)})
    key = ["s", "sess", "ver", "cella_i", "jump"]
    combos = d[key].drop_duplicates().reset_index(drop=True)
    combos["c"] = np.arange(len(combos))
    d = d.merge(combos, on=key, how="left")
    fdt = np.float64 if jax.config.jax_enable_x64 else np.float32
    to = lambda x, t=None: jnp.asarray(np.asarray(x, dtype=t or fdt))
    data = {
        "rt": to(d.rt_raw), "correct": to(d.correct, np.int32), "c": to(d.c, np.int32),
        "cs": to(combos.s, np.int32), "csess": to(combos.sess), "cver": to(combos.ver),
        "ccell": to(combos.cella_i, np.int32), "cjump": to(combos.jump),
        "n_subj": int(d.s.max() + 1),
    }
    return d, combos, data


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def model(rt, correct, c, cs, csess, cver, ccell, cjump, n_subj):
    N01 = lambda s=1.0: dist.Normal(0.0, s)
    hn = lambda s: dist.HalfNormal(s)

    # population-level parameters (the order of the sample sites is kept as in the article's fits)
    b_v = numpyro.sample("b_v", dist.Normal(2.0, 1.5).expand([9]))
    b_v_sess = numpyro.sample("b_v_sess", N01())
    b_v_jump = numpyro.sample("b_v_jump", N01())
    b_a = numpyro.sample("b_a", dist.Normal(np.log(0.9), 0.5).expand([9]))
    b_a_sess = numpyro.sample("b_a_sess", N01(0.3))
    c_t = numpyro.sample("c_t", N01())
    c_t_ver = numpyro.sample("c_t_ver", N01(0.5))
    c_t_sess = numpyro.sample("c_t_sess", N01(0.5))
    b_p = numpyro.sample("b_p", N01(1.5).expand([9]))
    b_p_sess = numpyro.sample("b_p_sess", N01())
    b_p_jump = numpyro.sample("b_p_jump", N01())
    m0 = numpyro.sample("m0", dist.Normal(0.0, 0.15))
    m_ver = numpyro.sample("m_ver", N01(0.1))
    m_sess = numpyro.sample("m_sess", N01(0.1))
    g0 = numpyro.sample("g0", dist.Normal(np.log(0.1), 0.5))
    g_ver = numpyro.sample("g_ver", N01(0.3))
    g_sess = numpyro.sample("g_sess", N01(0.3))
    r0 = numpyro.sample("r0", dist.Normal(2.0, 1.0))
    r_jump = numpyro.sample("r_jump", N01())
    r_ver = numpyro.sample("r_ver", N01())
    r_sess = numpyro.sample("r_sess", N01())

    # child-specific effects (non-centred, independent)
    sd = {k: numpyro.sample(f"sd_{k}", hn(s)) for k, s in RANEF}
    with numpyro.plate("subj", n_subj):
        z = {k: numpyro.sample(f"z_{k}", N01()) for k in sd}
    u = {k: sd[k] * z[k] for k in sd}

    # parameters per combination (child x session x cell x obstacle type)
    v = b_v[ccell] + b_v_sess * csess + b_v_jump * cjump + u["v0"][cs] + u["vver"][cs] * cver
    a = jnp.exp(b_a[ccell] + b_a_sess * csess + u["a0"][cs] + u["aver"][cs] * cver)
    t = T_MIN + (T_MAX - T_MIN) * expit(c_t + c_t_ver * cver + c_t_sess * csess + u["t"][cs])
    logit_pi = b_p[ccell] + b_p_sess * csess + b_p_jump * cjump + u["p0"][cs] + u["pver"][cs] * cver
    mu_A = m0 + m_ver * cver + m_sess * csess + u["m"][cs]
    sd_A = jnp.exp(g0 + g_ver * cver + g_sess * csess + u["g"][cs])
    logit_pA = r0 + r_jump * cjump + r_ver * cver + r_sess * csess + u["r"][cs]
    numpyro.factor("ll", m1_loglik(rt, correct, c, v, a, t, logit_pi, mu_A, sd_A, logit_pA).sum())


def run(tag, df=None, n_subj=None, warmup=1500, samples=1000, chains=4, depth=10):
    """Fits Model 1 and saves, in results/fits/:
    idata_{tag}.nc (posterior), trials_{tag}.csv and combos_{tag}.csv (data as indexed by the model),
    ll_{tag}.npy and post_{tag}.npz (pointwise log-likelihood and draws, for PSIS-LOO)."""
    d, combos, data = build_data(n_subj, df)
    env = lambda k, default: int(os.environ.get(k, default))
    warmup, samples, chains = env("FIT_WARMUP", warmup), env("FIT_SAMPLES", samples), env("FIT_CHAINS", chains)
    depth = env("FIT_DEPTH", depth)
    print(f"{tag}: responses {len(d)}, children {data['n_subj']}, combinations {len(combos)}; device {jax.devices()}")
    print(f"warmup {warmup}, draws {samples}, chains {chains}, max tree depth {depth}")
    method = "vectorized" if jax.default_backend() == "gpu" or chains == 1 else "parallel"
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9, max_tree_depth=depth, init_strategy=init_to_median),
                num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method=method, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(2026), **data, extra_fields=("diverging", "num_steps"))
    mcmc.print_summary(exclude_deterministic=True)
    import arviz as az
    idata = az.from_numpyro(mcmc)
    idata.to_netcdf(FITS / f"idata_{tag}.nc")
    d.to_csv(FITS / f"trials_{tag}.csv", index=False)
    combos.to_csv(FITS / f"combos_{tag}.csv", index=False)
    save_for_loo(mcmc, data, combos, tag, depth)
    return mcmc


def save_for_loo(mcmc, data, combos, tag, depth):
    """Pointwise log-likelihood and draws in the format of models.py (ll_{tag}.npy, post_{tag}.npz),
    so that the main fit is also the Model 1 of the model comparison."""
    import models as C
    from numpyro.diagnostics import summary
    S = mcmc.get_samples(group_by_chain=True)
    sm = summary({k: v for k, v in S.items() if not k.startswith("z_")}, group_by_chain=True)
    rhat = max(float(np.nanmax(np.asarray(s["r_hat"]))) for s in sm.values())
    ess = min(float(np.nanmin(np.asarray(s["n_eff"]))) for s in sm.values())
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    steps = np.asarray(mcmc.get_extra_fields()["num_steps"])
    at_max = float((steps >= 2 ** depth - 1).mean())
    print(f"{tag}: R-hat max {rhat:.3f}, ESS min {ess:.0f}, divergences {div}, mean leapfrog steps "
          f"{steps.mean():.0f}, iterations at max tree depth {100 * at_max:.0f}%")
    D = {k: v for k, v in data.items() if k != "n_subj"}
    D["clev"] = jnp.asarray((combos.ver.values > 0).astype(np.int32))
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in S.items()}
    ll_fn = jax.jit(jax.vmap(lambda s: C.loglik("model1", C.params("model1", s, D), D)))
    n_draw = next(iter(flat.values())).shape[0]
    blocks = [np.asarray(ll_fn({k: v[i:i + 5] for k, v in flat.items()}), dtype=np.float32)
              for i in range(0, n_draw, 5)]
    chains = next(iter(S.values())).shape[0]
    np.save(FITS / f"ll_{tag}.npy", np.concatenate(blocks).reshape(chains, -1, D["rt"].shape[0]))
    np.savez(FITS / f"post_{tag}.npz", **{k: np.asarray(v) for k, v in S.items()},
             _diag=np.array([rhat, ess, div, steps.mean(), at_max]))
