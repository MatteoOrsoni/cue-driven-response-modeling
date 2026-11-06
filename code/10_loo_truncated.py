# =============================================================================
# Step 10. Truncated DDM vs Model 1 on the responses given at least 150 ms after the cue (Method and
# Results: "Model comparison", last paragraph).
#
# The hierarchical truncated DDM predicts only responses in [150, 500] ms. Model 1 is evaluated on the
# same responses with its predictive density conditioned on a response falling in that window:
#   log p(rt, correct | rt in window) = log p(rt, correct) - log P(window | combination),
#   P(window) = lapse * 0.35 / 0.8 + (1 - lapse) * [pi * timed mass + (1 - pi) * diffusion mass].
#
#   python code/10_loo_truncated.py
# Output: results/loo_truncated.json
# =============================================================================
import json

import jax
import jax.numpy as jnp
import numpy as np

import mixture as MD
import models as C
from config import CUT, FITS, RESULTS, RT_HI, RT_LO
from data_prep import load_trials
from loo_utils import diff, loo_safe

THIN = 4                                    # one draw in 4 of Model 1 (1,000 draws)

d_all, combos, D = C.data_for()
D.pop("n_subj")
late = d_all.rt_raw.values >= CUT
d_tr, _, _ = C.data_for(df=load_trials().pipe(lambda t: t[t.rt_raw >= CUT]))
assert late.sum() == len(d_tr) and np.allclose(d_all.rt_raw.values[late], d_tr.rt_raw.values), "data not aligned"

z = np.load(FITS / "post_model1.npz")
ok = C.good_chains(z)
ll = np.load(FITS / "ll_model1.npy")[ok][:, ::THIN]                           # (chains, draws, responses)
S = {k: z[k][ok][:, ::THIN].reshape(-1, *z[k].shape[2:]) for k in z.files if k != "_diag"}


@jax.jit
def log_mass(s):
    """log P(response in [CUT, RT_HI]) per combination under Model 1."""
    P = C.params("model1", s, D)
    mu, sd = P["mu"], P["sdA"]
    tm = jnp.exp(C.log_tn_mass((CUT - mu) / sd, (RT_HI - mu) / sd) - C.log_tn_mass((RT_LO - mu) / sd, (RT_HI - mu) / sd))
    dm = C.ddm_mass(P["v"], P["a"], CUT - P["t"], RT_HI - P["t"]) / jnp.maximum(
        MD.ddm_mass_window(P["v"], P["a"], P["t"]), 1e-12)
    pi = jax.nn.sigmoid(P["pi"])
    return jnp.log(C.LAPSE * (RT_HI - CUT) / C.WIN + (1 - C.LAPSE) * (pi * tm + (1 - pi) * dm))


n = next(iter(S.values())).shape[0]
lm = np.stack([np.asarray(log_mass({k: v[i] for k, v in S.items()})) for i in range(n)])     # (draws, combinations)
c_late = d_all.c.values[late]
ll_cond = ll[..., late] - lm.reshape(ll.shape[0], ll.shape[1], -1)[..., c_late]

zt = np.load(FITS / "post_ddm_trunc_all.npz")
ll_tr = np.load(FITS / "ll_ddm_trunc_all.npy")[C.good_chains(zt)]

L = {"Model 1, conditional on 150-500 ms": loo_safe(ll_cond), "hierarchical truncated DDM": loo_safe(ll_tr)}
R = {"responses": int(late.sum()), "truncated DDM R-hat max": float(zt["_diag"][0])}
for k, lo in L.items():
    print(f"{k}: elpd {lo.elpd:.1f} (SE {lo.se:.1f}), p_loo {lo.p:.1f}, Pareto k > 0.7: {int((lo.pareto_k > 0.7).sum())}")
    R[k] = {"elpd": lo.elpd, "se": lo.se, "p_loo": lo.p, "k_gt_0.7": int((lo.pareto_k > 0.7).sum())}
dl, se = diff(L["Model 1, conditional on 150-500 ms"], L["hierarchical truncated DDM"])
R["delta elpd (Model 1 - truncated DDM)"] = [dl, se]
print(f"Model 1 - truncated DDM on the {late.sum()} late responses: delta elpd {dl:.1f} (SE {se:.1f})")
json.dump(R, open(RESULTS / "loo_truncated.json", "w"), indent=1)
