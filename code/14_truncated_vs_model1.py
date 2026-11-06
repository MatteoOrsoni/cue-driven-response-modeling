# =============================================================================
# Step 14. Do the conclusions depend on how early responses are treated? Contrasts of RQ2-RQ4 for the
# diffusion parameters (drift, boundary, non-decision time) from the hierarchical truncated DDM
# (responses >= 150 ms) and from Model 1 (all responses), side by side (Results RQ2-RQ4; Discussion:
# "the two approaches reached different conclusions for five of the 19 contrasts").
#
# A conclusion differs when the 89% CI excludes zero in one model only, or in both with opposite signs.
#
#   python code/14_truncated_vs_model1.py
# Output: results/truncated_vs_model1_contrasts.csv
# =============================================================================
import numpy as np
import pandas as pd

import model1_tools as T
import models as C
from config import RESULTS, summarise

# the 19 contrasts compared in the article (keys of model1_tools.contrasts)
KEYS = ([("RQ2", "drift rate v"), ("RQ2", "log boundary a"), ("RQ2", "non-decision t (ms)")]
        + [("RQ3", f"{p}: {c}") for p in ("v", "log a")
           for c in ("L1 min 2 - 1", "L1 min 4 - 3", "L2 min 2 - 1", "L2 min 4 - 3", "L2 min 5 - 4")]
        + [("RQ4", f"{p}: {c}") for p in ("v", "log a") for c in ("L2 - L1 (min 1-4)", "L2 - L1 (min 1)")]
        + [("RQ4", "non-decision t (ms)"), ("Obstacle", "v: jump - slide")])


def tr_contrasts(P):
    """Same definitions as model1_tools.contrasts, for the parameters the truncated DDM has."""
    dummy = {k: np.zeros(len(P["b_v"])) for k in ("b_p_sess", "m_sess", "g_sess", "r_sess", "m_ver", "g_ver",
                                                   "r_ver", "b_p_jump", "r_jump")}
    dummy["b_p"] = np.zeros_like(P["b_v"])
    return T.contrasts({**dummy, **P})


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    z = np.load(RESULTS / "fits" / "post_ddm_trunc_all.npz")
    Ptr = C.load_draws("ddm_trunc_all")
    _, P1, _, _ = T.load("model1")
    ktr, k1 = tr_contrasts(Ptr), T.contrasts(P1)
    cred = lambda s: s["lo"] > 0 or s["hi"] < 0
    fmt = lambda s: f"{s['median']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]"
    rows = []
    for k in KEYS:
        a, b = summarise(ktr[k]), summarise(k1[k])
        rows.append({"block": k[0], "contrast": k[1], "truncated DDM": fmt(a), "Model 1": fmt(b),
                     "credible, truncated DDM": cred(a), "credible, Model 1": cred(b),
                     "different conclusion": cred(a) != cred(b) or (cred(a) and cred(b) and np.sign(a["median"]) != np.sign(b["median"]))})
    tab = pd.DataFrame(rows)
    print(f"truncated DDM: R-hat max {z['_diag'][0]:.3f}")
    print(tab.to_string(index=False))
    print(f"\ndifferent conclusions: {int(tab['different conclusion'].sum())} of {len(tab)}")
    tab.to_csv(RESULTS / "truncated_vs_model1_contrasts.csv", index=False)
