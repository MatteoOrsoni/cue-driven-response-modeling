# =============================================================================
# Step 9. Formal model comparison with PSIS-LOO on all 42,458 responses (Results: "Model comparison",
# Table 2).
#
#   Model 1 and Model 2: LOO on the pooled chains.
#   Model 3: its posterior has several separate modes, so the chains do not converge with one another.
#     Chains are grouped into modes (chains whose mean onset of visual accumulation, tau, in each level
#     and mean non-decision time agree within MODE_TOL ms); the best mode is the one with the highest
#     mean log-likelihood. LOO is computed for each chain and for the pooled chains of the best mode
#     (one draw every MODE_THIN), and R-hat / ESS are computed within the best mode. The chains of the
#     best mode are saved for 11_posterior_predictive.py.
#   Also reports the estimates of Model 3 (best mode) and the share of timed responses in Model 2.
#
#   python code/09_model_comparison.py
# Output: results/model_comparison.json, results/table2_model_comparison.csv
# =============================================================================
import json
import os

import numpy as np
import pandas as pd
from scipy.special import expit

import models as C
from config import CELLS, FITS, RESULTS, summarise
from loo_utils import diff, loo_safe

MODE_TOL = float(os.environ.get("MODE_TOL", 20))      # ms
MODE_THIN = int(os.environ.get("MODE_THIN", 4))
DESC = {"Model 1": "Timed component + diffusion starting at the cue",
        "Model 2": "As Model 1, timed responses correct at chance",
        "Model 3": "As Model 1, diffusion starting with the visual information"}
TAGS = {"Model 1": "model1", "Model 2": "model2", "Model 3": "model3"}


def tau_ms(e):
    return 1000 * (C.TAU_MIN + (C.TAU_MAX - C.TAU_MIN) * expit(e))


def t_ms(e):
    return 1000 * (C.MD.T_MIN + (C.T_MAX - C.MD.T_MIN) * expit(e))


def n_params(post):
    """Nominal parameters: population + child effects (one per child)."""
    skip = ("z_", "w_", "zeta_", "u_")
    pop = sum(int(np.prod(post[k].shape[2:])) for k in post.files
              if k != "_diag" and not k.startswith(skip) and not k.endswith("_id"))
    child = sum(int(np.prod(post[k].shape[2:])) for k in post.files if k.startswith("z_"))
    return pop + child


def modes(z, ll, chains):
    """Groups chains into modes and orders the modes by mean log-likelihood (best first)."""
    feats = np.array([[tau_ms(z["c_tau"][c][:, 0]).mean(), tau_ms(z["c_tau"][c][:, 1]).mean(),
                       t_ms(z["c_t"][c]).mean()] for c in chains])
    loglik = np.array([float(np.asarray(ll[c, ::50]).sum(-1).mean()) for c in chains])
    groups = []
    for i, c in enumerate(chains):
        for g in groups:
            if np.max(np.abs(feats[g[0]] - feats[i])) < MODE_TOL:
                g.append(i)
                break
        else:
            groups.append([i])
    groups.sort(key=lambda g: -loglik[g].mean())
    return [[int(chains[i]) for i in g] for g in groups], feats, loglik


def model3_estimates(z, ch):
    P = {k: z[k][ch].reshape(-1, *z[k].shape[2:]) for k in z.files if k != "_diag"}
    return {
        "tau Level 1 (ms before the cue)": summarise(tau_ms(P["c_tau"][:, 0])),
        "tau Level 2 (ms before the cue)": summarise(tau_ms(P["c_tau"][:, 1])),
        "visual drift v1, Level 1": summarise(P["b_v1"][:, 0]),
        "visual drift v1, Level 2": summarise(P["b_v1"][:, 1]),
        "visual drift v1, Level 2 - Level 1": summarise(P["b_v1"][:, 1] - P["b_v1"][:, 0]),
        "non-decision time (ms)": summarise(t_ms(P["c_t"])),
        "drift v after the cue (mean of cells)": summarise(P["b_v"].mean(1)),
        "effective boundary a_eff (mean of cells)": summarise((C.A_CAP * np.tanh(np.exp(P["b_a"]) / C.A_CAP)).mean(1)),
        "P(timed) (mean of cells)": summarise(expit(P["b_p"]).mean(1)),
        "timed centre (ms)": summarise(1000 * P["m0"]),
    }


if __name__ == "__main__":
    L, info, R = {}, {}, {}
    for name, tag in TAGS.items():
        post = np.load(FITS / f"post_{tag}.npz")
        ok = C.good_chains(post)
        ll = np.load(FITS / f"ll_{tag}.npy", mmap_mode="r")
        rhat, ess = C.diagnostics(post, ok)
        info[name] = {"chains": ok, "rhat_all_chains": rhat, "ess_all_chains": ess, "div": int(post["_diag"][2]),
                      "n_par": n_params(post)}
        if name != "Model 3":
            L[name] = loo_safe(np.asarray(ll[ok]))
            continue
        # Model 3: per chain, then modes
        per = {int(c): loo_safe(np.asarray(ll[c:c + 1])) for c in ok}
        groups, feats, loglik = modes(post, ll, ok)
        best = groups[0]
        L[name] = loo_safe(np.asarray(ll[best][:, ::MODE_THIN]))
        rh_b, ess_b = C.diagnostics(post, np.asarray(best))
        info[name].update({"per_chain": per, "modes": groups, "best_mode": best, "rhat_best_mode": rh_b,
                           "ess_best_mode": ess_b,
                           "chain_features": [{"chain": int(c), "tau L1 (ms)": float(f[0]), "tau L2 (ms)": float(f[1]),
                                               "t (ms)": float(f[2]), "mean loglik": float(l)}
                                              for c, f, l in zip(ok, feats, loglik)],
                           "estimates": model3_estimates(post, np.asarray(best))})
        del ll
        print(f"Model 3 modes (chains): {groups}; best mode {best}")

    base = L["Model 1"]
    rows = []
    for name in TAGS:
        lo = L[name]
        d, se = diff(lo, base) if name != "Model 1" else (0.0, 0.0)
        i = info[name]
        three = name == "Model 3"
        R[name] = {"description": DESC[name],
                   "chains_used": f"{len(i['best_mode']) if three else len(i['chains'])}/{len(i['chains'])}",
                   "elpd_loo": lo.elpd, "se": lo.se, "d_elpd_vs_model1": d, "se_d": se, "p_loo": lo.p,
                   "k_gt_0.7": int((lo.pareto_k > 0.7).sum()), "n_nominal_parameters": i["n_par"],
                   "rhat_max": i["rhat_best_mode"] if three else i["rhat_all_chains"],
                   "ess_min": i["ess_best_mode"] if three else i["ess_all_chains"],
                   "divergences": i["div"]}
        if three:
            R[name].update({
                "rhat_all_chains": i["rhat_all_chains"], "modes": i["modes"], "best_mode": i["best_mode"],
                "per_chain": [{"chain": c, "elpd": x.elpd, "p_loo": x.p, "k_gt_0.7": int((x.pareto_k > 0.7).sum()),
                               "d_elpd_vs_model1": diff(x, base)[0], "se_d": diff(x, base)[1]}
                              for c, x in i["per_chain"].items()],
                "chain_features": i["chain_features"], "estimates_best_mode": i["estimates"]})
        rows.append({"Model": name, "Description": DESC[name], "Chains used": R[name]["chains_used"],
                     "elpd_loo": round(lo.elpd), "SE": round(lo.se), "delta_elpd": round(d), "SE_delta": round(se),
                     "p_loo": round(lo.p), "k>0.7": R[name]["k_gt_0.7"], "R-hat max": round(R[name]["rhat_max"], 2)})

    # share of timed responses in Model 2 (by cell)
    z2 = C.load_draws("model2")
    R["Model 2"]["timed share by cell (median)"] = {c: float(np.median(expit(z2["b_p"][:, j]))) for j, c in enumerate(CELLS)}

    pd.DataFrame(rows).to_csv(RESULTS / "table2_model_comparison.csv", index=False)
    json.dump(R, open(RESULTS / "model_comparison.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(R["Model 3"]["estimates_best_mode"], indent=1))
