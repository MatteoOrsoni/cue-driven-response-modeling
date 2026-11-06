# =============================================================================
# Step 13. Split-half reliability (Method: "Parameter recovery and reliability"; Results: "Reliability",
# Table 6). Odd versus even obstacles, so that both halves cover the same minutes of play; Spearman-Brown
# correction; 89% bootstrap intervals over children.
#
#   behavioural indices   omission rate, action-error rate, mean and SD of the timing of correct
#                         responses, proportion of early responses (< 150 ms); per level, also projected
#                         to 154 obstacles for Level 2 (rho_k = k rho / (1 + (k - 1) rho), k = 154 / 204)
#   Model 1               posterior means of the child-specific parameters from separate fits to even and
#                         odd obstacles (06_fit_model1.py half0 / half1)
#   truncated DDM         same, hierarchical truncated DDM (08_fit_alternative_models.py ddm_trunc half0/half1)
#
#   python code/13_reliability.py
# Output: results/reliability_behavioral.csv, results/reliability_model1.csv,
#         results/reliability_ddm_trunc.csv, results/table6_reliability.csv
# =============================================================================
import numpy as np
import pandas as pd

import mixture as MD
import model1_tools as T
import models as C
from config import CUT, FITS, N_OBST, RESULTS
from data_prep import load_trials


def sb(r, k=2.0):
    return k * r / (1 + (k - 1) * r)


# -----------------------------------------------------------------------------
# Behavioural indices
# -----------------------------------------------------------------------------
def indices(g):
    resp = g[g.omission == 0]
    corr = resp[resp.correct == 1]
    return {"omission rate": g.omission.mean(),
            "action-error rate": 1 - resp.correct.mean() if len(resp) else np.nan,
            "timing, mean": 1000 * corr.rt_raw.mean(),
            "timing, SD": 1000 * corr.rt_raw.std(),
            "proportion of early responses": (resp.rt_raw < CUT).mean() if len(resp) else np.nan}


def behavioral(df, rng):
    rows = []
    for (s, ver), g in df.groupby(["subj", "versione"]):
        for half, h in [("odd", g[g.trial_n % 2 == 1]), ("even", g[g.trial_n % 2 == 0]), ("full", g)]:
            rows.append({"subj": s, "versione": ver, "half": half, **indices(h)})
    ind = pd.DataFrame(rows)
    measures = [c for c in ind.columns if c not in ("subj", "versione", "half")]
    res = []
    for ver in ("breve", "lunga"):
        tab = ind[ind.versione == ver].pivot(index="subj", columns="half", values=measures)
        for m in measures:
            o, e = tab[(m, "odd")], tab[(m, "even")]
            ok = o.notna() & e.notna()
            rel = sb(np.corrcoef(o[ok], e[ok])[0, 1])
            boots = []
            subj = tab.index.to_numpy()
            for _ in range(1000):
                b = tab.loc[rng.choice(subj, len(subj))]
                ob, eb = b[(m, "odd")], b[(m, "even")]
                okb = ob.notna() & eb.notna()
                boots.append(sb(np.corrcoef(ob[okb], eb[okb])[0, 1]))
            res.append({"level": "Level 1" if ver == "breve" else "Level 2", "index": m, "children": int(ok.sum()),
                        "SB": rel, "CI89_lo": np.nanquantile(boots, .055), "CI89_hi": np.nanquantile(boots, .945),
                        "SB_at_154_obstacles": sb(rel, N_OBST["breve"] / N_OBST[ver])})
    return pd.DataFrame(res)


# -----------------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------------
def boot_table(h0, h1, rng, n_boot):
    rows = []
    for col in h0.columns:
        x, y = h0[col].values, h1[col].values
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(x), len(x))
            boots.append(sb(np.corrcoef(x[i], y[i])[0, 1]))
        rows.append({"parameter": col, "SB": sb(np.corrcoef(x, y)[0, 1]),
                     "CI89_lo": np.quantile(boots, .055), "CI89_hi": np.quantile(boots, .945), "children": len(x)})
    return pd.DataFrame(rows)


def model1():
    halves = []
    for h in ("model1_half0", "model1_half1"):
        _, P, d, _ = T.load(h)
        subj = d.drop_duplicates("s").sort_values("s").subj.values       # model index -> child
        halves.append(pd.DataFrame(T.child_level_scores(P), index=subj))
    common = halves[0].index.intersection(halves[1].index)
    return boot_table(halves[0].loc[common], halves[1].loc[common], np.random.default_rng(11), 2000)


def ddm_trunc(trials):
    trials = trials[trials.rt_raw >= CUT]
    halves = []
    for k in (0, 1):
        z = np.load(FITS / f"post_ddm_trunc_half{k}.npz")
        ok = C.good_chains(z)
        U = {e: (z[f"sd_{e}"][ok][..., None] * z[f"z_{e}"][ok]).reshape(-1, z[f"z_{e}"].shape[-1]).mean(0)
             for e in ("v0", "vver", "a0", "aver", "t")}
        d, _, _ = MD.build_data(df=trials[trials.trial_n % 2 == k])
        subj = d.drop_duplicates("s").sort_values("s").subj.values
        halves.append(pd.DataFrame({
            "drift, Level 1": U["v0"] - 0.5 * U["vver"], "drift, Level 2": U["v0"] + 0.5 * U["vver"],
            "log a, Level 1": U["a0"] - 0.5 * U["aver"], "log a, Level 2": U["a0"] + 0.5 * U["aver"],
            "non-decision time (logit)": U["t"]}, index=subj))
    common = halves[0].index.intersection(halves[1].index)
    return boot_table(halves[0].loc[common], halves[1].loc[common], np.random.default_rng(11), 1000)


def table6(beh, m1, tr):
    rows = []
    for _, r in beh.iterrows():
        rows.append({"block": "Behavioral indices", "index": r["index"], "level": r.level, "SB": r.SB,
                     "SB_at_154_obstacles": r.SB_at_154_obstacles if r.level == "Level 2" else np.nan})
    for blk, tab in (("Truncated DDM (hierarchical, responses >= 150 ms)", tr), ("Model 1", m1)):
        for _, r in tab.iterrows():
            lev = "Level 1" if "Level 1" in r.parameter else "Level 2" if "Level 2" in r.parameter else "both levels"
            rows.append({"block": blk, "index": r.parameter.replace(", Level 1", "").replace(", Level 2", ""),
                         "level": lev, "SB": r.SB, "SB_at_154_obstacles": np.nan})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    trials = load_trials()
    beh = behavioral(trials, np.random.default_rng(7))
    m1 = model1()
    tr = ddm_trunc(trials)
    for name, tab in (("behavioral", beh), ("model1", m1), ("ddm_trunc", tr)):
        print(f"\n=== {name}\n", tab.round(3).to_string(index=False))
        tab.to_csv(RESULTS / f"reliability_{name}.csv", index=False)
    t6 = table6(beh, m1, tr)
    t6.to_csv(RESULTS / "table6_reliability.csv", index=False)
