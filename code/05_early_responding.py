# =============================================================================
# Step 5. Individual differences in early responding (Method and Results: "Individual differences in
# early responding"). Purely descriptive.
#
#   - proportion of responses < 150 ms per session (omissions excluded): distribution, range
#   - Gaussian mixtures with 1-3 components on the logit of the proportions (clipped to .01-.99), BIC
#   - split-half reliability (odd vs even obstacles, Spearman-Brown) per level
#   - stability across the two sessions (played on different levels): Pearson r
#   - accuracy of early vs later responses within session: Wilcoxon signed-rank test
#   - correlation across children of the proportion with accuracy (all, early and later responses) and
#     omissions, per level (Spearman)
#
#   python code/05_early_responding.py
# Output: results/early_responding.json, results/early_responding_by_session.csv
# =============================================================================
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

from config import CUT, RESULTS
from data_prep import load_trials


def sb(r):
    return 2 * r / (1 + r)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    out = {}
    df = load_trials()
    r = df[df.omission == 0].copy()
    r["early"] = (r.rt_raw < CUT).astype(int)

    acc = r.groupby(["early", "obstacle"]).correct.mean().unstack()
    print("=== Accuracy by response mode (0 = later, 1 = early) and obstacle type ===\n", acc.round(3))
    out["accuracy_by_mode_and_obstacle"] = {f"{'early' if e else 'later'}_{o}": float(acc.loc[e, o])
                                           for e in acc.index for o in acc.columns}

    # per session (child x level)
    run = r.groupby(["subj", "versione", "sessione"]).agg(p_early=("early", "mean"), acc=("correct", "mean")).reset_index()
    om = df.groupby(["subj", "versione"]).omission.mean().rename("om").reset_index()
    run = run.merge(om)
    acc_mode = r.groupby(["subj", "versione", "early"]).correct.mean().unstack()
    acc_mode.columns = ["acc_later", "acc_early"]
    run = run.merge(acc_mode.reset_index())
    print("\n=== Proportion of early responses per session ===\n", run.p_early.describe().round(3))
    out["p_early_per_session"] = {"mean": float(run.p_early.mean()), "min": float(run.p_early.min()),
                                  "max": float(run.p_early.max()), "n_sessions": int(len(run))}

    # are there distinct types of player? Gaussian mixtures on the logit
    p = run.p_early.clip(.01, .99)
    x = np.log(p / (1 - p)).to_numpy()[:, None]
    out["gmm_bic"] = {}
    for k in (1, 2, 3):
        bic = GaussianMixture(k, random_state=0).fit(x).bic(x)
        out["gmm_bic"][k] = float(bic)
        print(f"  GMM k = {k}: BIC = {bic:.1f}")

    # reliability and stability
    out["split_half_sb"] = {}
    for ver, g in r.groupby("versione"):
        o = g[g.trial_n % 2 == 1].groupby("subj").early.mean()
        e_ = g[g.trial_n % 2 == 0].groupby("subj").early.mean()
        rel = sb(np.corrcoef(o, e_.loc[o.index])[0, 1])
        out["split_half_sb"]["Level 1" if ver == "breve" else "Level 2"] = float(rel)
        print(f"  split-half {ver}: SB = {rel:.3f}")
    w = run.pivot(index="subj", columns="sessione", values="p_early").dropna()
    out["session1_vs_session2"] = {"r": float(np.corrcoef(w[1], w[2])[0, 1]), "n": int(len(w))}
    print(f"  session 1 vs 2 (different levels): r = {out['session1_vs_session2']['r']:.2f}, n = {len(w)}")

    # does early responding cost accuracy?
    d = run.dropna(subset=["acc_early", "acc_later"])
    t = stats.wilcoxon(d.acc_early, d.acc_later)
    out["within_session_accuracy"] = {"early": float(d.acc_early.mean()), "later": float(d.acc_later.mean()),
                                      "wilcoxon_p": float(t.pvalue)}
    print(f"\nAccuracy early vs later within session: {d.acc_early.mean():.3f} vs {d.acc_later.mean():.3f} "
          f"(Wilcoxon p = {t.pvalue:.3f})")
    out["spearman"] = {}
    for ver, g in run.groupby("versione"):
        for m in ["acc", "om", "acc_early", "acc_later"]:
            ok = g[[m, "p_early"]].dropna()
            rho = stats.spearmanr(ok.p_early, ok[m])
            out["spearman"][f"{'Level 1' if ver == 'breve' else 'Level 2'}: p_early vs {m}"] = {
                "rho": float(rho.statistic), "p": float(rho.pvalue)}
            print(f"  {ver:6s} p_early vs {m:9s} rho = {rho.statistic:+.2f} (p = {rho.pvalue:.3f})")
    run.to_csv(RESULTS / "early_responding_by_session.csv", index=False)
    json.dump(out, open(RESULTS / "early_responding.json", "w"), indent=1)
