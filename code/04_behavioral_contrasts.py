# =============================================================================
# Step 4. Contrasts of the behavioural models (Tables 3, 4A and 5; Results RQ2-RQ4) and Figure 5.
#
# Every comparison is computed draw by draw from the cell means:
#   RQ2  session 2 - session 1
#   RQ3  consecutive minutes within each level, and last minute - first minute
#   RQ4  Level 2 - Level 1 at the same minute (1-4) and averaged over minutes 1-4
# Scales: logit for binary outcomes, ms for the timing centre, log for the timing scale.
# Also reports the effects of the sequence covariates (obstacle type, change of type).
#
#   python code/04_behavioral_contrasts.py
# Output: results/behavioral_contrasts.csv, results/behavioral_cell_means.csv,
#         results/behavioral_sequence_effects.csv, figures/figure5_minutes.png
# =============================================================================
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES, FITS, RESULTS, summarise

# (outcome, prefix of the parameters): the timing model has a location part ("") and a scale part ("sigma_")
PARTS = [("omissions", ""), ("action_errors", ""), ("early", ""), ("timing", ""), ("timing", "sigma_")]


def label(outcome, pre):
    return f"{outcome}{' (scale, log)' if pre else ''}"


def cells(post, pre):
    da = post[pre + "cella"]
    dim = [d for d in da.dims if d not in ("chain", "draw")][0]
    return {str(c): da.sel({dim: c}).values.reshape(-1) for c in da[dim].values}


def contrasts(post, pre):
    c = cells(post, pre)
    rows = {("RQ2", "session 2 - 1"): post[pre + "sess"].values.reshape(-1)}
    for lev, n in [("B", 4), ("L", 5)]:
        name = "L1" if lev == "B" else "L2"
        for m in range(2, n + 1):
            rows[("RQ3", f"{name} min {m} - {m - 1}")] = c[f"{lev}{m}"] - c[f"{lev}{m - 1}"]
        rows[("RQ3", f"{name} last - min 1")] = c[f"{lev}{n}"] - c[f"{lev}1"]
    for m in range(1, 5):
        rows[("RQ4", f"L2 - L1 min {m}")] = c[f"L{m}"] - c[f"B{m}"]
    rows[("RQ4", "L2 - L1 mean of min 1-4")] = np.mean([c[f"L{m}"] - c[f"B{m}"] for m in range(1, 5)], 0)
    return rows, c


def sequence_effects(post, pre):
    g = lambda n: post[pre + n].values.reshape(-1)
    rows = {"jump - slide": g("jump_c"), "change - repeat": g("switch_c"),
            "type x change interaction": g("jump_c:switch_c"),
            "change - repeat, to a jump": g("switch_c") + 0.5 * g("jump_c:switch_c"),
            "change - repeat, to a slide": g("switch_c") - 0.5 * g("jump_c:switch_c")}
    return rows


def figure5(means):
    LIV1, LIV2 = "#2a78d6", "#eb6834"
    panels = [("omissions", "Omissions (%)", 100), ("action_errors", "Action errors (%)", 100),
              ("early", "Early responses (%)", 100), ("timing (scale, log)", "Timing scale (ms)", 1)]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#d8dde3", "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False})
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), sharex=True)
    for ax, (k, lab, sc), letter in zip(axes.flat, panels, "ABCD"):
        g = means[means.outcome == k].copy()
        g["lev"], g["min"] = g.cell.str[0], g.cell.str[1].astype(int)
        for lev, c, name, off in [("B", LIV1, "Level 1 (240 s, 16 m/s)", -0.06),
                                  ("L", LIV2, "Level 2 (300 s, 17 m/s)", 0.06)]:
            h = g[g.lev == lev].sort_values("min")
            ax.errorbar(h["min"] + off, h["median"] * sc,
                        yerr=[(h["median"] - h.lo) * sc, (h.hi - h["median"]) * sc],
                        color=c, lw=2, marker="o", ms=4, capsize=0, elinewidth=1.2, label=name)
        ax.set_ylabel(lab)
        ax.set_title(letter, loc="left", fontweight="bold")
        ax.set_xticks(range(1, 6))
    for ax in axes[1]:
        ax.set_xlabel("Minute of the session")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5_minutes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    tabs, means, seq = [], [], []
    for outcome, pre in PARTS:
        idata = az.from_netcdf(FITS / f"idata_behav_{outcome}.nc")
        post = idata["posterior"]
        names = [n for n in post.data_vars if n.startswith(pre) and "|" not in n
                 and (pre or not n.startswith("sigma_")) and n != "nu"]
        s = az.summary(idata, var_names=names)
        div = int(np.asarray(idata["sample_stats"]["diverging"]).sum())
        lab = label(outcome, pre)
        rows, c = contrasts(post, pre)
        tab = pd.DataFrame({k: summarise(v) for k, v in rows.items()}).T
        tab.index.names = ["block", "contrast"]
        tab.insert(0, "outcome", lab)
        print(f"\n=== {lab}  [R-hat max {pd.to_numeric(s['r_hat']).max():.3f}, "
              f"ESS min {pd.to_numeric(s['ess_bulk']).min():.0f}, divergences {div}]")
        print(tab.round(3).to_string())
        tabs.append(tab.reset_index())
        # cell means on the natural scale (average session, average obstacle type and transition)
        for k, v in c.items():
            if outcome == "timing":
                val = np.exp(v) if pre else v
            else:
                val = 1 / (1 + np.exp(-v))
            means.append({"outcome": lab, "cell": k, **summarise(val)})
        for k, v in sequence_effects(post, pre).items():
            seq.append({"outcome": lab, "effect": k, **summarise(v)})
    pd.concat(tabs).to_csv(RESULTS / "behavioral_contrasts.csv", index=False)
    means = pd.DataFrame(means)
    means.to_csv(RESULTS / "behavioral_cell_means.csv", index=False)
    seq = pd.DataFrame(seq)
    seq.to_csv(RESULTS / "behavioral_sequence_effects.csv", index=False)
    print("\n=== sequence covariates\n", seq.round(3).to_string(index=False))
    figure5(means)
    print("saved results/behavioral_*.csv and figures/figure5_minutes.png")
