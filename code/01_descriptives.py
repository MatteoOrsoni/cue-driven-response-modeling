# =============================================================================
# Step 1. Data description and accuracy of early responses (Method: Data; Results: "Early responses are
# informed"), Figures 1 and 2.
#
#   python code/01_descriptives.py
# Output: results/descriptives.json, results/accuracy_by_rt_bin.csv,
#         figures/figure1_timeline.png, figures/figure2_rt_accuracy.png
# =============================================================================
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CUT, FIGURES, RESULTS
from data_prep import load_trials

INK, INK2, GRID = "#141a21", "#4b5562", "#d8dde3"
JUMP, SLIDE = "#1baf7a", "#4a3aa7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def describe(df):
    r = df[df.omission == 0]
    early = r.rt_raw < CUT
    out = {
        "children": int(df.subj.nunique()),
        "children_level2": int(df[df.versione == "lunga"].subj.nunique()),
        "sessions": int(df.groupby(["subj", "sessione"]).ngroups),
        "sessions_not_first_attempt": int((df.groupby(["subj", "sessione"]).tentativo.first() > 1).sum()),
        "order_level1_first": int(df[df.schedule == 21].subj.nunique()),
        "order_level2_first": int(df[df.schedule == 22].subj.nunique()),
        "obstacles": int(len(df)),
        "omissions": int(df.omission.sum()), "omission_rate": float(df.omission.mean()),
        "responses": int(len(r)), "accuracy": float(r.correct.mean()),
        "share_before_cue": float((r.rt_raw < 0).mean()),
        "share_early_lt150": float(early.mean()),
        "earliest_response_ms": float(1000 * r.rt_raw.min()),
        "slides_share": float((df.obstacle == "SLIDE").mean()),
        "repeat_prob": float(df.sort_values(["subj", "sessione", "trial_n"]).groupby(["subj", "sessione"])
                             .obstacle.apply(lambda s: (s == s.shift()).iloc[1:].mean()).mean()),
        # accuracy by response-time range (Results: "Early responses are informed")
        "accuracy_before_cue": float(r.correct[r.rt_raw < 0].mean()),
        "accuracy_0_100": float(r.correct[(r.rt_raw >= 0) & (r.rt_raw < 0.100)].mean()),
        "accuracy_100_150": float(r.correct[(r.rt_raw >= 0.100) & (r.rt_raw < CUT)].mean()),
        "accuracy_ge150": float(r.correct[r.rt_raw >= CUT].mean()),
        # share of responses that a 150-ms cut-off keeps, by level and by level x minute
        "kept_by_cutoff_level1": float((~early[r.versione == "breve"]).mean()),
        "kept_by_cutoff_level2": float((~early[r.versione == "lunga"]).mean()),
        "kept_by_cutoff_cells": {k: float(v) for k, v in (~early).groupby(r.cella).mean().items()},
    }
    return out


def accuracy_bins(df):
    """Conditional accuracy functions: accuracy in 25-ms bins, by obstacle type."""
    r = df[df.omission == 0]
    edges = np.arange(-0.300, 0.5251, 0.025)
    r = r.assign(bin=pd.cut(r.rt_raw, edges, labels=False))
    tab = r.groupby(["bin", "obstacle"]).agg(n=("correct", "size"), accuracy=("correct", "mean")).reset_index()
    tab["bin_centre_ms"] = ((edges[:-1] + edges[1:]) / 2 * 1000)[tab.bin.astype(int)]
    return tab[["obstacle", "bin_centre_ms", "n", "accuracy"]]


def figure1():
    """Timeline of a single obstacle, relative to the auditory cue (ms)."""
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    ax.set_xlim(-620, 620)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    grad = np.linspace(0, 1, 256)[None, :]          # visual phase: onset unknown, hence the gradient
    ax.imshow(grad, extent=(-600, 0, 2.6, 3.3), aspect="auto", cmap="Greys", vmin=-0.2, vmax=4, alpha=0.9)
    ax.text(-300, 3.5, "Obstacle visible: jump or slide can already be identified", ha="center", fontsize=8.5, color=INK)
    ax.add_patch(plt.Rectangle((-300, 1.5), 300, 0.7, color=SLIDE, alpha=0.18, lw=0))
    ax.add_patch(plt.Rectangle((0, 1.5), 150, 0.7, color=SLIDE, alpha=0.35, lw=0))
    ax.add_patch(plt.Rectangle((150, 1.5), 350, 0.7, color=JUMP, alpha=0.30, lw=0))
    ax.text(-150, 1.85, "before the cue", ha="center", va="center", fontsize=8, color=INK)
    ax.text(75, 1.85, "< 150 ms", ha="center", va="center", fontsize=8, color=INK)
    ax.text(325, 1.85, "cue can contribute", ha="center", va="center", fontsize=8, color=INK)
    ax.text(-315, 1.85, "recorded\nresponses", ha="right", va="center", fontsize=8, color=INK2)
    ax.annotate("", xy=(610, 0.7), xytext=(-610, 0.7), arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    for x, lab in [(-281, "earliest\nresponse"), (0, "auditory cue\n(jump / slide)"),
                   (150, "150 ms\n "), (500, "end of window\n(omission)")]:
        ax.plot([x, x], [0.6, 3.3 if x == 0 else 2.2], color=INK, lw=1.2 if x in (0, 500) else 0.8,
                ls="-" if x in (0, 500) else "--")
        ax.text(x, 0.05, lab, ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.text(615, 0.85, "ms", fontsize=7.5, color=INK2, ha="right")
    fig.savefig(FIGURES / "figure1_timeline.png")
    plt.close(fig)


def figure2(df):
    """(A) Response-time distribution in 25-ms bins; (B) accuracy by response time and obstacle type."""
    r = df[df.omission == 0]
    edges = np.arange(-0.300, 0.5251, 0.025)
    mids = (edges[:-1] + edges[1:]) / 2 * 1000
    r = r.assign(bin=pd.cut(r.rt_raw, edges, labels=False))
    n = r.groupby("bin").size().reindex(range(len(mids)), fill_value=0)
    acc = r.groupby(["bin", "obstacle"]).correct.mean().unstack()
    nn = r.groupby(["bin", "obstacle"]).size().unstack()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.12})
    cols = np.where(mids < 150, SLIDE, "#9aa6b2")
    a1.bar(mids, n.values, width=23, color=cols, alpha=0.85, linewidth=0)
    a1.set_ylabel("Responses")
    a1.text(-295, n.max() * 0.80, "purple: < 150 ms,\ntoo early to use the cue", ha="left", color=INK2, fontsize=8)
    for ax in (a1, a2):
        ax.axvline(0, color=INK, lw=1)
        ax.axvline(150, color=INK, lw=1, ls="--")
    a1.text(4, n.max() * 1.02, "cue onset", fontsize=8, color=INK)
    a1.text(154, n.max() * 1.02, "150 ms", fontsize=8, color=INK)
    ok = lambda col: (nn[col] >= 50).reindex(range(len(mids)), fill_value=False).values
    for col, c, lab in [("JUMP", JUMP, "Jump obstacle"), ("SLIDE", SLIDE, "Slide obstacle")]:
        y = acc[col].reindex(range(len(mids))).values
        m = ok(col)
        a2.plot(mids[m], y[m] * 100, color=c, lw=2, marker="o", ms=3, label=lab)
    a2.axhline(50, color=INK2, lw=0.8, ls=":")
    a2.text(-295, 51.5, "chance", fontsize=8, color=INK2)
    a2.set_ylim(40, 100)
    a2.set_ylabel("Accuracy (%)")
    a2.set_xlabel("Response time relative to the auditory cue (ms)")
    a2.set_xlim(-310, 510)
    a2.legend(loc="lower right")
    a1.set_title("A", loc="left", fontweight="bold")
    a2.set_title("B", loc="left", fontweight="bold")
    fig.savefig(FIGURES / "figure2_rt_accuracy.png")
    plt.close(fig)


if __name__ == "__main__":
    df = load_trials()
    res = describe(df)
    print(json.dumps(res, indent=1))
    json.dump(res, open(RESULTS / "descriptives.json", "w"), indent=1)
    accuracy_bins(df).to_csv(RESULTS / "accuracy_by_rt_bin.csv", index=False)
    figure1()
    figure2(df)
    print("saved results/descriptives.json, results/accuracy_by_rt_bin.csv, figures 1-2")
