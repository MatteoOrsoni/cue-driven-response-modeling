# =============================================================================
# Reading and coding of the Eye-Riders A data (one row per obstacle).
#
# Input: the Excel export of the game (one row per completed level, see data/README.md). The column
# `collisionData` holds a JSON dictionary with one entry per obstacle: obstacle type (JUMP / SLIDE),
# outcome (CORRECT / WRONG / NO_REACTION) and reactionTime, in seconds relative to the auditory cue
# (negative = response given before the cue).
#
# Output columns used by the analyses (raw Italian codes are kept so that saved model outputs remain
# compatible):
#   subj        child identifier (child_id, C001-C128)  subj_idx  integer code of subj
#   tentativo   attempt number for that level           schedule  21 / 22 = order group
#   sessione    1 / 2 = first / second session          sess      -0.5 / +0.5 coding of sessione
#   versione    "breve" = Level 1, "lunga" = Level 2    ver       -0.5 (Level 1) / +0.5 (Level 2)
#   trial_n     position of the obstacle in the level (0-based, omissions included)
#   time_s      elapsed time at the obstacle = trial_n * 25 m / speed
#   minuto      minute of the level (1-4 in Level 1, 1-5 in Level 2)
#   cella       level x minute cell: B1-B4 (Level 1), L1-L5 (Level 2)
#   obstacle    JUMP / SLIDE
#   omission    1 = no response within 500 ms of the cue
#   correct     1 = response matched the obstacle
#   rt_raw      response time relative to the cue (s); omissions carry the placeholder 0.75 logged by
#               the game and are always excluded before rt_raw is used
# =============================================================================
import json

import numpy as np
import pandas as pd

from config import DATA_FILE


def load_trials(filepath=DATA_FILE):
    raw = pd.read_excel(filepath)
    # child_id in the de-identified file distributed with the repository; accessToken in the original export
    id_col = "child_id" if "child_id" in raw else "accessToken"
    rows = []
    for _, row in raw.iterrows():
        if pd.isna(row["collisionData"]):
            continue
        sch, lp = int(row["scheduleId"]), int(row["levelProg"])
        # schedule 21: Level 1 first; schedule 22: Level 2 first
        if sch == 21:
            versione = "breve" if lp == 1 else "lunga"
        elif sch == 22:
            versione = "lunga" if lp == 1 else "breve"
        else:
            continue
        trials = json.loads(row["collisionData"])
        n_trials = len(trials)
        for key, tr in trials.items():
            rows.append({
                "subj": row[id_col],
                "tentativo": int(row["trial"]),
                "schedule": sch,
                "sessione": lp,
                "versione": versione,
                "trial_n": int(key),
                "n_trials_run": n_trials,
                "reaction": tr["reaction"],
                "rt_raw": float(tr["reactionTime"]),
                "obstacle": (tr.get("obstacleData") or {}).get("type"),
            })
    df = pd.DataFrame(rows).sort_values(["subj", "sessione", "trial_n"], kind="mergesort")

    # absolute time in the level (obstacles every 25 m; Level 1 = 16 m/s, Level 2 = 17 m/s)
    speed = np.where(df["versione"] == "lunga", 17.0, 16.0)
    df["time_s"] = df["trial_n"] * 25.0 / speed
    df["minuto"] = np.minimum(df["time_s"] // 60, np.where(df["versione"] == "lunga", 4, 3)).astype(int) + 1
    df["cella"] = df["versione"].str[0].str.upper() + df["minuto"].astype(str)   # B1..B4, L1..L5

    # effect coding (+-0.5)
    df["ver"] = np.where(df["versione"] == "lunga", 0.5, -0.5)
    df["sess"] = np.where(df["sessione"] == 2, 0.5, -0.5)

    df["omission"] = (df["reaction"] == "NO_REACTION").astype(int)
    df["correct"] = (df["reaction"] == "CORRECT").astype(int)
    df["subj_idx"] = df["subj"].astype("category").cat.codes
    return df.reset_index(drop=True)


def responses(df):
    """Obstacles with a response (omissions removed)."""
    return df[df.omission == 0].copy()


if __name__ == "__main__":
    df = load_trials()
    print(df.groupby(["versione", "sessione"]).agg(
        n=("trial_n", "size"), children=("subj", "nunique"),
        acc=("correct", "mean"), om=("omission", "mean")).round(3))
