# =============================================================================
# Paths and constants shared by all scripts.
#
# Every script can be run from any working directory: paths are resolved from the
# location of this file. The raw data file can be moved with the environment variable
# EYERIDERS_DATA (path to the Excel export of the game).
# =============================================================================
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = Path(os.environ.get("EYERIDERS_DATA", ROOT / "data" / "ER_A.xlsx"))
INITS = ROOT / "inits"
RESULTS = ROOT / "results"          # small outputs: csv / json tables (tracked in git)
FITS = RESULTS / "fits"             # large outputs: posterior draws, log-likelihoods (not tracked)
FIGURES = ROOT / "figures"
for _p in (RESULTS, FITS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# Response window, relative to the auditory cue (s). The earliest observed response is -0.281 s;
# a response still missing 0.5 s after the cue is an omission.
RT_LO, RT_HI = -0.3, 0.5
CUT = 0.150                         # "early" responses: < 150 ms after the cue

# Level x minute cells, in the order used by all models.
# B = Level 1 (240 s, 16 m/s, 154 obstacles; "breve" in the raw coding),
# L = Level 2 (300 s, 17 m/s, 204 obstacles; "lunga" in the raw coding); the digit is the minute.
CELLS = ["B1", "B2", "B3", "B4", "L1", "L2", "L3", "L4", "L5"]
N_OBST = {"breve": 154, "lunga": 204}


def summarise(x, prob=0.89):
    """Posterior median, equal-tailed credible interval and P(x > 0)."""
    import numpy as np
    x = np.asarray(x).reshape(-1)
    lo, hi = np.quantile(x, [(1 - prob) / 2, 1 - (1 - prob) / 2])
    return {"median": float(np.median(x)), "lo": float(lo), "hi": float(hi), "p_pos": float((x > 0).mean())}
