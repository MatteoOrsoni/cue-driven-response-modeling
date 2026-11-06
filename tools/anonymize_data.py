# =============================================================================
# Creates the de-identified data file distributed with the repository (data/ER_A.xlsx) from the original
# export of the game. Run by the authors once; replicators do not need it.
#
#   python tools/anonymize_data.py <original export .xlsx> [output, default data/ER_A.xlsx]
#
# - accessToken (platform identifier of the child) is replaced by child_id = C001 ... C128. Codes follow
#   the alphabetical order of the original tokens, so the integer child index used by the models
#   (subj_idx, s) is the same as in the original analysis and the saved model outputs stay valid.
# - sessionId (platform session identifier) and createdAt (date and time of play) are removed.
# =============================================================================
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DROP = ["sessionId", "createdAt"]


def child_codes(tokens):
    """Original token -> C001, C002, ... in alphabetical order of the tokens."""
    uniq = sorted(pd.Series(tokens).dropna().unique())
    width = max(3, len(str(len(uniq))))
    return {t: f"C{i + 1:0{width}d}" for i, t in enumerate(uniq)}


def anonymize(raw):
    codes = child_codes(raw["accessToken"])
    out = raw.drop(columns=[c for c in DROP if c in raw])
    out.insert(0, "child_id", out.pop("accessToken").map(codes))
    return out.sort_values(["child_id", "levelProg"], kind="mergesort").reset_index(drop=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python tools/anonymize_data.py <original export .xlsx> [output .xlsx]")
    raw = pd.read_excel(sys.argv[1])
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "ER_A.xlsx"
    out = anonymize(raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(dest, index=False)
    print(f"{len(out)} rows, {out.child_id.nunique()} children -> {dest}")
