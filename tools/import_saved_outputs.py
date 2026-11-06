# =============================================================================
# Imports the model outputs produced during the original analysis (folder `output_ipotesi/` of the
# authors' working copy, file names in Italian) into results/fits/ with the names used by this repository,
# so that steps 04, 07 and 09-14 can be run without refitting the models.
#
#   python tools/import_saved_outputs.py <path to output_ipotesi> --raw <original export .xlsx> [--copy]
#
# Files are hard-linked when possible (no extra disk space), otherwise copied (--copy forces a copy).
# The csv files with one row per response (trials_*, sim_recovery) identify children by the original
# accessToken: with --raw they are rewritten with the codes of the de-identified data file (C001 ...,
# see tools/anonymize_data.py), which is required before depositing them. Without --raw they are copied
# unchanged and a warning is printed.
# =============================================================================
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from anonymize_data import child_codes

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results" / "fits"

MAP = {
    # behavioural models (step 03)
    "idata_minuti_omissioni_item.nc": "idata_behav_omissions.nc",
    "idata_minuti_errori_item.nc": "idata_behav_action_errors.nc",
    "idata_minuti_anticipate_item.nc": "idata_behav_early.nc",
    "idata_minuti_timing_item.nc": "idata_behav_timing.nc",
    # Model 1 (step 06)
    "sim_recovery_sigma.csv": "sim_recovery.csv",
    "sim_recovery_truth_sigma.npz": "sim_recovery_truth.npz",
    # Models 2-3 and truncated DDM (step 08)
    "ll_M1sg_m1.npy": "ll_model2.npy", "post_M1sg_m1.npz": "post_model2.npz",
    "ll_M5s_lungo.npy": "ll_model3.npy", "post_M5s_lungo.npz": "post_model3.npz",
}
for part in ("all", "half0", "half1"):
    MAP[f"ll_DDMtr_tr_{part}.npy"] = f"ll_ddm_trunc_{part}.npy"
    MAP[f"post_DDMtr_tr_{part}.npz"] = f"post_ddm_trunc_{part}.npz"
for old, new in (("_m1_sigma", "model1"), ("_m1_sigma_half0", "model1_half0"),
                 ("_m1_sigma_half1", "model1_half1"), ("_m1_sigma_recovery", "model1_recovery")):
    MAP[f"idata_misture{old}.nc"] = f"idata_{new}.nc"
    MAP[f"misture_trials{old}.csv"] = f"trials_{new}.csv"
    MAP[f"misture_combos{old}.csv"] = f"combos_{new}.csv"
    MAP[f"ll_M1{old}.npy"] = f"ll_{new}.npy"
    MAP[f"post_M1{old}.npz"] = f"post_{new}.npz"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python tools/import_saved_outputs.py <path to output_ipotesi> "
                         "--raw <original export .xlsx> [--copy]")
    src = Path(sys.argv[1])
    copy = "--copy" in sys.argv
    codes = None
    if "--raw" in sys.argv:
        codes = child_codes(pd.read_excel(sys.argv[sys.argv.index("--raw") + 1])["accessToken"])
    DEST.mkdir(parents=True, exist_ok=True)
    for old, new in MAP.items():
        s, d = src / old, DEST / new
        if not s.exists():
            print(f"missing  {old}")
            continue
        if d.exists():
            d.unlink()
        if s.suffix == ".csv" and "subj" in pd.read_csv(s, nrows=0).columns:
            if codes is None:
                shutil.copy2(s, d)
                print(f"WARNING  {old} -> results/fits/{new} still holds the original tokens (use --raw)")
                continue
            tab = pd.read_csv(s)
            tab["subj"] = tab["subj"].map(codes)
            if tab["subj"].isna().any():
                raise SystemExit(f"{old}: children not found in the --raw file")
            tab.to_csv(d, index=False)
            print(f"recoded  {old}  ->  results/fits/{new}")
            continue
        try:
            if copy:
                raise OSError
            os.link(s, d)
            how = "linked"
        except OSError:
            shutil.copy2(s, d)
            how = "copied"
        print(f"{how:7s}  {old}  ->  results/fits/{new}")
