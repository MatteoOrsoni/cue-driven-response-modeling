# =============================================================================
# Step 6. Fit of Model 1, the timed-diffusion mixture (Method: "Mixture model (Model 1)", "Estimation").
# NUTS, 4 chains, 1,500 warmup + 1,000 draws, target acceptance .90, max tree depth 10. GPU recommended
# (about 15 minutes per fit on a laptop RTX 4060; see README).
#
#   python code/06_fit_model1.py main        all responses                      -> results/fits/*_model1.*
#   python code/06_fit_model1.py half0       even obstacles (split-half)        -> *_model1_half0.*
#   python code/06_fit_model1.py half1       odd obstacles (split-half)         -> *_model1_half1.*
#   python code/06_fit_model1.py recovery    data simulated by 12_parameter_recovery.py simulate
#                                                                               -> *_model1_recovery.*
#   python code/06_fit_model1.py test        4 children, very short chains (checks the installation)
# =============================================================================
import sys

import pandas as pd

import mixture as M
from config import RESULTS
from data_prep import load_trials

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "main"
    if what == "main":
        M.run("model1")
    elif what in ("half0", "half1"):
        k = int(what[-1])
        trials = load_trials()
        M.run(f"model1_{what}", df=trials[trials.trial_n % 2 == k])
    elif what == "recovery":
        M.run("model1_recovery", df=pd.read_csv(RESULTS / "fits" / "sim_recovery.csv"))
    elif what == "test":
        M.run("model1_test", n_subj=4, warmup=50, samples=50, chains=2)
    else:
        raise SystemExit("usage: python code/06_fit_model1.py [main | half0 | half1 | recovery | test]")
