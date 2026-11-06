# =============================================================================
# Step 3. Bayesian hierarchical regression models of the four behavioural outcomes (Method: "Behavioral
# models"; priors in Table A1).
#
#   omissions      P(no response within 500 ms)               Bernoulli, logit link, all obstacles
#   action_errors  P(wrong action | response)                  Bernoulli, logit link, responses
#   early          P(response < 150 ms after the cue)          Bernoulli, logit link, responses
#   timing         time of correct responses (ms from cue)     Student-t location-scale (centre and scale)
#
# Predictors (same in all models): one mean per level x minute cell (`cella`, no intercept), session
# (-0.5 / +0.5), obstacle type, change of type with respect to the previous obstacle and their
# interaction (-0.5 / +0.5), child-specific intercept and level slope.
# In the timing model the same predictors enter the scale (log link).
#
#   python code/03_behavioral_models.py all            (or one outcome: omissions | action_errors | early | timing)
# Output: results/fits/idata_behav_<outcome>.nc
# =============================================================================
import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import arviz as az
import bambi as bmb
import numpy as np

from config import CUT, FITS
from data_prep import load_trials

OUTCOMES = ("omissions", "action_errors", "early", "timing")


def model_data(df, outcome):
    """Data for one outcome, with the sequence covariates."""
    d = df.sort_values(["subj", "sessione", "trial_n"]).copy()
    prev = d.groupby(["subj", "sessione"])["obstacle"].shift()
    d["jump_c"] = np.where(d["obstacle"] == "JUMP", 0.5, -0.5)
    d["switch_c"] = np.where(prev.notna() & (d["obstacle"] != prev), 0.5, -0.5)
    if outcome != "omissions":
        d = d[d["omission"] == 0].copy()
    if outcome == "omissions":
        d["y"] = d["omission"]
    elif outcome == "action_errors":
        d["y"] = 1 - d["correct"]
    elif outcome == "early":
        d["y"] = (d["rt_raw"] < CUT).astype(int)
    elif outcome == "timing":
        d = d[d["correct"] == 1].copy()
        d["y"] = d["rt_raw"] * 1000
    return d


def fit(df, outcome):
    d = model_data(df, outcome)
    cov = " + jump_c * switch_c"
    rand = "(1 + ver | subj_idx)"
    if outcome == "timing":
        model = bmb.Model(bmb.Formula(f"y ~ 0 + cella + sess{cov} + {rand}",
                                      f"sigma ~ 0 + cella + sess{cov} + {rand}"),
                          data=d, family="t",
                          priors={"cella": bmb.Prior("Normal", mu=150, sigma=100),
                                  "sess": bmb.Prior("Normal", mu=0, sigma=50)})
    else:
        model = bmb.Model(f"y ~ 0 + cella + sess{cov} + {rand}", data=d, family="bernoulli",
                          priors={"cella": bmb.Prior("Normal", mu=0, sigma=2),
                                  "sess": bmb.Prior("Normal", mu=0, sigma=1)})
    print(model)
    idata = model.fit(inference_method="numpyro", chains=4, draws=1000, tune=1000,
                      target_accept=0.9, random_seed=2026)
    idata.to_netcdf(FITS / f"idata_behav_{outcome}.nc")
    print(az.summary(idata, var_names=["~subj_idx"], filter_vars="like"))
    return idata


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = OUTCOMES if which == "all" else (which,)
    df = load_trials()
    for o in todo:
        assert o in OUTCOMES, f"unknown outcome {o}; choose among {OUTCOMES}"
        fit(df, o)
