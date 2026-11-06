# =============================================================================
# Step 8. Fits of the alternative models (Method: "Alternative models", "Estimation", "Model comparison",
# "Parameter recovery and reliability"). GPU recommended.
#
#   python code/08_fit_alternative_models.py model2
#       Model 2, fast-guess mixture: 4 chains, 1,500 warmup + 1,000 draws, max tree depth 10
#   python code/08_fit_alternative_models.py model3
#       Model 3, two-stage mixture: 8 chains, 2,000 warmup + 3,000 draws, max tree depth 10, exact
#       reparameterisation (FIT_REPARAM=1) and starting values dispersed across chains (FIT_JITTER)
#       (article: desktop RTX 5090; the posterior has several modes, see 09_model_comparison.py)
#   python code/08_fit_alternative_models.py ddm_trunc all | half0 | half1
#       hierarchical truncated DDM on responses >= 150 ms (all obstacles, even or odd obstacles):
#       4 chains, 1,500 warmup + 1,000 draws, max tree depth 10
#   python code/08_fit_alternative_models.py test model2 | model3 | ddm_trunc
#       4 children, very short chains (checks the installation; not meant to converge)
#
# Sampler settings can be overridden with FIT_WARMUP, FIT_SAMPLES, FIT_CHAINS, FIT_DEPTH.
# Output: results/fits/ll_<tag>.npy and results/fits/post_<tag>.npz
# =============================================================================
import os
import sys

USAGE = "usage: python code/08_fit_alternative_models.py [model2 | model3 | ddm_trunc all|half0|half1 | test MODEL]"
args = sys.argv[1:]
if not args:
    raise SystemExit(USAGE)
target = args[1] if args[0] == "test" and len(args) > 1 else args[0]
if target == "model3":
    # must be set before models.py is imported
    os.environ.setdefault("FIT_REPARAM", "1")
    os.environ.setdefault("FIT_JITTER", "1.0")

import models as C  # noqa: E402
from config import CUT  # noqa: E402
from data_prep import load_trials  # noqa: E402

env = lambda k, default: int(os.environ.get(k, default))


def settings(warmup, samples, chains, depth=10):
    return dict(warmup=env("FIT_WARMUP", warmup), samples=env("FIT_SAMPLES", samples),
                chains=env("FIT_CHAINS", chains), max_tree_depth=env("FIT_DEPTH", depth))


if __name__ == "__main__":
    if args[0] == "test":
        m = target
        df = None
        if m == "ddm_trunc":
            tr = load_trials()
            df = tr[tr.rt_raw >= CUT]
        C.fit(m, tag=f"{m}_test", n_subj=4, warmup=50, samples=50, chains=2, df=df,
              max_tree_depth=6 if m == "model3" else 10)
    elif args[0] == "model2":
        C.fit("model2", tag="model2", **settings(1500, 1000, 4))
    elif args[0] == "model3":
        C.fit("model3", tag="model3", **settings(2000, 3000, 8))
    elif args[0] == "ddm_trunc":
        part = args[1] if len(args) > 1 else "all"
        tr = load_trials()
        tr = tr[tr.rt_raw >= CUT]
        if part != "all":
            tr = tr[tr.trial_n % 2 == int(part[-1])]
        C.fit("ddm_trunc", tag=f"ddm_trunc_{part}", df=tr, **settings(1500, 1000, 4))
    else:
        raise SystemExit(USAGE)
