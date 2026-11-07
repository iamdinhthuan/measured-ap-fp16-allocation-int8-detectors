"""rules_11_12d_add1.md A3: pooled output SQNR on boxes and score only (class-index column excluded from noise and
signal). fx_factorial.py unchanged; its module-level sq() is rebound to drop column 5 of the (1,300,6) output."""
import sys

import numpy as np

import rd_common  # noqa: F401  (must come first)
import fx_factorial

_sq = fx_factorial.sq


def sq_bs(a, b=None):
    assert a.shape[-1] == 6, a.shape
    return _sq(np.asarray(a)[..., :5], None if b is None else np.asarray(b)[..., :5])


fx_factorial.sq = sq_bs
sys.argv = [sys.argv[0], "--model", "rtdetr-l", "--mode", "noise", "--n-img", "2000", "--no-recon", "--tag", "_n2000bs"]
fx_factorial.main()
