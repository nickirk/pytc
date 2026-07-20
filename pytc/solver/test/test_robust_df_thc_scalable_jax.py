import numpy as np
import pytest

from pytc.solver.robust_df_thc_scalable_jax import (
    direct_df_sandwiches_panelled_jax,
    fit_panelled_lsthc_jax,
)
from pytc.solver.test.robust_df_thc_scalable import (
    direct_df_sandwiches_panelled,
    fit_panelled_lsthc,
)


def test_jax_fit_and_sandwich_match_panelled_oracle():
    rng = np.random.default_rng(4)
    p = rng.normal(size=(4, 3))
    b = rng.normal(size=(4, 4, 5))
    t2 = rng.normal(size=(2, 2, 4, 4))
    oracle_fit = fit_panelled_lsthc(p, b, rcond=1e-12, virtual_panel=2)
    jax_fit = fit_panelled_lsthc_jax(p, b, rcond=1e-12, virtual_panel=2)
    assert np.max(np.abs(oracle_fit.y - np.asarray(jax_fit.y))) < 1e-12
    oracle = direct_df_sandwiches_panelled(b, oracle_fit, t2, rank_panel=2, aux_panel=3)
    actual = direct_df_sandwiches_panelled_jax(b, jax_fit, t2, rank_panel=2, aux_panel=3)
    assert np.max(np.abs(oracle.robust - np.asarray(actual.robust))) < 1e-12


@pytest.mark.parametrize("rcond", (0.0, -1.0, 1.1, np.nan, np.inf))
def test_jax_fit_rejects_invalid_rcond(rcond):
    p = np.eye(2)
    b = np.ones((2, 2, 1))
    with pytest.raises(ValueError, match="rcond"):
        fit_panelled_lsthc_jax(p, b, rcond=rcond, virtual_panel=1)
