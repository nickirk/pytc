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


def test_sandwich_keeps_b_host_resident(monkeypatch):
    # Regression cover for the 1200 cycle-1 OOM (JID 20633292): the full
    # 3-index B block must never be cast onto the device wholesale -- the
    # panel loops consume it one aux slice at a time from the host.
    import pytc.solver.robust_df_thc_scalable_jax as mod

    uploaded = []
    real = mod._as_fp64_jax

    def recording(name, value, ndim):
        uploaded.append(name)
        return real(name, value, ndim)

    monkeypatch.setattr(mod, "_as_fp64_jax", recording)
    rng = np.random.default_rng(7)
    p = rng.normal(size=(4, 3))
    b = rng.normal(size=(4, 4, 5))
    t2 = rng.normal(size=(2, 2, 4, 4))
    jax_fit = fit_panelled_lsthc_jax(p, b, rcond=1e-12, virtual_panel=2)
    uploaded.clear()
    direct_df_sandwiches_panelled_jax(b, jax_fit, t2, rank_panel=2, aux_panel=3)
    assert "b" not in uploaded
    assert {"t2", "p_virtual", "y"} <= set(uploaded)


def test_exact_panel_cap_preserves_result(monkeypatch):
    # PYTC_EXACT_PANEL_CAP_GB clamps the (nocc^2, nvir, nvir, q) scratch
    # inside the exact sandwich (157 GiB at aux_panel=32 at the 1200 deck);
    # forcing q_step=1 must still reproduce the wider-panel result.
    rng = np.random.default_rng(11)
    p = rng.normal(size=(4, 3))
    b = rng.normal(size=(4, 4, 5))
    t2 = rng.normal(size=(2, 2, 4, 4))
    jax_fit = fit_panelled_lsthc_jax(p, b, rcond=1e-12, virtual_panel=2)
    wide = direct_df_sandwiches_panelled_jax(b, jax_fit, t2, rank_panel=2, aux_panel=3)
    monkeypatch.setenv("PYTC_EXACT_PANEL_CAP_GB", str(512 / 1024 ** 3))  # q_step=1
    clamped = direct_df_sandwiches_panelled_jax(b, jax_fit, t2, rank_panel=2, aux_panel=3)
    assert np.max(np.abs(np.asarray(wide.robust) - np.asarray(clamped.robust))) < 1e-12
