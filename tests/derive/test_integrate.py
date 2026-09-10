"""Tests for lanfactory.derive.integrate: batched ONNX predictor and choice mass.

Fast and deterministic: a tiny untrained TorchMLP for the ONNX parity check,
and a closed-form gamma mixture standing in for a LAN for the numerics.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import onnx
import pytest
import torch
from scipy.integrate import trapezoid
from scipy.stats import gamma

from lanfactory.derive import (
    ChoiceMass,
    IntegrationGrid,
    choice_mass,
    load_onnx_predictor,
)
from lanfactory.onnx import assert_single_trial_contract
from tests._onnx_utils import export_tiny_torch_lan
from tests.onnx.test_contract import make_onnx

INPUT_WIDTH = 6


# --------------------------------------------------------------------------
# 1. Batched ONNX predictor vs TorchMLP
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch_lan_and_onnx(tmp_path_factory):
    """A seeded TorchMLP and its ONNX export via the real ``transform-onnx`` path."""
    return export_tiny_torch_lan(tmp_path_factory.mktemp("derive"), INPUT_WIDTH)


def _batch_dim_values(path) -> list[int]:
    model = onnx.load(str(path))
    return [
        vi.type.tensor_type.shape.dim[0].dim_value
        for vi in (*model.graph.input, *model.graph.output)
    ]


def test_batched_predictor_matches_torch_and_leaves_file_untouched(
    torch_lan_and_onnx,
):
    net, onnx_file = torch_lan_and_onnx
    assert _batch_dim_values(onnx_file) == [1, 1], "exporter should trace (1, D)"

    predictor = load_onnx_predictor(onnx_file)
    assert predictor.input_width == INPUT_WIDTH
    assert isinstance(predictor.input_name, str)

    rng = np.random.default_rng(0)
    rows = rng.standard_normal((1000, INPUT_WIDTH)).astype(np.float32)
    with torch.no_grad():
        expected = net(torch.from_numpy(rows)).numpy()[:, 0]
    got = predictor(rows)
    assert got.shape == (1000,)
    np.testing.assert_allclose(got, expected, atol=1e-5)

    # Default-dtype (float64) rows are cast, not rejected by onnxruntime.
    rows64 = rng.standard_normal((64, INPUT_WIDTH))
    np.testing.assert_array_equal(
        predictor(rows64), predictor(rows64.astype(np.float32))
    )

    # The batch axis was made symbolic in memory only.
    assert _batch_dim_values(onnx_file) == [1, 1]
    result = assert_single_trial_contract(onnx_file, expected_input_width=INPUT_WIDTH)
    assert result["input_shape"] == [1, INPUT_WIDTH]


def test_predictor_runs_the_batch_in_one_session_call(torch_lan_and_onnx):
    _, onnx_file = torch_lan_and_onnx
    predictor = load_onnx_predictor(onnx_file)
    rows = np.random.default_rng(1).standard_normal((100_000, INPUT_WIDTH))
    rows = rows.astype(np.float32)
    with patch.object(predictor.session, "run", wraps=predictor.session.run) as run:
        out = predictor(rows)
    assert out.shape == (100_000,)
    assert np.isfinite(out).all()
    # The whole batch goes through the widened graph at once: no per-row loop.
    assert run.call_count == 1


def test_predictor_rejects_wrong_row_width(tmp_path):
    predictor = load_onnx_predictor(make_onnx(tmp_path / "m.onnx", (1, INPUT_WIDTH)))
    with pytest.raises(ValueError, match=rf"\(N, {INPUT_WIDTH}\)"):
        predictor(np.zeros((3, INPUT_WIDTH + 1), dtype=np.float32))


def test_rank_one_artifact_is_rejected_clearly(tmp_path):
    # sbi / bayesflow exports trace (D,): no batch axis to widen.
    path = make_onnx(tmp_path / "rank1.onnx", (INPUT_WIDTH,))
    with pytest.raises(ValueError, match=r"rank 1"):
        load_onnx_predictor(path)


# --------------------------------------------------------------------------
# 2. Closed form: two-choice gamma mixture
# --------------------------------------------------------------------------

CHOICES = np.array([-1, 1])
WEIGHTS = {-1: 0.3, 1: 0.7}
# shape >= 2 keeps the density C^1 at zero so the trapezoid rule converges
# at its nominal O(h^2); the (4, 2) row puts ~1% of its mass past max_t.
THETA = np.array([[2.0, 0.5], [3.0, 1.0], [4.0, 2.0]])


class GammaMixturePredictor:
    """Rows ``[shape, scale, rt, choice]`` -> log( w[choice] * Gamma(rt) )."""

    input_width = 4

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=np.float64)
        shape, scale, rt, choice = batch.T
        log_w = np.where(choice == 1, np.log(WEIGHTS[1]), np.log(WEIGHTS[-1]))
        return log_w + gamma.logpdf(rt, a=shape, scale=scale)


def expected_mass(choice: int, grid: IntegrationGrid) -> np.ndarray:
    """``w[choice] * (G(max_t) - G(t_min))`` per theta row."""
    shape, scale = THETA.T
    g = gamma.cdf(grid.max_t, a=shape, scale=scale) - gamma.cdf(
        grid.t_min, a=shape, scale=scale
    )
    return WEIGHTS[choice] * g


@pytest.fixture(scope="module")
def gamma_mass() -> ChoiceMass:
    return choice_mass(GammaMixturePredictor(), THETA, CHOICES)


def test_masses_match_closed_form(gamma_mass):
    for choice in CHOICES:
        err = np.abs(gamma_mass.mass(choice) - expected_mass(choice, IntegrationGrid()))
        assert err.max() < 1e-3, f"choice {choice}: max error {err.max():.2e}"


def test_mass_error_shrinks_with_grid_resolution():
    predictor = GammaMixturePredictor()

    def max_err(n_points: int) -> float:
        grid = IntegrationGrid(n_points=n_points)
        cm = choice_mass(predictor, THETA, CHOICES, grid=grid)
        return max(np.abs(cm.mass(c) - expected_mass(c, grid)).max() for c in CHOICES)

    assert max_err(250) > max_err(1000)

    coarse = choice_mass(predictor, THETA, CHOICES, grid=IntegrationGrid(1000))
    fine = choice_mass(predictor, THETA, CHOICES, grid=IntegrationGrid(4000))
    for c in CHOICES:
        assert np.abs(fine.mass(c) - coarse.mass(c)).max() < 2e-4


def test_total_is_sum_over_choices(gamma_mass):
    summed = sum(gamma_mass.mass(c) for c in CHOICES)
    np.testing.assert_allclose(gamma_mass.total, summed, rtol=0, atol=1e-12)
    # The (4, 2) row deliberately leaks mass past max_t: no renormalisation.
    assert gamma_mass.total[2] < 0.995
    assert np.all(gamma_mass.total < 1.0)


def test_cdf_shape_monotone_and_matches_trapezoid(gamma_mass):
    assert gamma_mass.cdf.shape == (len(THETA), len(CHOICES), 1000)
    assert np.all(gamma_mass.cdf[..., 0] == 0.0)
    assert np.all(np.diff(gamma_mass.cdf, axis=-1) >= 0.0)

    # Rows reach the predictor as float32 (a LAN's input dtype), so rt carries
    # float32 rounding relative to the float64 grid: agree to ~1e-7, not 1e-15.
    t = gamma_mass.t
    for i, (shape, scale) in enumerate(THETA):
        for j, c in enumerate(CHOICES):
            density = WEIGHTS[c] * gamma.pdf(t, a=shape, scale=scale)
            np.testing.assert_allclose(
                gamma_mass.cdf[i, j, -1], trapezoid(density, t), rtol=1e-6
            )


class SpyPredictor(GammaMixturePredictor):
    """Records every batch it is handed."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        self.calls.append(np.array(batch, copy=True))
        return super().__call__(batch)


def test_chunking_feeds_a_full_chunk_then_the_partial_remainder():
    spy = SpyPredictor()
    cm = choice_mass(spy, THETA, CHOICES, chunk_size=2)  # 3 thetas -> [2, 1]
    per_theta = len(CHOICES) * IntegrationGrid().n_points
    assert [c.shape[0] for c in spy.calls] == [2 * per_theta, per_theta]

    # Every theta row reached the predictor exactly as given.
    fed = np.concatenate(spy.calls)
    np.testing.assert_array_equal(
        np.unique(fed[:, :2], axis=0), np.unique(THETA.astype(np.float32), axis=0)
    )
    # And each slot of the chunked result equals an independent single-theta run.
    for i in range(len(THETA)):
        single = choice_mass(GammaMixturePredictor(), THETA[i], CHOICES)
        np.testing.assert_array_equal(cm.cdf[i], single.cdf[0])


def test_single_theta_vector_is_promoted(gamma_mass):
    cm = choice_mass(GammaMixturePredictor(), THETA[0], CHOICES)
    assert cm.cdf.shape == (1, 2, 1000)
    np.testing.assert_array_equal(cm.cdf[0], gamma_mass.cdf[0])


# --------------------------------------------------------------------------
# 3. mass_before
# --------------------------------------------------------------------------


def test_mass_before_at_max_t_is_mass(gamma_mass):
    grid = IntegrationGrid()
    for c in CHOICES:
        np.testing.assert_allclose(
            gamma_mass.mass_before(grid.max_t, c), gamma_mass.mass(c), rtol=1e-12
        )
    np.testing.assert_allclose(
        gamma_mass.mass_before(grid.max_t), gamma_mass.total, rtol=1e-12
    )


def test_mass_before_at_or_below_t_min_is_zero(gamma_mass):
    grid = IntegrationGrid()
    assert np.all(gamma_mass.mass_before(grid.t_min, 1) == 0.0)
    assert np.all(gamma_mass.mass_before(0.0, 1) == 0.0)
    assert np.all(gamma_mass.mass_before(-3.0) == 0.0)


def test_mass_before_on_and_between_grid_points(gamma_mass):
    t, cdf = gamma_mass.t, gamma_mass.cdf
    k = 137
    np.testing.assert_allclose(gamma_mass.mass_before(t[k], -1), cdf[:, 0, k])
    np.testing.assert_allclose(gamma_mass.mass_before(t[k]), cdf[:, :, k].sum(axis=1))

    midpoint = 0.5 * (t[k] + t[k + 1])
    np.testing.assert_allclose(
        gamma_mass.mass_before(midpoint, 1),
        0.5 * (cdf[:, 1, k] + cdf[:, 1, k + 1]),
    )


def test_mass_before_accepts_a_deadline_per_theta(gamma_mass):
    deadlines = np.array([0.5, 2.0, 10.0])
    got = gamma_mass.mass_before(deadlines, 1)
    assert got.shape == (len(THETA),)
    for i, d in enumerate(deadlines):
        assert got[i] == pytest.approx(gamma_mass.mass_before(d, 1)[i])
    with pytest.raises(ValueError):
        gamma_mass.mass_before(np.array([0.5, 2.0]), 1)


def test_mass_before_clamps_past_max_t(gamma_mass):
    for c in CHOICES:
        np.testing.assert_array_equal(
            gamma_mass.mass_before(1e6, c), gamma_mass.mass(c)
        )


# --------------------------------------------------------------------------
# 4. quantile
# --------------------------------------------------------------------------


@pytest.mark.parametrize("choice", [-1, 1, None])
def test_quantile_round_trips_through_mass_before(gamma_mass, choice):
    reference = gamma_mass.total if choice is None else gamma_mass.mass(choice)
    for u in (0.05, 0.5, 0.95):
        q = gamma_mass.quantile(u, choice)
        assert q.shape == (len(THETA),)
        assert np.all((q > gamma_mass.t[0]) & (q < gamma_mass.t[-1]))
        np.testing.assert_allclose(
            gamma_mass.mass_before(q, choice), u * reference, rtol=1e-9
        )


def test_quantile_is_monotone_in_u_and_close_to_scipy(gamma_mass):
    us = np.linspace(0.05, 0.95, 19)
    qs = np.stack([gamma_mass.quantile(u, 1) for u in us])
    assert np.all(np.diff(qs, axis=0) > 0.0)

    # Row 0 has negligible mass past max_t, so u of the truncated total is
    # essentially the untruncated gamma quantile.
    shape, scale = THETA[0]
    np.testing.assert_allclose(
        qs[:, 0], gamma.ppf(us, a=shape, scale=scale), rtol=2e-3, atol=2e-3
    )


def test_quantile_accepts_a_fraction_per_theta(gamma_mass):
    us = np.array([0.2, 0.5, 0.8])
    got = gamma_mass.quantile(us, -1)
    for i, u in enumerate(us):
        assert got[i] == pytest.approx(gamma_mass.quantile(u, -1)[i])


# --------------------------------------------------------------------------
# 5. Errors
# --------------------------------------------------------------------------


def test_wrong_input_width_names_expected_and_actual():
    class Wrong:
        input_width = 7

        def __call__(self, batch):  # pragma: no cover
            raise AssertionError("must not be called")

    with pytest.raises(ValueError, match=r"width 7.*2 parameters.*4 wide"):
        choice_mass(Wrong(), THETA, CHOICES)


def test_unknown_choice_code_raises(gamma_mass):
    with pytest.raises(ValueError, match=r"unknown choice code 2"):
        gamma_mass.mass(2)
    with pytest.raises(ValueError, match=r"unknown choice code"):
        gamma_mass.mass_before(1.0, 0)
    with pytest.raises(ValueError, match=r"unknown choice code"):
        gamma_mass.quantile(0.5, 3)


def test_quantile_rejects_u_outside_open_unit_interval(gamma_mass):
    for u in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            gamma_mass.quantile(u, 1)


def test_grid_validation():
    with pytest.raises(ValueError, match="n_points"):
        IntegrationGrid(n_points=1)
    with pytest.raises(ValueError, match="t_min"):
        IntegrationGrid(t_min=0.0)
    with pytest.raises(ValueError, match="t_min"):
        IntegrationGrid(t_min=30.0, max_t=20.0)
    grid = IntegrationGrid()
    assert grid.t[0] == grid.t_min and grid.t[-1] == grid.max_t
    assert grid.t.shape == (grid.n_points,)


def test_chunk_size_validation():
    with pytest.raises(ValueError, match="chunk_size"):
        choice_mass(GammaMixturePredictor(), THETA, CHOICES, chunk_size=0)
