"""Tests for lanfactory.derive.integrate: batched ONNX predictor and choice mass.

Fast and deterministic: a tiny untrained TorchMLP for the ONNX parity check,
and a closed-form gamma mixture standing in for a LAN for the numerics.
"""

from __future__ import annotations

import json
import time
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
    OnsetGrid,
    choice_mass,
    load_onnx_predictor,
    survey,
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


# --------------------------------------------------------------------------
# 6. OnsetGrid
# --------------------------------------------------------------------------


def test_onset_grid_rows_are_sorted_unique_rectangular_and_contain_t():
    grid = OnsetGrid()
    onsets = np.array([0.0, 0.03, 0.4, 1.0, 2.0])
    rows = grid.for_theta(onsets)

    assert grid.n_points == 32 + 128 + 600 + 400 == 1160
    assert rows.shape == (len(onsets), grid.n_points)
    assert np.all(np.diff(rows, axis=1) > 0.0), "rows must be strictly increasing"
    np.testing.assert_array_equal(rows[:, 0], grid.t_min)
    np.testing.assert_array_equal(rows[:, -1], grid.max_t)

    # Every onset inside the clip interval appears exactly, as a grid point.
    for row, t in zip(rows[1:], onsets[1:], strict=True):
        assert np.any(row == t), f"onset {t} missing from its row"
        assert np.sum(row < t) == 32 + 128, "pre and knee points sit below t"
    # t = 0 (the bottom of the ddm box) is clipped up by the margin, not to
    # t_min itself, so the pre/knee segments keep positive length.
    lo, _ = grid.onset_bounds
    assert lo == pytest.approx(grid.t_min + 1e-3)
    assert np.any(rows[0] == lo) and not np.any(rows[0] == 0.0)
    assert np.sum(rows[0] < lo) == 32 + 128


def test_onset_grid_clips_the_onset_at_the_top():
    # With the default max_t = 20 an onset of 2 s needs no clipping; shrink
    # max_t so the same onset hits the upper clip and the tail keeps room.
    grid = OnsetGrid(max_t=2.5)
    _, hi = grid.onset_bounds
    assert hi == pytest.approx(2.5 - 1.0 - 1e-3)
    (row,) = grid.for_theta([2.0])
    assert np.all(np.diff(row) > 0.0)
    assert np.any(row == hi) and not np.any(row == 2.0)
    assert row[-1] == 2.5 and row[0] == grid.t_min
    assert np.sum(row >= hi + grid.onset_window) == grid.n_tail


def test_onset_grid_knee_moves_up_when_t_is_below_the_knee_width():
    grid = OnsetGrid()
    (row,) = grid.for_theta([0.03])  # t - knee < t_min
    knee_start = row[grid.n_pre]
    assert knee_start == pytest.approx(0.5 * (grid.t_min + 0.03))
    (row,) = grid.for_theta([0.4])
    assert row[grid.n_pre] == pytest.approx(0.4 - grid.knee)


def test_onset_grid_pre_segment_never_degenerates():
    # A threshold rule ("t - knee > t_min, else the midpoint") gives the pre
    # segment a width of t - knee - t_min just above t_min + knee, which goes
    # to zero continuously: 32 points crammed into ~1e-14 s, distinct in
    # float64 but duplicates in the float32 rows the network sees. The max
    # rule keeps the pre segment at least margin / 2 wide everywhere.
    grid = OnsetGrid()
    onsets = np.concatenate(
        [
            [grid.t_min + grid.knee + 1e-9, grid.t_min + grid.knee + 1e-14],
            np.linspace(0.0, 2.0, 2001),
        ]
    )
    rows = grid.for_theta(onsets)
    pre_width = rows[:, grid.n_pre] - rows[:, 0]
    assert np.all(pre_width >= 0.5 * grid.margin)
    assert pre_width[0] == pytest.approx(0.5 * grid.knee, abs=1e-6)
    # Every row stays strictly increasing once rounded to float32.
    assert np.all(np.diff(rows.astype(np.float32), axis=1) > 0)
    # The two rules meet at t = t_min + 2 knee: the knee start is continuous.
    switch = grid.t_min + 2 * grid.knee
    (row,) = grid.for_theta([switch])
    assert row[grid.n_pre] == pytest.approx(switch - grid.knee)
    assert row[grid.n_pre] == pytest.approx(0.5 * (grid.t_min + switch))
    knee_start = rows[2:, grid.n_pre]
    assert np.all(np.abs(np.diff(knee_start)) < 2 * (2.0 / 2000))


def test_onset_grid_validation():
    with pytest.raises(ValueError, match="n_pre"):
        OnsetGrid(n_pre=0)
    with pytest.raises(ValueError, match="n_tail"):
        OnsetGrid(n_tail=1)
    with pytest.raises(ValueError, match="knee"):
        OnsetGrid(knee=0.0)
    with pytest.raises(ValueError, match="onset_window"):
        OnsetGrid(onset_window=-1.0)
    with pytest.raises(ValueError, match="t_min"):
        OnsetGrid(t_min=0.0)
    with pytest.raises(ValueError, match="max_t"):
        OnsetGrid(max_t=1.0, onset_window=1.0)
    with pytest.raises(ValueError, match=r"\(n_theta,\)"):
        OnsetGrid().for_theta(np.zeros((2, 2)))


# --------------------------------------------------------------------------
# 7. Per-theta grid: a peaked density with a sharp onset
# --------------------------------------------------------------------------

# shape = 2 keeps the density C^1 at the onset (as in section 2); scale 0.03
# puts the whole peak inside two steps of the uniform 1000-point grid.
PEAK_SHAPE, PEAK_SCALE = 2.0, 0.03
ONSETS = np.array([[0.0], [0.4], [2.0]])


class PeakedPredictor:
    """Rows ``[t, rt, choice]`` -> log( w[choice] * Gamma(rt - t) ), 0 below t."""

    input_width = 3

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=np.float64)
        t, rt, choice = batch.T
        log_w = np.where(choice == 1, np.log(WEIGHTS[1]), np.log(WEIGHTS[-1]))
        x = rt - t
        log_density = np.full(x.shape, -np.inf)
        above = x > 0.0
        log_density[above] = gamma.logpdf(x[above], a=PEAK_SHAPE, scale=PEAK_SCALE)
        return log_w + log_density


def expected_peak_mass(choice: int, t_min: float = 1e-4, max_t: float = 20.0):
    """``w[choice] * (G(max_t - t) - G(max(t_min - t, 0)))`` per onset row."""
    t = ONSETS[:, 0]
    g = gamma.cdf(max_t - t, a=PEAK_SHAPE, scale=PEAK_SCALE) - gamma.cdf(
        np.maximum(t_min - t, 0.0), a=PEAK_SHAPE, scale=PEAK_SCALE
    )
    return WEIGHTS[choice] * g


@pytest.fixture(scope="module")
def peak_mass() -> ChoiceMass:
    return choice_mass(
        PeakedPredictor(), ONSETS, CHOICES, grid=OnsetGrid(), onset=ONSETS[:, 0]
    )


def test_uniform_grid_is_not_fit_for_a_peaked_density():
    # Pins the grid policy: 1000 uniform points (20 ms apart) straddle a
    # peak that is a few tens of ms wide, and the total comes out wrong by
    # more than the corpus's own 0.02 flag threshold.
    uniform = choice_mass(PeakedPredictor(), ONSETS, CHOICES, grid=IntegrationGrid())
    err = np.abs(uniform.total - sum(expected_peak_mass(c) for c in CHOICES))
    (row_0_4,) = np.flatnonzero(ONSETS[:, 0] == 0.4)
    assert err[row_0_4] > 0.02
    # How wrong depends on where the onset falls between two grid points (an
    # artefact in its own right), but never within the refined grid's 1e-3.
    assert np.all(err > 0.01)


def test_onset_grid_resolves_the_peak_to_1e3(peak_mass):
    for c in CHOICES:
        err = np.abs(peak_mass.mass(c) - expected_peak_mass(c))
        assert err.max() < 1e-3, f"choice {c}: max error {err.max():.2e}"
    assert np.all(
        np.abs(peak_mass.total - sum(expected_peak_mass(c) for c in CHOICES)) < 1e-3
    )


def test_onset_grid_result_carries_a_grid_per_theta(peak_mass):
    grid = OnsetGrid()
    assert peak_mass.t.shape == (len(ONSETS), grid.n_points)
    assert peak_mass.cdf.shape == (len(ONSETS), len(CHOICES), grid.n_points)
    np.testing.assert_array_equal(peak_mass.t, grid.for_theta(ONSETS[:, 0]))
    assert np.all(peak_mass.cdf[..., 0] == 0.0)
    assert np.all(np.diff(peak_mass.cdf, axis=-1) >= 0.0)


class RecordingPredictor:
    """Wraps any predictor and records every batch it is handed."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.input_width = inner.input_width
        self.calls: list[np.ndarray] = []

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        self.calls.append(np.array(batch, copy=True))
        return self.inner(batch)


def test_onset_grid_chunks_agree_with_single_theta_runs(peak_mass):
    spy = RecordingPredictor(PeakedPredictor())
    chunked = choice_mass(
        spy, ONSETS, CHOICES, grid=OnsetGrid(), chunk_size=2, onset=ONSETS[:, 0]
    )
    per_theta = len(CHOICES) * OnsetGrid().n_points
    assert [c.shape[0] for c in spy.calls] == [2 * per_theta, per_theta]
    np.testing.assert_array_equal(chunked.cdf, peak_mass.cdf)
    np.testing.assert_array_equal(chunked.t, peak_mass.t)
    for i in range(len(ONSETS)):
        single = choice_mass(
            PeakedPredictor(), ONSETS[i], CHOICES, grid=OnsetGrid(), onset=ONSETS[i]
        )
        np.testing.assert_array_equal(peak_mass.cdf[i], single.cdf[0])


def test_leak_below_is_mass_before_summed_over_choices(peak_mass):
    onset = ONSETS[:, 0]
    np.testing.assert_array_equal(
        peak_mass.leak_below(onset), peak_mass.mass_before(onset)
    )
    # The synthetic density is exactly zero at and below the onset.
    np.testing.assert_array_equal(peak_mass.leak_below(onset), 0.0)
    # A deadline past the onset does see mass: the definition is not a no-op.
    assert np.all(peak_mass.leak_below(onset + 0.05) > 0.1)


def test_choice_mass_rejects_mismatched_grid_and_onset():
    with pytest.raises(ValueError, match="OnsetGrid needs onset"):
        choice_mass(PeakedPredictor(), ONSETS, CHOICES, grid=OnsetGrid())
    with pytest.raises(ValueError, match="takes no onset"):
        choice_mass(PeakedPredictor(), ONSETS, CHOICES, onset=ONSETS[:, 0])
    with pytest.raises(ValueError, match=r"onset must be \(n_theta,\)"):
        choice_mass(
            PeakedPredictor(), ONSETS, CHOICES, grid=OnsetGrid(), onset=ONSETS[:2, 0]
        )


# --------------------------------------------------------------------------
# 8. mass_before / quantile on a per-theta (2-D) grid
# --------------------------------------------------------------------------


def test_mass_before_on_2d_grid_hits_grid_points_and_clamps(peak_mass):
    t, cdf = peak_mass.t, peak_mass.cdf
    k = 32 + 128 + 40  # 40 points into the onset segment: the peak region
    np.testing.assert_allclose(peak_mass.mass_before(t[:, k], -1), cdf[:, 0, k])
    np.testing.assert_allclose(peak_mass.mass_before(t[:, k]), cdf[:, :, k].sum(axis=1))
    midpoint = 0.5 * (t[:, k] + t[:, k + 1])
    np.testing.assert_allclose(
        peak_mass.mass_before(midpoint, 1), 0.5 * (cdf[:, 1, k] + cdf[:, 1, k + 1])
    )
    # Per-row deadlines use each row's own grid, not a shared one.
    deadlines = ONSETS[:, 0] + np.array([0.01, 0.05, 0.2])
    got = peak_mass.mass_before(deadlines, 1)
    for i, d in enumerate(deadlines):
        assert got[i] == pytest.approx(peak_mass.mass_before(d, 1)[i])
    assert got[0] < got[1] < got[2]
    # Clamping at both ends.
    for c in CHOICES:
        np.testing.assert_array_equal(peak_mass.mass_before(1e6, c), peak_mass.mass(c))
        np.testing.assert_array_equal(peak_mass.mass_before(-1.0, c), 0.0)
        np.testing.assert_array_equal(peak_mass.mass_before(0.0, c), 0.0)


@pytest.mark.parametrize("choice", [-1, 1, None])
def test_quantile_on_2d_grid_round_trips_and_matches_scipy(peak_mass, choice):
    reference = peak_mass.total if choice is None else peak_mass.mass(choice)
    t = peak_mass.t
    for u in (0.05, 0.5, 0.95):
        q = peak_mass.quantile(u, choice)
        assert q.shape == (len(ONSETS),)
        assert np.all((q > t[:, 0]) & (q < t[:, -1]))
        np.testing.assert_allclose(
            peak_mass.mass_before(q, choice), u * reference, rtol=1e-9
        )
        # Each row's quantile sits on that row's own onset.
        expected = ONSETS[:, 0] + gamma.ppf(u, a=PEAK_SHAPE, scale=PEAK_SCALE)
        np.testing.assert_allclose(q, expected, atol=2e-3)


# --------------------------------------------------------------------------
# 9. survey
# --------------------------------------------------------------------------

SURVEY_BOUNDS = {"a": (0.3, 2.5), "t": (0.0, 2.0)}
SURVEY_PARAMS = ["a", "t"]
PLANTED_SHARE = (2.5 - 2.0) / (2.5 - 0.3)


class ScaledPredictor:
    """Rows ``[a, t, rt, choice]``: a smooth shifted gamma, times 1.2 when a > 2.

    Zero at and below ``t``; shape 2 / scale 0.2 is resolved to ~1e-4 by both
    grids, so any deviation from one is the planted scale, not quadrature.
    """

    input_width = 4

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=np.float64)
        a, t, rt, choice = batch.T
        log_w = np.where(choice == 1, np.log(WEIGHTS[1]), np.log(WEIGHTS[-1]))
        x = rt - t
        log_density = np.full(x.shape, -np.inf)
        above = x > 0.0
        log_density[above] = gamma.logpdf(x[above], a=2.0, scale=0.2)
        return log_w + log_density + np.where(a > 2.0, np.log(1.2), 0.0)


@pytest.fixture(scope="module")
def scaled_survey() -> dict:
    return survey(
        ScaledPredictor(), SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, n_theta=4000, seed=3
    )


def test_survey_reports_the_planted_share_of_bad_totals(scaled_survey):
    r = scaled_survey
    assert r["n_theta"] == 4000
    assert r["grid"] == repr(OnsetGrid())
    total = r["total"]
    assert total["frac_gt_0.10"] == pytest.approx(PLANTED_SHARE, abs=0.03)
    assert total["frac_gt_0.05"] == total["frac_gt_0.10"] == total["frac_gt_0.02"]
    assert total["max"] == pytest.approx(1.2, abs=1e-3)
    assert total["min"] == pytest.approx(1.0, abs=1e-3)
    assert total["p50_abs_dev"] < 1e-3
    assert total["p99_abs_dev"] == pytest.approx(0.2, abs=1e-3)
    assert total["mean"] == pytest.approx(1.0 + 0.2 * PLANTED_SHARE, abs=0.01)


def test_survey_by_param_localises_the_defect(scaled_survey):
    bins = scaled_survey["by_param"]["a"]
    assert len(bins) == 10 and len(scaled_survey["by_param"]["t"]) == 10
    assert bins[0]["lo"] == 0.3 and bins[-1]["hi"] == 2.5
    assert sum(b["n"] for b in bins) == 4000
    for b in bins:
        if b["lo"] >= 2.0:
            assert b["mean_dev"] == pytest.approx(0.2, abs=1e-3)
            assert b["max_abs_dev"] == pytest.approx(0.2, abs=1e-3)
        elif b["hi"] <= 2.0:
            assert abs(b["mean_dev"]) < 1e-3
            assert b["max_abs_dev"] < 1e-3
    # No t-bin is special: the planted defect is in a only.
    assert all(
        abs(b["mean_dev"] - 0.2 * PLANTED_SHARE) < 0.05
        for b in scaled_survey["by_param"]["t"]
    )


def test_survey_worst_cell_sits_above_a_equals_two(scaled_survey):
    cell = scaled_survey["worst_cell"]
    assert {cell["param_x"], cell["param_y"]} == {"a", "t"}
    axis = "x" if cell["param_x"] == "a" else "y"
    assert cell[f"{axis}_lo"] >= 2.0
    assert cell["mean_dev"] == pytest.approx(0.2, abs=1e-3)
    assert cell["n"] > 0


def test_survey_leak_and_shrunk_box(scaled_survey):
    leak = scaled_survey["leak_below_onset"]
    # Exactly zero except for onsets below the grid clip (t < 1.1e-3), where
    # a sliver of the true density sits below the clipped onset: ~1e-13.
    assert leak["p99"] == 0.0 and leak["mean"] < 1e-9 and leak["max"] < 1e-9
    box = scaled_survey["shrunk_box"]
    assert set(box) == set(scaled_survey["total"]) | {"frac_of_theta"}
    assert box["frac_of_theta"] == pytest.approx(0.8**2, abs=0.03)
    # Shrinking the box by 10% per side keeps a in [0.52, 2.28]: fewer bad theta.
    assert box["frac_gt_0.10"] == pytest.approx((2.28 - 2.0) / (2.28 - 0.52), abs=0.03)


def test_survey_is_json_serialisable_and_fast(scaled_survey):
    json.dumps(scaled_survey)
    started = time.perf_counter()
    r = survey(ScaledPredictor(), SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, n_theta=2000)
    elapsed = time.perf_counter() - started
    assert r["seconds"] < 30.0 and elapsed < 30.0
    assert r["seconds"] <= elapsed


def test_survey_without_an_onset_uses_the_uniform_grid():
    r = survey(
        ScaledPredictor(),
        SURVEY_BOUNDS,
        SURVEY_PARAMS,
        CHOICES,
        n_theta=300,
        onset_param=None,
    )
    assert r["grid"] == repr(IntegrationGrid())
    assert r["leak_below_onset"] is None
    assert r["total"]["frac_gt_0.10"] == pytest.approx(PLANTED_SHARE, abs=0.08)


def test_survey_is_deterministic_in_the_seed():
    kwargs = dict(n_theta=200, seed=11)
    a = survey(ScaledPredictor(), SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, **kwargs)
    b = survey(ScaledPredictor(), SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, **kwargs)
    a.pop("seconds"), b.pop("seconds")
    assert a == b


def test_survey_validation():
    pred = ScaledPredictor()
    with pytest.raises(ValueError, match="onset_param 'tau'"):
        survey(pred, SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, onset_param="tau")
    with pytest.raises(ValueError, match="lacks bounds"):
        survey(pred, {"a": (0.3, 2.5)}, SURVEY_PARAMS, CHOICES)
    with pytest.raises(ValueError, match="does not go with"):
        survey(pred, SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, grid=IntegrationGrid())
    with pytest.raises(ValueError, match="does not go with"):
        survey(
            pred,
            SURVEY_BOUNDS,
            SURVEY_PARAMS,
            CHOICES,
            onset_param=None,
            grid=OnsetGrid(),
        )
    with pytest.raises(ValueError, match="n_theta"):
        survey(pred, SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, n_theta=0)
    with pytest.raises(ValueError, match="shrink"):
        survey(pred, SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, shrink=0.5)
