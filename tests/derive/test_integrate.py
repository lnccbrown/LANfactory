"""Tests for lanfactory.derive.integrate: batched ONNX predictor and choice mass.

Fast and deterministic: a tiny untrained TorchMLP for the ONNX parity check,
and a closed-form gamma mixture standing in for a LAN for the numerics.
"""

from __future__ import annotations

import json

import numpy as np
import onnx
import pytest
import torch
from scipy.integrate import trapezoid
from scipy.stats import gamma

from lanfactory.derive import (
    ChoiceMass,
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
# shape >= 2 keeps the density continuous (C^0) at zero so the trapezoid rule
# keeps its O(h^2) rate (the derivative still jumps there for shape = 2); the
# (4, 2) row puts ~1% of its mass past max_t.
THETA = np.array([[2.0, 0.5], [3.0, 1.0], [4.0, 2.0]])
# The gamma mixture has no onset parameter: its grid is built around onset 0.
ZEROS = np.zeros(len(THETA))


class GammaMixturePredictor:
    """Rows ``[shape, scale, rt, choice]`` -> log( w[choice] * Gamma(rt) )."""

    input_width = 4

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=np.float64)
        shape, scale, rt, choice = batch.T
        log_w = np.where(choice == 1, np.log(WEIGHTS[1]), np.log(WEIGHTS[-1]))
        return log_w + gamma.logpdf(rt, a=shape, scale=scale)


def expected_mass(choice: int, grid: OnsetGrid) -> np.ndarray:
    """``w[choice] * (G(max_t) - G(t_min))`` per theta row."""
    shape, scale = THETA.T
    g = gamma.cdf(grid.max_t, a=shape, scale=scale) - gamma.cdf(
        grid.t_min, a=shape, scale=scale
    )
    return WEIGHTS[choice] * g


@pytest.fixture(scope="module")
def gamma_mass() -> ChoiceMass:
    return choice_mass(GammaMixturePredictor(), THETA, CHOICES, onset=ZEROS)


def test_masses_match_closed_form(gamma_mass):
    for choice in CHOICES:
        err = np.abs(gamma_mass.mass(choice) - expected_mass(choice, OnsetGrid()))
        assert err.max() < 1e-3, f"choice {choice}: max error {err.max():.2e}"


def test_mass_error_shrinks_with_grid_resolution():
    # Measured 1.15e-3, 7.1e-5, 4.4e-6: second order in the point count.
    def max_err(k: float) -> float:
        grid = OnsetGrid(
            n_pre=int(32 * k),
            n_knee=int(128 * k),
            n_onset=int(600 * k),
            n_tail=int(400 * k),
        )
        cm = choice_mass(
            GammaMixturePredictor(), THETA, CHOICES, grid=grid, onset=ZEROS
        )
        return max(np.abs(cm.mass(c) - expected_mass(c, grid)).max() for c in CHOICES)

    coarse, default, fine = (max_err(k) for k in (0.25, 1, 4))
    assert coarse > default > fine


def test_total_is_sum_over_choices(gamma_mass):
    summed = sum(gamma_mass.mass(c) for c in CHOICES)
    np.testing.assert_allclose(gamma_mass.total, summed, rtol=0, atol=1e-12)
    # The (4, 2) row deliberately leaks mass past max_t: no renormalisation.
    assert gamma_mass.total[2] < 0.995


def test_cdf_shape_monotone_and_matches_trapezoid(gamma_mass):
    assert gamma_mass.cdf.shape == (len(THETA), len(CHOICES), OnsetGrid().n_points)
    assert np.all(gamma_mass.cdf[..., 0] == 0.0)
    assert np.all(np.diff(gamma_mass.cdf, axis=-1) >= 0.0)

    # Rows reach the predictor as float32 (a LAN's input dtype), so rt carries
    # float32 rounding relative to the float64 grid: agree to ~1e-7, not 1e-15.
    t = gamma_mass.t
    for i, (shape, scale) in enumerate(THETA):
        for j, c in enumerate(CHOICES):
            density = WEIGHTS[c] * gamma.pdf(t[i], a=shape, scale=scale)
            np.testing.assert_allclose(
                gamma_mass.cdf[i, j, -1], trapezoid(density, t[i]), rtol=1e-6
            )


def test_single_theta_vector_is_promoted(gamma_mass):
    cm = choice_mass(GammaMixturePredictor(), THETA[0], CHOICES, onset=ZEROS[:1])
    assert cm.cdf.shape == (1, 2, OnsetGrid().n_points)
    np.testing.assert_array_equal(cm.cdf[0], gamma_mass.cdf[0])


# --------------------------------------------------------------------------
# 3. quantile
# --------------------------------------------------------------------------


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
# 4. Errors
# --------------------------------------------------------------------------


def test_wrong_input_width_names_expected_and_actual():
    class Wrong:
        input_width = 7

        def __call__(self, batch):  # pragma: no cover
            raise AssertionError("must not be called")

    with pytest.raises(ValueError, match=r"width 7.*2 parameters.*4 wide"):
        choice_mass(Wrong(), THETA, CHOICES, onset=ZEROS)


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


def test_quantile_is_nan_without_mass():
    # Row 0 has mass on choice -1 only, row 1 none, row 2 a NaN total.
    cdf = np.zeros((3, 2, 3))
    cdf[0, 0] = [0.0, 0.5, 1.0]
    cdf[2] = np.nan
    mass = ChoiceMass(t=np.tile([0.1, 0.2, 0.3], (3, 1)), choices=CHOICES, cdf=cdf)
    q = mass.quantile(0.5)
    assert q[0] == pytest.approx(0.2) and np.isnan(q[1:]).all()
    assert np.isnan(mass.quantile(0.5, 1)).all()


def test_chunk_size_validation():
    with pytest.raises(ValueError, match="chunk_size"):
        choice_mass(GammaMixturePredictor(), THETA, CHOICES, chunk_size=0, onset=ZEROS)


# --------------------------------------------------------------------------
# 5. OnsetGrid
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
    (row,) = grid.for_theta([0.4])
    assert row[grid.n_pre] == pytest.approx(0.4 - grid.knee)


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
# 6. Per-theta grid: a peaked density with a sharp onset
# --------------------------------------------------------------------------

# shape = 2 keeps the density continuous (C^0) at the onset (as in section 2)
# so the trapezoid rule keeps its O(h^2) rate; the derivative still jumps from
# 0 to 1/scale^2 there, so a grid without a point at the onset errs by an
# amount that depends on where the onset falls between two of its points.
# Scale 0.03 puts the whole peak inside a few tens of ms.
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


def expected_peak_mass(
    choice: int,
    onsets: np.ndarray = ONSETS[:, 0],
    t_min: float = 1e-4,
    max_t: float = 20.0,
):
    """``w[choice] * (G(max_t - t) - G(max(t_min - t, 0)))`` per onset."""
    t = np.asarray(onsets, dtype=np.float64)
    g = gamma.cdf(max_t - t, a=PEAK_SHAPE, scale=PEAK_SCALE) - gamma.cdf(
        np.maximum(t_min - t, 0.0), a=PEAK_SHAPE, scale=PEAK_SCALE
    )
    return WEIGHTS[choice] * g


@pytest.fixture(scope="module")
def peak_mass() -> ChoiceMass:
    return choice_mass(
        PeakedPredictor(), ONSETS, CHOICES, grid=OnsetGrid(), onset=ONSETS[:, 0]
    )


def test_onset_grid_is_accurate_at_every_onset_phase():
    # Pins the grid policy: a uniform 1000-point grid (20 ms apart) missed
    # this peak by 1e-4 to 0.036 depending on where the onset fell between
    # two of its points. Sweep the onset over one such step: the refined grid
    # stays within 1e-3 at every phase.
    onsets = 0.4 + 0.02 * np.arange(8) / 8
    expected = sum(expected_peak_mass(c, onsets) for c in CHOICES)
    mass = choice_mass(PeakedPredictor(), onsets[:, None], CHOICES, onset=onsets)
    assert np.abs(mass.total - expected).max() < 1e-3


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


def test_choice_mass_rejects_an_onset_of_the_wrong_shape():
    with pytest.raises(ValueError, match=r"onset must be \(n_theta,\)"):
        choice_mass(
            PeakedPredictor(), ONSETS, CHOICES, grid=OnsetGrid(), onset=ONSETS[:2, 0]
        )


# --------------------------------------------------------------------------
# 7. mass_before / quantile on a per-theta (2-D) grid
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
    # The synthetic density is exactly zero at and below the onset; a deadline
    # past the onset does see mass (what survey's leak_below_onset relies on).
    onset = ONSETS[:, 0]
    np.testing.assert_array_equal(peak_mass.mass_before(onset), 0.0)
    assert np.all(peak_mass.mass_before(onset + 0.05) > 0.1)
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
# 8. survey
# --------------------------------------------------------------------------

SURVEY_BOUNDS = {"a": (0.3, 2.5), "t": (0.0, 2.0)}
SURVEY_PARAMS = ["a", "t"]
PLANTED_SHARE = (2.5 - 2.0) / (2.5 - 0.3)


class ScaledPredictor:
    """Rows ``[a, t, rt, choice]``: a smooth shifted gamma, times 1.2 when a > 2.

    Zero at and below ``t``; shape 2 / scale 0.2 is resolved to ~1e-4 by a grid
    built around ``t``, so any deviation from one is the planted scale.
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
    json.dumps(r)
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


def test_survey_leak_and_shrunk_box(scaled_survey):
    leak = scaled_survey["leak_below_onset"]
    # Exactly zero except for onsets below the grid clip (t < 1.1e-3), where
    # a sliver of the true density sits below the clipped onset: at most
    # ~1e-9 (one pre-segment interval of the trapezoid, spacing ~1.6e-5 s).
    assert leak["p99"] == 0.0 and leak["mean"] < 1e-9 and leak["max"] < 1e-9
    box = scaled_survey["shrunk_box"]
    assert set(box) == set(scaled_survey["total"]) | {"frac_of_theta"}
    assert box["frac_of_theta"] == pytest.approx(0.8**2, abs=0.03)
    # Shrinking the box by 10% per side keeps a in [0.52, 2.28]: fewer bad theta.
    assert box["frac_gt_0.10"] == pytest.approx((2.28 - 2.0) / (2.28 - 0.52), abs=0.03)


def test_survey_without_an_onset_builds_the_grid_around_zero():
    r = survey(
        ScaledPredictor(),
        SURVEY_BOUNDS,
        SURVEY_PARAMS,
        CHOICES,
        n_theta=300,
        onset_param=None,
    )
    assert r["grid"] == repr(OnsetGrid())
    assert r["leak_below_onset"] is None
    assert r["total"]["frac_gt_0.10"] == pytest.approx(PLANTED_SHARE, abs=0.08)


class NaNPredictor(ScaledPredictor):
    """:class:`ScaledPredictor`, but NaN where that one scales (a > 2)."""

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        return np.where(batch[:, 0] > 2.0, np.nan, super().__call__(batch))


def test_survey_counts_a_nan_total_as_failed():
    r = survey(NaNPredictor(), SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, n_theta=300)
    json.dumps(r)
    total = r["total"]
    assert total["frac_gt_0.02"] == pytest.approx(PLANTED_SHARE, abs=0.08)
    assert total["p99_abs_dev"] == np.inf
    assert total["p50_abs_dev"] < 1e-3


def test_survey_validation():
    pred = ScaledPredictor()
    with pytest.raises(ValueError, match="onset_param 'tau'"):
        survey(pred, SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, onset_param="tau")
    with pytest.raises(ValueError, match="lacks bounds"):
        survey(pred, {"a": (0.3, 2.5)}, SURVEY_PARAMS, CHOICES)
    with pytest.raises(ValueError, match="n_theta"):
        survey(pred, SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, n_theta=0)
