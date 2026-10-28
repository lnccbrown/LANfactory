"""Tests for lanfactory.derive.integrate: batched ONNX predictor and choice mass.

Fast and deterministic: the production ddm LAN for the ONNX parity check, and
closed-form shifted gamma densities standing in for a LAN for the numerics.
"""

from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
import pytest
from scipy.stats import gamma

from lanfactory.derive import (
    ChoiceMass,
    OnsetGrid,
    choice_mass,
    load_onnx_predictor,
    survey,
)
from lanfactory.derive import integrate as integrate_module
from lanfactory.onnx import assert_single_trial_contract
from tests._onnx_utils import DDM_ONNX
from tests.onnx.test_contract import make_onnx

# --------------------------------------------------------------------------
# 1. Batched ONNX predictor
# --------------------------------------------------------------------------


def test_batched_predictor_matches_the_published_session_and_leaves_file_untouched():
    before = DDM_ONNX.read_bytes()
    predictor = load_onnx_predictor(DDM_ONNX)
    assert predictor.input_width == 6
    assert isinstance(predictor.input_name, str)

    # Rows [v, a, z, t, rt, choice], fed one at a time to the file as published.
    rng = np.random.default_rng(0)
    rows = rng.uniform([-3, 0.3, 0.1, 0, 0, -1], [3, 2.5, 0.9, 2, 5, 1], (64, 6))
    rows[:, 5] = np.sign(rows[:, 5])
    rows = rows.astype(np.float32)
    session = ort.InferenceSession(str(DDM_ONNX))
    name = session.get_inputs()[0].name
    expected = [session.run(None, {name: row[None, :]})[0].item() for row in rows]
    np.testing.assert_allclose(predictor(rows), expected, rtol=1e-5, atol=1e-5)

    # Default-dtype (float64) rows are cast, not rejected by onnxruntime.
    np.testing.assert_array_equal(predictor(rows.astype(np.float64)), predictor(rows))
    # The batch axis was made symbolic in memory only.
    assert DDM_ONNX.read_bytes() == before
    assert assert_single_trial_contract(DDM_ONNX, expected_input_width=6)


def test_predictor_rejects_wrong_row_width(tmp_path):
    predictor = load_onnx_predictor(make_onnx(tmp_path / "m.onnx", (1, 6)))
    with pytest.raises(ValueError, match=r"\(N, 6\)"):
        predictor(np.zeros((3, 7), dtype=np.float32))


def test_rank_one_artifact_is_rejected_clearly(tmp_path):
    # sbi / bayesflow exports trace (D,): no batch axis to widen.
    with pytest.raises(ValueError, match=r"rank 1"):
        load_onnx_predictor(make_onnx(tmp_path / "rank1.onnx", (6,)))


# --------------------------------------------------------------------------
# 2. Closed form: shifted gamma densities
# --------------------------------------------------------------------------

CHOICES = np.array([-1, 1])
WEIGHTS = {-1: 0.3, 1: 0.7}


def log_density(choice, x, shape, scale):
    """``log(w[choice] * Gamma(x))``; ``-inf`` at and below 0."""
    log_w = np.where(choice == 1, np.log(WEIGHTS[1]), np.log(WEIGHTS[-1]))
    return log_w + gamma.logpdf(x, a=shape, scale=scale)


def expected_mass(choice, onset, shape, scale, grid=OnsetGrid()):  # noqa: B008
    """``w[choice] * (G(max_t - t) - G(max(t_min - t, 0)))``: mass on the grid."""
    g = gamma.cdf(grid.max_t - onset, a=shape, scale=scale) - gamma.cdf(
        np.maximum(grid.t_min - onset, 0.0), a=shape, scale=scale
    )
    return WEIGHTS[choice] * g


class GammaMixturePredictor:
    """Rows ``[shape, scale, rt, choice]``: a gamma mixture with onset 0."""

    input_width = 4

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        shape, scale, rt, choice = np.asarray(batch, dtype=np.float64).T
        return log_density(choice, rt, shape, scale)


# shape >= 2 keeps the density continuous (C^0) at zero so the trapezoid rule
# keeps its O(h^2) rate (the derivative still jumps there for shape = 2); the
# (4, 2) row puts ~1% of its mass past max_t.
THETA = np.array([[2.0, 0.5], [3.0, 1.0], [4.0, 2.0]])


@pytest.fixture(scope="module")
def gamma_mass() -> ChoiceMass:
    # The mixture has no onset parameter: a scalar 0 builds every row around 0.
    return choice_mass(GammaMixturePredictor(), THETA, CHOICES, 0.0)


def test_masses_match_closed_form_without_renormalising(gamma_mass):
    # The (4, 2) row loses ~1 % past max_t: a renormalised total would miss
    # the closed form by far more than the tolerance.
    for j, choice in enumerate(CHOICES):
        err = np.abs(gamma_mass.by_choice[:, j] - expected_mass(choice, 0.0, *THETA.T))
        assert err.max() < 1e-3, f"choice {choice}: max error {err.max():.2e}"
    np.testing.assert_allclose(gamma_mass.total, gamma_mass.by_choice.sum(axis=1))


def test_mass_error_shrinks_with_grid_resolution():
    # Measured 1.15e-3, 7.1e-5, 4.4e-6: second order in the point count.
    def max_err(k: float) -> float:
        grid = OnsetGrid(
            n_pre=int(32 * k),
            n_knee=int(128 * k),
            n_onset=int(600 * k),
            n_tail=int(400 * k),
        )
        cm = choice_mass(GammaMixturePredictor(), THETA, CHOICES, 0.0, grid=grid)
        return max(
            np.abs(cm.by_choice[:, j] - expected_mass(c, 0.0, *THETA.T, grid)).max()
            for j, c in enumerate(CHOICES)
        )

    coarse, default, fine = (max_err(k) for k in (0.25, 1, 4))
    assert coarse / default > 10 and default / fine > 10  # 16 at second order


def test_quantile_accepts_a_fraction_per_theta(gamma_mass):
    us = np.array([0.2, 0.5, 0.8])
    got = gamma_mass.quantile(us)
    for i, u in enumerate(us):
        assert got[i] == pytest.approx(gamma_mass.quantile(u)[i])


# --------------------------------------------------------------------------
# 3. Errors
# --------------------------------------------------------------------------


def test_wrong_input_width_names_expected_and_actual():
    class Wrong:
        input_width = 7

        def __call__(self, batch):  # pragma: no cover
            raise AssertionError("must not be called")

    with pytest.raises(ValueError, match=r"width 7.*2 parameters.*4 wide"):
        choice_mass(Wrong(), THETA, CHOICES, 0.0)


def test_quantile_rejects_u_outside_open_unit_interval(gamma_mass):
    for u in (0.0, 1.0, -0.1, 1.5, np.nan):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            gamma_mass.quantile(u)


def test_quantile_is_nan_without_mass():
    # Row 0 has mass, row 1 none, row 2 a NaN total.
    cdf = np.zeros((3, 3))
    cdf[0] = [0.0, 0.5, 1.0]
    cdf[2] = np.nan
    mass = ChoiceMass(
        t=np.tile([0.1, 0.2, 0.3], (3, 1)), cdf=cdf, by_choice=cdf[:, -1:]
    )
    q = mass.quantile(0.5)
    assert q[0] == pytest.approx(0.2) and np.isnan(q[1:]).all()


def test_inputs_of_the_wrong_shape_are_rejected():
    with pytest.raises(ValueError, match=r"theta must be \(n_theta, n_params\)"):
        choice_mass(GammaMixturePredictor(), np.zeros((1, 2, 2)), CHOICES, 0.0)
    with pytest.raises(ValueError, match="broadcast"):
        choice_mass(GammaMixturePredictor(), THETA, CHOICES, np.zeros(2))
    with pytest.raises(ValueError, match="onset must be finite"):
        choice_mass(GammaMixturePredictor(), THETA, CHOICES, None)
    with pytest.raises(ValueError, match=r"\(n_theta,\)"):
        OnsetGrid().for_theta(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="onset must be finite"):
        OnsetGrid().for_theta([0.4, np.nan])
    with pytest.raises(ValueError, match="no room for onset_window"):
        OnsetGrid(max_t=0.5).for_theta([0.1])


# --------------------------------------------------------------------------
# 4. OnsetGrid
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
    # t = 0 (the bottom of the ddm box) is clipped up by 1e-3, not to t_min
    # itself, so the pre/knee segments keep positive length.
    lo = grid.t_min + 1e-3
    assert np.any(rows[0] == lo) and not np.any(rows[0] == 0.0)
    assert np.sum(rows[0] < lo) == 32 + 128


def test_onset_grid_clips_the_onset_at_the_top():
    # With the default max_t = 20 an onset of 2 s needs no clipping; shrink
    # max_t so the same onset hits the upper clip and the tail keeps room.
    grid = OnsetGrid(max_t=2.5)
    hi = 2.5 - 1.0 - 1e-3
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
    # rule keeps the pre segment at least half the 1e-3 clip margin wide.
    grid = OnsetGrid()
    onsets = np.concatenate(
        [
            [grid.t_min + grid.knee + 1e-9, grid.t_min + grid.knee + 1e-14],
            np.linspace(0.0, 2.0, 2001),
        ]
    )
    rows = grid.for_theta(onsets)
    pre_width = rows[:, grid.n_pre] - rows[:, 0]
    assert np.all(pre_width >= 0.5e-3)
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


# --------------------------------------------------------------------------
# 5. A peaked density with a sharp onset
# --------------------------------------------------------------------------

# shape = 2 keeps the density continuous (C^0) at the onset (as in section 2)
# so the trapezoid rule keeps its O(h^2) rate; the derivative still jumps from
# 0 to 1/scale^2 there, so a grid without a point at the onset errs by an
# amount that depends on where the onset falls between two of its points.
# Scale 0.03 puts the whole peak inside a few tens of ms.
PEAK_SHAPE, PEAK_SCALE = 2.0, 0.03
ONSETS = np.array([[0.0], [0.4], [2.0]])


class PeakedPredictor:
    """Rows ``[t, rt, choice]``: a gamma peak starting at ``t``."""

    input_width = 3

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        t, rt, choice = np.asarray(batch, dtype=np.float64).T
        return log_density(choice, rt - t, PEAK_SHAPE, PEAK_SCALE)


def expected_peak(onsets=ONSETS[:, 0]):
    return sum(expected_mass(c, onsets, PEAK_SHAPE, PEAK_SCALE) for c in CHOICES)


@pytest.fixture(scope="module")
def peak_mass() -> ChoiceMass:
    return choice_mass(PeakedPredictor(), ONSETS, CHOICES, ONSETS[:, 0])


def test_onset_grid_is_accurate_at_every_onset_phase():
    # Pins the grid policy: a uniform 1000-point grid (20 ms apart) missed
    # this peak by 1e-4 to 0.036 depending on where the onset fell between
    # two of its points. Sweep the onset over one such step: the refined grid
    # stays within 1e-3 at every phase.
    onsets = 0.4 + 0.02 * np.arange(8) / 8
    mass = choice_mass(PeakedPredictor(), onsets[:, None], CHOICES, onsets)
    assert np.abs(mass.total - expected_peak(onsets)).max() < 1e-3


def test_result_carries_a_grid_per_theta_and_a_monotone_cdf(peak_mass):
    grid = OnsetGrid()
    assert peak_mass.t.shape == peak_mass.cdf.shape == (len(ONSETS), grid.n_points)
    assert peak_mass.by_choice.shape == (len(ONSETS), len(CHOICES))
    np.testing.assert_array_equal(peak_mass.t, grid.for_theta(ONSETS[:, 0]))
    assert np.all(peak_mass.cdf[:, 0] == 0.0)
    assert np.all(np.diff(peak_mass.cdf, axis=-1) >= 0.0)
    # A kept total must not keep the (n_theta, n_points) cdf alive.
    assert not np.shares_memory(peak_mass.total, peak_mass.cdf)
    for j, c in enumerate(CHOICES):
        expected = expected_mass(c, ONSETS[:, 0], PEAK_SHAPE, PEAK_SCALE)
        assert np.abs(peak_mass.by_choice[:, j] - expected).max() < 1e-3


class RecordingPredictor:
    """Wraps any predictor and records every batch it is handed."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.input_width = inner.input_width
        self.calls: list[np.ndarray] = []

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        self.calls.append(np.array(batch, copy=True))
        return self.inner(batch)


def test_chunks_agree_with_single_theta_runs(peak_mass, monkeypatch):
    monkeypatch.setattr(integrate_module, "_CHUNK", 2)
    spy = RecordingPredictor(PeakedPredictor())
    chunked = choice_mass(spy, ONSETS, CHOICES, ONSETS[:, 0])
    per_theta = len(CHOICES) * OnsetGrid().n_points
    assert [c.shape[0] for c in spy.calls] == [2 * per_theta, per_theta]
    np.testing.assert_array_equal(chunked.cdf, peak_mass.cdf)
    np.testing.assert_array_equal(chunked.by_choice, peak_mass.by_choice)
    for i in range(len(ONSETS)):  # a single (n_params,) vector is promoted
        single = choice_mass(PeakedPredictor(), ONSETS[i], CHOICES, ONSETS[i])
        np.testing.assert_array_equal(peak_mass.cdf[i], single.cdf[0])


def test_mass_before_hits_grid_points_and_clamps(peak_mass):
    t, cdf = peak_mass.t, peak_mass.cdf
    k = 32 + 128 + 40  # 40 points into the onset segment: the peak region
    np.testing.assert_allclose(peak_mass.mass_before(t[:, k]), cdf[:, k])
    midpoint = 0.5 * (t[:, k] + t[:, k + 1])
    np.testing.assert_allclose(
        peak_mass.mass_before(midpoint), 0.5 * (cdf[:, k] + cdf[:, k + 1])
    )
    # Per-row deadlines use each row's own grid, not a shared one.
    deadlines = ONSETS[:, 0] + np.array([0.01, 0.05, 0.2])
    got = peak_mass.mass_before(deadlines)
    for i, d in enumerate(deadlines):
        assert got[i] == pytest.approx(peak_mass.mass_before(d)[i])
    assert got[0] < got[1] < got[2]
    # The synthetic density is exactly zero at and below the onset; a deadline
    # past the onset does see mass (what survey's leak_below_onset relies on).
    onset = ONSETS[:, 0]
    np.testing.assert_array_equal(peak_mass.mass_before(onset), 0.0)
    assert np.all(peak_mass.mass_before(onset + 0.05) > 0.1)
    # Clamping at both ends.
    np.testing.assert_array_equal(peak_mass.mass_before(1e6), peak_mass.total)
    np.testing.assert_array_equal(peak_mass.mass_before(-1.0), 0.0)
    np.testing.assert_array_equal(peak_mass.mass_before(0.0), 0.0)


def test_quantile_round_trips_and_matches_scipy(peak_mass):
    t = peak_mass.t
    for u in (0.05, 0.5, 0.95):
        q = peak_mass.quantile(u)
        assert q.shape == (len(ONSETS),)
        assert np.all((q > t[:, 0]) & (q < t[:, -1]))
        np.testing.assert_allclose(
            peak_mass.mass_before(q), u * peak_mass.total, rtol=1e-9
        )
        # Each row's quantile sits on that row's own onset.
        expected = ONSETS[:, 0] + gamma.ppf(u, a=PEAK_SHAPE, scale=PEAK_SCALE)
        np.testing.assert_allclose(q, expected, atol=2e-3)


# --------------------------------------------------------------------------
# 6. survey
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
        a, t, rt, choice = np.asarray(batch, dtype=np.float64).T
        return log_density(choice, rt - t, 2.0, 0.2) + np.where(a > 2.0, np.log(1.2), 0)


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


def test_survey_leak_below_onset(scaled_survey):
    leak = scaled_survey["leak_below_onset"]
    # Exactly zero except for onsets below the grid clip (t < 1.1e-3), where
    # a sliver of the true density sits below the clipped onset: at most
    # ~1e-9 (one pre-segment interval of the trapezoid, spacing ~1.6e-5 s).
    assert leak["p99"] == 0.0 and leak["mean"] < 1e-9 and leak["max"] < 1e-9


class LeakyPredictor:
    """Rows ``[a, t, rt, choice]``: ScaledPredictor's gamma, unscaled, from t - 50 ms."""

    input_width = 4

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        _, t, rt, choice = np.asarray(batch, dtype=np.float64).T
        return log_density(choice, rt - t + 0.05, 2.0, 0.2)


def test_survey_leak_below_onset_measures_mass_before_t():
    r = survey(LeakyPredictor(), SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, n_theta=300)
    leak = r["leak_below_onset"]
    # Every onset past 50 ms leaks the gamma's first 50 ms; the few below it
    # (t < 0.05, the density cut at t_min) leak less.
    assert leak["p99"] == pytest.approx(gamma.cdf(0.05, a=2.0, scale=0.2), abs=1e-5)
    assert leak["mean"] < leak["p99"]


def test_survey_without_an_onset_builds_the_grid_around_zero():
    r = survey(
        ScaledPredictor(),
        SURVEY_BOUNDS,
        SURVEY_PARAMS,
        CHOICES,
        n_theta=300,
        onset_param=None,
    )
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
    a_bins = r["by_param"]["a"]
    assert a_bins[-1]["max_abs_dev"] == np.inf and a_bins[0]["max_abs_dev"] < 1e-3


def test_survey_validation():
    pred = ScaledPredictor()
    with pytest.raises(ValueError, match="at least one parameter"):
        survey(pred, SURVEY_BOUNDS, [], CHOICES, onset_param=None)
    with pytest.raises(ValueError, match="lo <= hi"):
        survey(pred, {**SURVEY_BOUNDS, "a": (2.5, 0.3)}, SURVEY_PARAMS, CHOICES)
    with pytest.raises(ValueError, match="onset_param 'tau'"):
        survey(pred, SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, onset_param="tau")
    with pytest.raises(ValueError, match="lacks bounds"):
        survey(pred, {"a": (0.3, 2.5)}, SURVEY_PARAMS, CHOICES)
    with pytest.raises(ValueError, match="n_theta"):
        survey(pred, SURVEY_BOUNDS, SURVEY_PARAMS, CHOICES, n_theta=0)
