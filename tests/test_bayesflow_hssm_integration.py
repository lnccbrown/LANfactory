"""End-to-end pipeline: bayesflow training → ONNX export → HSSM MCMC.

Mirrors ``tests/test_sbi_hssm_integration.py`` so the bayesflow path gets the
same cross-repo coverage as the sbi path.

1. Train a tiny bayesflow ``ContinuousApproximator`` (NLE) on synthetic DDM
   data via ssm-simulators.
2. Export via ``lanfactory.onnx.transform_bayesflow_to_onnx``.
3. Build an HSSM model with ``loglik_kind="approx_differentiable"`` and
   ``loglik=<path>``.
4. Run a short MCMC and verify posterior means recover the ground truth
   (within ±2σ) and r_hat is reasonable.

Skip guard: requires HSSM importable in the test environment. LANfactory's
own CI does not currently install HSSM, so this test is a no-op locally —
it's intended for the coordinated cross-repo CI matrix.
"""

# KERAS_BACKEND must be set BEFORE any keras / bayesflow import.
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

hssm = pytest.importorskip("hssm")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bayesflow as bf  # noqa: E402
import keras  # noqa: E402
from bayesflow.datasets import OfflineDataset  # noqa: E402
from bayesflow.networks.inference.coupling.transforms import AffineTransform  # noqa: E402
from ssms.basic_simulators.simulator import simulator  # noqa: E402

from lanfactory.onnx import transform_bayesflow_to_onnx  # noqa: E402

_DDM_PARAM_NAMES = ["v", "a", "z", "t"]
_DDM_PARAM_LOW = np.array([-2.0, 0.6, 0.3, 0.1], dtype=np.float32)
_DDM_PARAM_HIGH = np.array([2.0, 1.8, 0.7, 0.5], dtype=np.float32)
_TRUE_THETA = np.array([0.5, 1.2, 0.5, 0.25], dtype=np.float32)
_N_OBS = 300
_N_TRAIN = 5000


def _simulate_ddm_rows(theta: np.ndarray) -> np.ndarray:
    """Simulate one (rt, choice) per row of theta. Returns (n, 2) float32."""
    rts = np.empty(theta.shape[0], dtype=np.float32)
    choices = np.empty(theta.shape[0], dtype=np.float32)
    for i, th in enumerate(theta):
        out = simulator(theta=th[None, :], model="ddm", n_samples=1)
        rts[i] = out["rts"].squeeze()
        choices[i] = out["choices"].squeeze()
    return np.stack([rts, choices], axis=-1)


def _build_observed_dataframe() -> pd.DataFrame:
    """Generate N_OBS trials at the true theta as an HSSM-shaped DataFrame."""
    out = simulator(theta=_TRUE_THETA[None, :], model="ddm", n_samples=_N_OBS)
    rts = out["rts"].squeeze().astype(np.float32)
    choices = out["choices"].squeeze().astype(np.float32)
    return pd.DataFrame({"rt": rts, "response": choices})


@pytest.fixture(scope="module")
def trained_bayesflow_for_ddm(tmp_path_factory) -> Path:
    """Train tiny bayesflow CouplingFlow NLE on DDM and return the .onnx path."""
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)

    theta = rng.uniform(
        _DDM_PARAM_LOW, _DDM_PARAM_HIGH, size=(_N_TRAIN, len(_DDM_PARAM_NAMES))
    ).astype(np.float32)
    x = _simulate_ddm_rows(theta).astype(np.float32)

    approximator = bf.ContinuousApproximator(
        inference_network=bf.networks.CouplingFlow(
            depth=4,
            subnet_kwargs={
                "widths": (64, 64),
                "activation": "silu",   # see lanfactory.onnx.bayesflow on hard_silu
                "dropout": None,
            },
            permutation=None,           # avoids aten::ravel in ONNX trace
            use_actnorm=False,
            transform=AffineTransform(clamp=False),
        ),
        standardize="inference_variables",  # standardize x (rt, choice)
    )
    approximator.build({
        "inference_variables": (None, 2),
        "inference_conditions": (None, len(_DDM_PARAM_NAMES)),
    })
    approximator.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4))

    dataset = OfflineDataset(
        data={"inference_variables": x, "inference_conditions": theta},
        batch_size=200,
        adapter=None,
    )
    approximator.fit(dataset=dataset, epochs=30, verbose=0)

    onnx_path = tmp_path_factory.mktemp("bf_ddm") / "ddm_nle.onnx"
    transform_bayesflow_to_onnx(
        approximator,
        str(onnx_path),
        mode="nle",
        example_theta_dim=len(_DDM_PARAM_NAMES),
        example_x_dim=2,
    )
    return onnx_path


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_hssm_model_builds_from_bayesflow_onnx(trained_bayesflow_for_ddm: Path) -> None:
    """Exported ONNX loads cleanly into an HSSM DDM model."""
    obs_data = _build_observed_dataframe()
    model = hssm.HSSM(
        data=obs_data,
        model="ddm",
        loglik_kind="approx_differentiable",
        loglik=str(trained_bayesflow_for_ddm),
        p_outlier=0,
    )
    assert model is not None


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_hssm_mcmc_recovers_ddm_parameters_via_bayesflow(
    trained_bayesflow_for_ddm: Path,
) -> None:
    """Short MCMC should recover the true DDM params within ±2σ."""
    obs_data = _build_observed_dataframe()
    model = hssm.HSSM(
        data=obs_data,
        model="ddm",
        loglik_kind="approx_differentiable",
        loglik=str(trained_bayesflow_for_ddm),
        p_outlier=0,
    )
    idata = model.sample(
        draws=500,
        tune=500,
        chains=2,
        cores=1,
        progressbar=False,
        target_accept=0.9,
    )

    if hasattr(hssm.utils, "summary"):
        summary = hssm.utils.summary(idata)
    else:
        import arviz as az
        summary = az.summary(idata, var_names=_DDM_PARAM_NAMES)

    posterior_means = summary.loc[_DDM_PARAM_NAMES, "mean"].to_numpy()
    posterior_sds = summary.loc[_DDM_PARAM_NAMES, "sd"].to_numpy()
    r_hats = summary.loc[_DDM_PARAM_NAMES, "r_hat"].to_numpy()

    assert (r_hats < 1.05).all(), f"r_hat above 1.05 for some params: {r_hats}"

    deviations = np.abs(posterior_means - _TRUE_THETA) / posterior_sds
    assert (deviations < 2.0).all(), (
        f"Posterior means more than 2σ from truth: "
        f"true={_TRUE_THETA}, mean={posterior_means}, sd={posterior_sds}, "
        f"deviations={deviations}"
    )
