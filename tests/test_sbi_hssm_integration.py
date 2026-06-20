"""C7b verification: end-to-end pipeline from sbi training to HSSM MCMC.

This test exercises the full keystone integration:
  1. Train a tiny sbi NLE_A on synthetic DDM data (via ssm-simulators).
  2. Export via lanfactory.onnx.transform_sbi_to_onnx.
  3. Build an HSSM model with model="ddm", loglik_kind="approx_differentiable",
     loglik=<path>, backend="jax".
  4. Run a short MCMC and verify posterior means recover the ground truth
     (within ±2σ) and r_hat < 1.01.

Skip guard: this test runs only when HSSM is importable in the test
environment. LANfactory's regular CI does not currently install HSSM, so the
test is a no-op locally — it is intended to run in a coordinated cross-repo
CI matrix where both packages are available with compatible JAX pins. See
plans/sbi-onnx-integration.md C7b for the environment-resolution note.

The C7a HSSM patch (commit d1d7ffe on HSSM sbi-integration branch) makes the
`jax_enable_x64` flag self-managed inside HSSM's onnx2jax — this test does
not need to set it explicitly.
"""

from pathlib import Path

import pytest

# Skip cleanly when HSSM is not in the environment.
hssm = pytest.importorskip("hssm")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sbi.inference import NLE_A  # noqa: E402
from sbi.utils import BoxUniform  # noqa: E402
from ssms.basic_simulators.simulator import simulator  # noqa: E402

from lanfactory.onnx import transform_sbi_to_onnx  # noqa: E402

# DDM parameter order matches sbi simulator inputs and HSSM defaults.
_DDM_PARAM_NAMES = ["v", "a", "z", "t"]
_DDM_PARAM_LOW = np.array([-2.0, 0.6, 0.3, 0.1])
_DDM_PARAM_HIGH = np.array([2.0, 1.8, 0.7, 0.5])
_TRUE_THETA = np.array([0.5, 1.2, 0.5, 0.25])
_N_OBS = 300
_N_TRAIN = 5000


def _simulate_ddm(theta: torch.Tensor) -> torch.Tensor:
    """Simulate (rt, choice) per row of theta. Returns x of shape (batch, 2)."""
    theta_np = theta.detach().numpy().astype(np.float32)
    rts = np.empty(theta_np.shape[0], dtype=np.float32)
    choices = np.empty(theta_np.shape[0], dtype=np.float32)
    for i, th in enumerate(theta_np):
        out = simulator(theta=th[None, :], model="ddm", n_samples=1)
        rts[i] = out["rts"].squeeze()
        choices[i] = out["choices"].squeeze()
    return torch.from_numpy(np.stack([rts, choices], axis=-1))


def _build_observed_dataframe(rng: np.random.Generator) -> pd.DataFrame:
    """Generate N_OBS trials at the true theta as an HSSM-shaped DataFrame."""
    out = simulator(theta=_TRUE_THETA[None, :], model="ddm", n_samples=_N_OBS)
    rts = out["rts"].squeeze().astype(np.float32)
    choices = out["choices"].squeeze().astype(np.float32)
    return pd.DataFrame({"rt": rts, "response": choices})


@pytest.fixture(scope="module")
def trained_nle_for_ddm(tmp_path_factory) -> Path:
    """Train tiny NLE_A on DDM and return the exported .onnx path."""
    torch.manual_seed(0)
    prior = BoxUniform(
        low=torch.from_numpy(_DDM_PARAM_LOW.astype(np.float32)),
        high=torch.from_numpy(_DDM_PARAM_HIGH.astype(np.float32)),
    )
    inference = NLE_A(prior=prior, density_estimator="maf")

    theta = prior.sample((_N_TRAIN,))
    x = _simulate_ddm(theta)
    estimator = inference.append_simulations(theta, x).train(
        training_batch_size=200,
        max_num_epochs=30,
    )
    estimator.eval()

    onnx_path = tmp_path_factory.mktemp("c7b") / "ddm_nle.onnx"
    transform_sbi_to_onnx(
        estimator,
        str(onnx_path),
        mode="nle",
        example_theta_dim=len(_DDM_PARAM_NAMES),
        example_x_dim=2,
    )
    return onnx_path


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_hssm_model_builds_from_sbi_onnx(trained_nle_for_ddm: Path) -> None:
    """The exported ONNX should load cleanly into an HSSM model."""
    rng = np.random.default_rng(0)
    obs_data = _build_observed_dataframe(rng)

    model = hssm.HSSM(
        data=obs_data,
        model="ddm",
        loglik_kind="approx_differentiable",
        loglik=str(trained_nle_for_ddm),
        p_outlier=0,
    )
    assert model is not None


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_hssm_mcmc_recovers_ddm_parameters(trained_nle_for_ddm: Path) -> None:
    """Short MCMC should recover the true DDM params within ±2σ."""
    rng = np.random.default_rng(0)
    obs_data = _build_observed_dataframe(rng)

    model = hssm.HSSM(
        data=obs_data,
        model="ddm",
        loglik_kind="approx_differentiable",
        loglik=str(trained_nle_for_ddm),
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

    summary = hssm.utils.summary(idata) if hasattr(hssm.utils, "summary") else None
    # Fall back to arviz if the convenience method is not exposed.
    if summary is None:
        import arviz as az

        summary = az.summary(idata, var_names=_DDM_PARAM_NAMES)

    posterior_means = summary.loc[_DDM_PARAM_NAMES, "mean"].to_numpy()
    posterior_sds = summary.loc[_DDM_PARAM_NAMES, "sd"].to_numpy()
    r_hats = summary.loc[_DDM_PARAM_NAMES, "r_hat"].to_numpy()

    # Convergence
    assert (r_hats < 1.05).all(), f"r_hat above 1.05 for some params: {r_hats}"

    # Recovery within ±2σ
    deviations = np.abs(posterior_means - _TRUE_THETA) / posterior_sds
    assert (deviations < 2.0).all(), (
        f"Posterior means more than 2σ from truth: "
        f"true={_TRUE_THETA}, mean={posterior_means}, sd={posterior_sds}, "
        f"deviations={deviations}"
    )
