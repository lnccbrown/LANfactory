"""Small, fail-closed LAN checks; toy runs may legitimately fail them.

G5 tests both implementation derivatives and sensitivity of independently
simulated densities. Finite differences of the network alone cannot validate
the latter. Production recovery/calibration remains a separate cluster job.
"""

import importlib.util

import numpy as np

from .guards import simulation_issues


def result(passed, **details):
    return {"passed": bool(passed), **details}


def checked(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as err:
        return result(False, error=f"{type(err).__name__}: {err}")


class ONNXPredictor:
    """Evaluate a concrete single-trial graph without changing its semantics."""

    def __init__(self, path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]

    def __call__(self, rows):
        rows = np.asarray(rows, dtype=np.float32)
        return np.array(
            [
                np.asarray(
                    self.session.run(
                        None, {self.input.name: row.reshape(self.input.shape)}
                    )[0]
                ).item()
                for row in rows
            ]
        )


def gate_structure(path, width):
    from lanfactory.onnx.contract import assert_single_trial_contract

    metadata = assert_single_trial_contract(path, expected_input_width=width)
    predictor = ONNXPredictor(path)
    if (
        len(predictor.session.get_inputs()) != 1
        or len(predictor.session.get_outputs()) != 1
    ):
        return result(False, error="expected exactly one input and output")
    value = predictor(np.zeros((1, width)))
    return result(np.isfinite(value).all(), **metadata)


def gate_parity(predict, network, rows, atol=1e-4):
    import torch

    network.eval()
    with torch.no_grad():
        expected = (
            network(
                torch.tensor(
                    rows, dtype=torch.float32, device=next(network.parameters()).device
                )
            )
            .cpu()
            .numpy()
            .ravel()
        )
    observed = predict(rows)
    error = float(np.max(np.abs(expected - observed)))
    return result(
        np.isfinite(observed).all() and error <= atol,
        max_abs_error=error,
        atol=atol,
        n_rows=len(rows),
    )


def hssm_config(config):
    bounds = config["param_bounds_dict"]
    fixed = {p: lo for p, (lo, hi) in bounds.items() if lo == hi}
    out = {
        "list_params": list(config["params"]),
        "choices": list(config["choices"]),
        "bounds": {p: list(b) for p, b in bounds.items() if p not in fixed},
        "default_priors": fixed,
        "backend": "jax",
        "response": ["rt", "response"],
    }
    if config.get("ndt_edge_shift") is not None:
        out["ndt_edge_shift"] = config["ndt_edge_shift"]
    return out


def simulate(config, theta, n, seed, max_t, delta_t):
    from ssms import Simulator

    out = Simulator(config).simulate(
        dict(zip(config["params"], theta, strict=True)),
        n_samples=n,
        random_state=int(seed),
        max_t=max_t,
        delta_t=delta_t,
    )
    issues = simulation_issues(out, max_t)
    if any(issues[k] for k in ("invalid_rt", "at_or_past_horizon", "invalid_choice")):
        raise ValueError(f"fresh simulation has invalid/horizon observations: {issues}")
    if set(np.asarray(out["choices"]).ravel()) != set(config["choices"]):
        raise ValueError("fresh simulation is missing a possible choice")
    return out


def make_kde(simulations):
    from ssms.dataset_generators.likelihood_estimators.kde_estimator import (
        KDELikelihoodEstimator,
    )

    estimator = KDELikelihoodEstimator(displace_t=False)
    estimator.fit(simulations)
    return estimator


def gate_hssm(path, config, simulations):
    if importlib.util.find_spec("hssm") is None:
        return result(False, skipped=True, reason="hssm is not installed")
    import hssm
    import pandas as pd

    data = pd.DataFrame(
        {
            "rt": np.asarray(simulations["rts"]).ravel()[:64],
            "response": np.asarray(simulations["choices"]).ravel()[:64],
        }
    )
    model = hssm.HSSM(
        data,
        model=config["name"],
        loglik=str(path),
        loglik_kind="approx_differentiable",
        model_config=hssm_config(config),
        p_outlier=0.0,
    )
    pymc_model = model.pymc_model
    value = float(pymc_model.compile_logp()(pymc_model.initial_point()))
    return result(np.isfinite(value), initial_logp=value, n_trials=len(data))


def hellinger(a, b, grid):
    return float(
        np.sqrt(max(0, np.trapezoid((np.sqrt(a) - np.sqrt(b)) ** 2, grid) / 2))
    )


def gate_density(
    predict,
    config,
    points,
    *,
    n_sim,
    seed,
    max_t,
    delta_t,
    mass_range=(0.9, 1.1),
    max_hellinger=0.15,
    max_ratio=3.0,
):
    records = []
    for i, theta in enumerate(points):
        draws = [
            simulate(config, theta, n_sim, seed + 1009 * i + j, max_t, delta_t)
            for j in (0, 1)
        ]
        kdes = [make_kde(draw) for draw in draws]
        # Dense where observations occur, plus the entire remaining horizon.
        upper = min(
            max_t, float(np.quantile(np.asarray(draws[0]["rts"]), 0.9995)) * 1.1
        )
        grid = np.unique(
            np.r_[np.linspace(1e-5, upper, 400), np.linspace(upper, max_t, 80)]
        )
        mass, distance2, floor2 = 0.0, 0.0, 0.0
        per_choice = []
        for c in config["choices"]:
            choices = np.full(grid.size, c)
            reference, replicate = [np.exp(kde.evaluate(grid, choices)) for kde in kdes]
            if not np.isfinite(reference).all() or not np.isfinite(replicate).all():
                return result(False, error="nonfinite reference density")
            rows = np.column_stack([np.tile(theta, (grid.size, 1)), grid, choices])
            logp = predict(rows)
            if not np.isfinite(logp).all() or np.max(logp) > 80:
                return result(False, error="nonfinite or overflowing network density")
            density = np.exp(logp)
            m = float(np.trapezoid(density, grid))
            h = hellinger(density, reference, grid)
            floor = hellinger(reference, replicate, grid)
            mass += m
            distance2 += h**2
            floor2 += floor**2
            per_choice.append({"choice": int(c), "mass": m, "hellinger": h})
        h, floor = np.sqrt(distance2), np.sqrt(floor2)
        ratio = float(h / max(floor, 1e-6))
        # Both absolute and noise-relative errors must be small.
        passed = (
            mass_range[0] <= mass <= mass_range[1]
            and h <= max_hellinger
            and ratio <= max_ratio
        )
        records.append(
            {
                "theta": list(map(float, theta)),
                "mass": mass,
                "hellinger": float(h),
                "sampling_floor": float(floor),
                "ratio": ratio,
                "passed": bool(passed),
                "choices": per_choice,
            }
        )
    return result(
        all(row["passed"] for row in records),
        points=records,
        n_sim=n_sim,
        mass_range=list(mass_range),
        max_hellinger=max_hellinger,
        max_ratio=max_ratio,
    )


def gate_gradient(
    path,
    predict,
    config,
    points,
    *,
    n_sim,
    seed,
    max_t,
    delta_t,
    autodiff_atol=0.01,
    fidelity_atol=0.5,
    fidelity_rtol=0.35,
    max_reference_noise=0.5,
):
    if importlib.util.find_spec("hssm") is None:
        return result(False, skipped=True, reason="hssm is not installed")
    import jax
    import jax.numpy as jnp
    from hssm.distribution_utils.onnx import make_jax_logp_funcs_from_onnx

    logp, _ = make_jax_logp_funcs_from_onnx(
        path, [False] * len(config["params"]), return_jit=False
    )
    gradients = jax.jit(jax.jacrev(lambda theta, obs: logp(obs, *theta), argnums=0))
    spans = np.array([hi - lo for lo, hi in config["param_bounds_dict"].values()])
    free = np.flatnonzero(spans > 0)
    records = []
    for i, theta in enumerate(points):
        draws = simulate(config, theta, n_sim, seed + i * 1009, max_t, delta_t)
        rts, choices = [np.asarray(draws[k]).ravel() for k in ("rts", "choices")]
        obs = np.array(
            [
                [np.quantile(rts[choices == c], q), c]
                for c in config["choices"]
                for q in (0.35, 0.65)
            ]
        )
        auto = np.asarray(gradients(jnp.asarray(theta), jnp.asarray(obs))).reshape(
            len(obs), -1
        )
        own, reference, noise = [], [], []
        for p in free:
            tiny = max(1e-3, spans[p] * 1e-3)
            plus, minus = np.array(theta), np.array(theta)
            plus[p] += tiny
            minus[p] -= tiny
            own.append(
                (
                    predict(np.column_stack([np.tile(plus, (len(obs), 1)), obs]))
                    - predict(np.column_stack([np.tile(minus, (len(obs), 1)), obs]))
                )
                / (2 * tiny)
                * spans[p]
            )
            step = 0.04 * spans[p]
            plus[p], minus[p] = theta[p] + step, theta[p] - step
            replicas = []
            for j in (0, 1):
                # Common random numbers at +/- step; a fresh independent pair
                # measures the uncertainty of the reference derivative.
                pair_seed = seed + 100_003 + i * 1009 + j
                pdfs = [
                    make_kde(
                        simulate(config, point, n_sim, pair_seed, max_t, delta_t)
                    ).evaluate(obs[:, 0], obs[:, 1])
                    for point in (plus, minus)
                ]
                replicas.append((pdfs[0] - pdfs[1]) / (2 * step) * spans[p])
            reference.append(np.mean(replicas, axis=0))
            noise.append(np.abs(replicas[0] - replicas[1]) / 2)
        own, reference, noise = [np.asarray(a).T for a in (own, reference, noise)]
        scaled_auto = auto[:, free] * spans[free]
        implementation_error = float(np.max(np.abs(own - scaled_auto)))
        reference_noise = float(np.sqrt(np.mean(noise**2)))
        error = float(np.sqrt(np.mean((scaled_auto - reference) ** 2)))
        tolerance = float(
            fidelity_atol + fidelity_rtol * np.sqrt(np.mean(reference**2))
        )
        finite = np.isfinite(
            np.r_[scaled_auto.ravel(), own.ravel(), reference.ravel(), noise.ravel()]
        ).all()
        passed = (
            finite
            and implementation_error <= autodiff_atol
            and reference_noise <= max_reference_noise
            and error <= tolerance
        )
        records.append(
            {
                "theta": list(map(float, theta)),
                "passed": bool(passed),
                "autodiff_max_error": implementation_error,
                "reference_rms_error": error,
                "reference_rms_noise": reference_noise,
                "fidelity_tolerance": tolerance,
                "parameters": [config["params"][p] for p in free],
                "autodiff_scaled": scaled_auto.tolist(),
                "reference_scaled": reference.tolist(),
            }
        )
    return result(
        all(row["passed"] for row in records),
        points=records,
        n_sim=n_sim,
        autodiff_atol=autodiff_atol,
        max_reference_noise=max_reference_noise,
        fidelity_atol=fidelity_atol,
        fidelity_rtol=fidelity_rtol,
        reference="paired-seed finite differences of fresh log-RT KDE; two independent pairs",
    )
