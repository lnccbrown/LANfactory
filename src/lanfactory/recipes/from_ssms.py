"""One-call, bounded ssms -> PyTorch -> ONNX -> HSSM recipe."""

import hashlib
import importlib.metadata
import inspect
import json
import subprocess
import time
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import validation as gates
from .guards import GuardedSimulationPipeline, normalize_model_config


@dataclass(frozen=True)
class Preset:
    n_parameter_sets: int
    n_simulations: int
    n_samples_per_param: int
    n_files: int
    n_epochs: int
    validation_n: int


PRESETS = {
    "toy": Preset(32, 2000, 128, 2, 6, 8000),
    "dev": Preset(256, 10000, 512, 4, 40, 20000),
    "production": Preset(5000, 200000, 2000, 300, 100, 200000),
}


@dataclass
class LANArtifact:
    onnx_path: Path
    hssm_model_config: dict | None
    validation: dict
    manifest: dict
    out_dir: Path

    @property
    def validated(self):
        return all(
            self.validation.get(g, {}).get("passed") is True
            and not self.validation[g].get("skipped", False)
            for g in ("G1", "G2", "G3", "G4", "G5")
        )


def _describe(value):
    if isinstance(value, dict):
        return {str(k): _describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _describe(value.tolist())
    if isinstance(value, np.generic):
        return _describe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        try:
            source = inspect.getsource(value)
        except (TypeError, OSError):
            source = "unavailable"
        return {
            "callable": f"{value.__module__}.{value.__qualname__}",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "state": _describe(vars(value)),
    }


def _write(path, data):
    path.write_text(json.dumps(_describe(data), indent=2, allow_nan=False) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batch_divisor(n_rows, requested):
    if n_rows <= 0 or requested <= 0:
        raise ValueError("row count and batch size must be positive")
    actual = next(i for i in range(min(n_rows, requested), 0, -1) if n_rows % i == 0)
    if actual != requested:
        warnings.warn(
            f"batch_size {requested} does not divide {n_rows} rows; using {actual}",
            stacklevel=2,
        )
    return actual


def _source_revision():
    try:
        root = Path(__file__).resolve().parents[3]
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            )
        )
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def train_lan_for_model(
    model: str,
    out_dir: str | Path,
    *,
    model_config: dict | None = None,
    preset: str = "toy",
    estimator: str = "kde",
    network_config: dict | None = None,
    train_config: dict | None = None,
    generator_config: dict | None = None,
    fixed_parameters: dict | None = None,
    seed: int = 0,
    validate: bool = True,
    allow_production: bool = False,
) -> LANArtifact:
    """Train a LAN and return evidence, including failures.

    A failed/skipped/disabled mandatory gate prevents ``model_config.json``
    from being written and leaves ``hssm_model_config`` as None. The ONNX and
    manifest remain available for diagnosis. Toy/dev are never production
    certificates. ``fixed_parameters`` explicitly fixes a conditional design;
    those values are included as fixed HSSM priors, never fitted/extrapolated.

    ``generator_config`` overrides nested ssms settings. n_files is controlled
    by the preset, n_cpus is always one locally. Production requires explicit
    opt-in; the recipe does not submit or publish anything.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose {list(PRESETS)}")
    if preset == "production" and not allow_production:
        raise ValueError(
            "production requires allow_production=True and an explicitly authorized cluster run"
        )
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    settings = PRESETS[preset]
    import cloudpickle
    import torch
    from ssms.config import ModelConfigBuilder, get_lan_kde_config
    from ssms.dataset_generators.estimator_builders.kde_builder import (
        KDEEstimatorBuilder,
    )
    from ssms.dataset_generators.lan_mlp import TrainingDataGenerator
    from ssms.dataset_generators.strategies import ResampleMixtureStrategy

    from lanfactory.config.network_configs import network_config_mlp, train_config_mlp
    from lanfactory.trainers.torch_mlp import (
        ModelTrainerTorchMLP,
        TorchMLPFactory,
        make_train_valid_dataloaders,
    )

    cfg = normalize_model_config(
        ModelConfigBuilder.from_model(model) if model_config is None else model_config,
        fixed_parameters,
    )
    if cfg["name"] != model:
        raise ValueError("model and model_config name must agree")
    gen = get_lan_kde_config(model)
    gen["pipeline"].update(
        n_parameter_sets=settings.n_parameter_sets,
        n_subruns=1,
        n_cpus=1,
        parameter_sampler_index_offset=seed
        * settings.n_files
        * settings.n_parameter_sets,
    )
    gen["simulator"]["n_samples"] = settings.n_simulations
    gen["training"]["n_samples_per_param"] = settings.n_samples_per_param
    for section, overrides in (generator_config or {}).items():
        if section not in gen or not isinstance(overrides, dict):
            raise ValueError(f"unknown or non-nested generator override: {section}")
        gen[section].update(deepcopy(overrides))
    gen["pipeline"]["n_cpus"] = 1
    # Both stale spelling and unsafe displacement are excluded explicitly.
    gen["estimator"].pop("kde_displace_t", None)
    gen["estimator"].update(type=estimator, displace_t=False)
    if estimator not in ("kde", "pyddm"):
        raise ValueError("estimator must be kde or pyddm")
    if estimator == "pyddm" and (
        cfg.get("n_particles", 1) != 1
        or cfg["nchoices"] != 2
        or "python_simulator" in cfg.get("tags", [])
    ):
        raise ValueError(
            "PyDDM requires a single-particle two-choice diffusion; see ssms #443"
        )
    output = Path(out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("out_dir must be empty: never reuse a prior certificate")
    output.mkdir(parents=True, exist_ok=True)
    gen["output"]["folder"] = str(output / "training_data")
    if estimator == "kde":
        pipeline = GuardedSimulationPipeline(
            gen, cfg, KDEEstimatorBuilder, ResampleMixtureStrategy
        )
    else:
        from ssms.dataset_generators.pipelines.pipeline_factory import (
            create_data_generation_pipeline,
        )

        pipeline = create_data_generation_pipeline(gen, cfg)
    generator = TrainingDataGenerator(config=pipeline)
    start = time.monotonic()
    old_rng = np.random.get_state()
    np.random.seed(seed)
    files, row_counts = [], []
    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for i in range(settings.n_files):
                data = generator.generate_data_training(save=False)
                rows = data["lan_data"].shape[0]
                if (
                    rows == 0
                    or not np.isfinite(data["lan_data"]).all()
                    or not np.isfinite(data["lan_labels"]).all()
                ):
                    raise ValueError(
                        "generator returned empty or nonfinite training data"
                    )
                path = output / "training_data" / f"part-{i:03}.pickle"
                with path.open("wb") as stream:
                    cloudpickle.dump(data, stream)
                files.append(str(path))
                row_counts.append(rows)
            train = (
                deepcopy(train_config_mlp)
                | {"n_epochs": settings.n_epochs}
                | (train_config or {})
            )
            batch = batch_divisor(
                int(np.gcd.reduce(row_counts)), int(train["cpu_batch_size"])
            )
            train["cpu_batch_size"] = batch
            train_dl, valid_dl, width = make_train_valid_dataloaders(
                files,
                batch_size=batch,
                network_type="lan",
                num_workers=0,
                pin_memory=False,
                label_lower_bound=-16.11809565095832,
            )
            net_config = deepcopy(network_config_mlp) | (network_config or {})
            if (
                net_config.get("train_output_type") != "logprob"
                or net_config["layer_sizes"][-1] != 1
            ):
                raise ValueError("recipe requires one log-density output")
            net = TorchMLPFactory(net_config, input_dim=width, network_type="lan")
            trainer = ModelTrainerTorchMLP(
                net, train, train_dl, valid_dl, pin_memory=False, seed=seed
            )
            trainer.train_and_evaluate(
                output_folder=output,
                output_file_id=model,
                run_id=str(seed),
                mlflow_on=False,
                save_outputs=True,
                verbose=0,
            )
            path = output / f"{model}_lan_{seed}_model.onnx"
            net.eval()
            validation = {
                g: gates.result(False, skipped=True, reason="validation disabled")
                for g in ("G1", "G2", "G3", "G4", "G5")
            }
            if validate:
                validation["G1"] = gates.checked(
                    gates.gate_structure, path, len(cfg["params"]) + 2
                )
                if validation["G1"]["passed"]:
                    predict = gates.ONNXPredictor(path)
                    probe = data["lan_data"][:1000]
                    validation["G2"] = gates.checked(
                        gates.gate_parity, predict, net, probe
                    )
                    box = np.array(list(cfg["param_bounds_dict"].values()))
                    rng = np.random.default_rng(seed + 7000003)
                    points = box[:, 0] + (box[:, 1] - box[:, 0]) * rng.uniform(
                        0.2, 0.8, (5, len(box))
                    )
                    sim_args = {
                        "max_t": gen["simulator"]["max_t"],
                        "delta_t": gen["simulator"]["delta_t"],
                    }
                    fresh = gates.checked(
                        lambda: gates.simulate(
                            cfg, points[0], 256, seed + 8000003, **sim_args
                        )
                    )
                    # checked() returns either the simulation or a failed gate.
                    validation["G3"] = (
                        gates.checked(gates.gate_hssm, path, cfg, fresh)
                        if "rts" in fresh
                        else fresh
                    )
                    validation["G4"] = gates.checked(
                        gates.gate_density,
                        predict,
                        cfg,
                        points,
                        n_sim=settings.validation_n,
                        seed=seed + 9000003,
                        **sim_args,
                    )
                    validation["G5"] = gates.checked(
                        gates.gate_gradient,
                        path,
                        predict,
                        cfg,
                        points,
                        n_sim=settings.validation_n,
                        seed=seed + 10000003,
                        **sim_args,
                    )
                else:
                    for g in ("G2", "G3", "G4", "G5"):
                        validation[g] = gates.result(
                            False, skipped=True, reason="G1 failed"
                        )
    finally:
        np.random.set_state(old_rng)
    manifest = {
        "schema": "lanfactory-ssms-recipe-1",
        "created": datetime.now(UTC).isoformat(),
        "model": model,
        "preset": preset,
        "settings": asdict(settings),
        "seed": seed,
        "production_validated": False,
        "fixed_parameters": dict(fixed_parameters or {}),
        "model_config": _describe(cfg),
        "generator_config": gen,
        "train_config": train,
        "network_config": net_config,
        "source": _source_revision(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "lanfactory",
                "ssm-simulators",
                "torch",
                "numpy",
                "pandas",
                "onnx",
                "onnxruntime",
            )
        },
        "elapsed_seconds": time.monotonic() - start,
        "files": [
            {"name": Path(f).name, "sha256": _sha(f), "rows": n}
            for f, n in zip(files, row_counts, strict=True)
        ],
        "onnx": {"name": path.name, "sha256": _sha(path)},
        "filter_guard": {
            "accepted": getattr(pipeline, "accepted", None),
            "rejected": getattr(pipeline, "rejected", None),
            "policy": "strict canonical filters plus no omissions/horizon hits",
        },
        "validation": validation,
    }
    manifest["model_config_sha256"] = hashlib.sha256(
        json.dumps(manifest["model_config"], sort_keys=True).encode()
    ).hexdigest()
    for dependency in ("hssm", "jax", "jaxonnxruntime"):
        try:
            manifest["versions"][dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            manifest["versions"][dependency] = None
    artifact = LANArtifact(path, None, validation, manifest, output)
    if artifact.validated:
        artifact.hssm_model_config = gates.hssm_config(cfg)
        _write(output / "model_config.json", artifact.hssm_model_config)
    manifest["validated"] = artifact.validated
    _write(output / "validation.json", validation)
    _write(output / "manifest.json", manifest)
    return artifact
