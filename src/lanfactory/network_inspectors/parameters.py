"""Parameter vectors and sweep ranges for the network inspector UI."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import ssms
from ssms.config import ModelConfigBuilder
from ssms.dataset_generators.parameter_samplers import UniformParameterSampler


def make_parameter_df(model: str, n_rows: int, seed: int) -> pd.DataFrame:
    """Sample parameter vectors within the model bounds.

    Mirrors how ssms samples training data: ``param_bounds_dict`` with the
    dependency-aware sampler (``full_ddm_rv`` bounds ``st`` by ``t``, which the
    positional ``param_bounds`` cannot express) plus the model's sampling
    transforms (e.g. LBA swaps ``b``/``A`` so ``b >= A``). Without them, the
    inspector would test networks on parameter sets they were never trained on.
    """
    config = ssms.config.model_config[model]
    sampler = UniformParameterSampler(
        param_space=config["param_bounds_dict"],
        constraints=ModelConfigBuilder.get_sampling_transforms(config),
    )
    samples = sampler.sample(n_samples=n_rows, rng=np.random.default_rng(seed))
    return pd.DataFrame({param: samples[param] for param in config["params"]})


def sweep_bounds(
    model: str, param: str, base_values: Mapping[str, float] | None = None
) -> tuple[float, float]:
    """Return numeric bounds for ``param``, resolving named endpoints.

    An endpoint may name another parameter (``full_ddm_rv``: ``st`` in
    ``(0.001, "t")``). It is resolved against ``base_values`` (the base
    parameter vector the sweep runs from), falling back to the model defaults.
    """
    config = ssms.config.model_config[model]
    values = dict(zip(config["params"], config["default_params"], strict=True))
    if base_values is not None:
        values.update(base_values)
    lower, upper = config["param_bounds_dict"][param]

    def resolve(endpoint: float | str) -> float:
        return float(values[endpoint] if isinstance(endpoint, str) else endpoint)

    return resolve(lower), resolve(upper)
