"""Parameter vectors and sweep ranges for the network inspector UI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ssms
from ssms.dataset_generators.parameter_samplers import UniformParameterSampler


def make_parameter_df(model: str, n_rows: int, seed: int) -> pd.DataFrame:
    """Sample parameter vectors within the model bounds.

    Uses ``param_bounds_dict`` with ssms's dependency-aware sampler because some
    models bound one parameter by another (e.g. ``full_ddm_rv`` has
    ``st <= t``), which the positional ``param_bounds`` cannot express.
    """
    config = ssms.config.model_config[model]
    sampler = UniformParameterSampler(
        param_space=config["param_bounds_dict"], constraints=[]
    )
    samples = sampler.sample(n_samples=n_rows, rng=np.random.default_rng(seed))
    return pd.DataFrame({param: samples[param] for param in config["params"]})


def sweep_bounds(model: str, param: str) -> tuple[float, float]:
    """Return numeric bounds for ``param``, resolving named endpoints.

    An endpoint may name another parameter (``full_ddm_rv``: ``st`` in
    ``(0.001, "t")``); it is resolved against the model's default vector.
    """
    config = ssms.config.model_config[model]
    defaults = dict(zip(config["params"], config["default_params"], strict=True))
    lower, upper = config["param_bounds_dict"][param]

    def resolve(endpoint: float | str) -> float:
        return float(defaults[endpoint] if isinstance(endpoint, str) else endpoint)

    return resolve(lower), resolve(upper)
