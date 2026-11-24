"""Local guards for ssms 0.16's filter and runtime-config contracts.

No process-wide patches: the recipe injects this pipeline into the public
TrainingDataGenerator. Remove the filter compatibility code once #432 ships;
the regression cases should remain.
"""

from copy import deepcopy

import numpy as np
from ssms.dataset_generators.pipelines.simulation_pipeline import SimulationPipeline


def normalize_model_config(config, fixed_parameters=None):
    """Copy and validate numeric training bounds in the simulator's param order."""
    cfg = deepcopy(config)
    params = cfg["params"]
    if not params or len(set(params)) != len(params):
        raise ValueError("params must be nonempty and unique")
    bounds = cfg.get("param_bounds")
    if bounds is None:
        bounds = cfg.get("param_bounds_dict")
    if not isinstance(bounds, dict):
        if (
            bounds is None
            or len(bounds) != 2
            or any(len(row) != len(params) for row in bounds)
        ):
            raise ValueError("param_bounds must have exactly two rows in params order")
        bounds = dict(zip(params, zip(*bounds, strict=True), strict=True))
    if set(bounds) != set(params):
        raise ValueError("training bounds must name exactly the simulator parameters")
    fixed = dict(fixed_parameters or {})
    if set(fixed) - set(params):
        raise ValueError("fixed_parameters names unknown parameters")
    normalized = {}
    for p in params:
        try:
            lo, hi = map(float, bounds[p])
        except (TypeError, ValueError) as err:
            raise ValueError(f"{p}: recipe requires finite numeric bounds") from err
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError(f"{p}: bounds must be finite with lower < upper")
        if p in fixed:
            value = float(fixed[p])
            if not lo <= value <= hi:
                raise ValueError(f"{p}: fixed value outside model bounds")
            lo = hi = value
        normalized[p] = (lo, hi)
    if len(fixed) == len(params):
        raise ValueError("at least one parameter must be free")
    cfg["param_bounds_dict"] = normalized
    cfg["param_bounds"] = [[normalized[p][i] for p in params] for i in (0, 1)]
    cfg["n_params"] = len(params)
    choices = cfg["choices"]
    if len(choices) < 2 or len(set(choices)) != len(choices):
        raise ValueError("recipe requires at least two distinct RT/choice outcomes")
    cfg["nchoices"] = len(choices)
    return cfg


def simulation_issues(simulations, max_t):
    """Reject censored, clamped, nonfinite or nonpositive training samples.

    This conservative test treats max_t as a bound on decision time and uses
    min(t,tT) for two-onset models. It also rejects RTs beyond the horizon for
    kernels that ignore max_t. It does not resolve ssms #404 globally.
    """
    rt = np.asarray(simulations["rts"]).ravel()
    meta = simulations["metadata"]
    onset = (
        min(float(np.min(meta.get("t", 0))), float(np.min(meta["tT"])))
        if "tT" in meta
        else float(np.min(meta.get("t", 0)))
    )
    choices = np.asarray(simulations["choices"]).ravel()
    return {
        "invalid_rt": int(np.count_nonzero(~np.isfinite(rt) | (rt <= 0))),
        "at_or_past_horizon": int(np.count_nonzero(rt >= max_t + onset - 1e-6)),
        "invalid_choice": int(
            np.count_nonzero(~np.isin(choices, meta["possible_choices"]))
        ),
        "n_samples": int(rt.size),
    }


class GuardedSimulationPipeline(SimulationPipeline):
    """Apply the canonical strict filters before fitting any KDE."""

    def _is_valid_simulation(self, simulations):
        filters = self.generator_config["pipeline"]["simulation_filters"]
        issues = simulation_issues(
            simulations, self.generator_config["simulator"]["max_t"]
        )
        keep = issues["n_samples"] > 0 and not any(
            issues[k] for k in ("invalid_rt", "at_or_past_horizon", "invalid_choice")
        )
        rt = np.asarray(simulations["rts"]).ravel()
        choice = np.asarray(simulations["choices"]).ravel()
        for c in simulations["metadata"]["possible_choices"]:
            values = rt[choice == c]
            if not values.size:
                keep = False
                continue
            unique, counts = np.unique(values, return_counts=True)
            mode = float(unique[np.argmax(counts)])
            keep = keep and (
                mode < filters.get("mode", np.inf)
                and float(values.mean()) <= filters.get("mean_rt", np.inf)
                and float(values.std()) > filters.get("std", -np.inf)
                and float(counts.max() / values.size) <= filters.get("mode_cnt_rel", 1)
                and values.size > filters.get("choice_cnt", -np.inf)
            )
        self.rejected = getattr(self, "rejected", 0) + int(not keep)
        self.accepted = getattr(self, "accepted", 0) + int(keep)
        self.consecutive_rejections = (
            0 if keep else getattr(self, "consecutive_rejections", 0) + 1
        )
        if self.consecutive_rejections > 100:
            raise RuntimeError(
                "100 consecutive rejected simulations: narrow the training box or extend max_t"
            )
        return bool(keep), np.array(list(issues.values()), dtype=np.float32)
