"""Locate trained LAN artifacts on disk for the network inspectors.

Finds per-model directories under a base folder (flat ``<base>/<model>/`` or
torchtrain's ``<base>/<network_type>/<model>/``) and pairs each state dict with
the network config written by the same training run.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import ssms


def available_models(base_dir: str) -> list[str]:
    """Return the sorted names of models with loadable artifacts."""
    return sorted(model_directories(base_dir))


# torchtrain writes ``<base>/<network_type>/<model>/``; flat ``<base>/<model>/``
# layouts are also supported, so two levels below the base is enough. A small
# margin keeps the scan cheap even when the base dir is something like ``~``.
_MAX_SCAN_DEPTH = 3
# Network types whose artifacts share model names with LANs but are not
# loadable as LANs (different input dimension / output head).
_NON_LAN_NETWORK_TYPES = frozenset({"cpn", "opn", "gonogo"})
# torchtrain embeds a ``uuid.uuid1().hex`` run id in every artifact filename.
_RUN_UUID_PATTERN = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")


def _iter_directories(root: Path, max_depth: int) -> Iterator[Path]:
    """Yield subdirectories of ``root`` up to ``max_depth`` levels deep."""
    frontier = [root]
    for _ in range(max_depth):
        next_frontier: list[Path] = []
        for directory in frontier:
            try:
                children = [child for child in directory.iterdir() if child.is_dir()]
            except OSError:
                continue
            next_frontier.extend(children)
        yield from next_frontier
        frontier = next_frontier


def model_directories(base_dir: str) -> dict[str, Path]:
    """Map each detected model name to the directory holding its artifacts."""
    model_root = Path(base_dir).expanduser()
    if not model_root.exists() or not model_root.is_dir():
        return {}

    valid_ssms_models = set(ssms.config.model_config.keys())
    candidates: dict[str, list[Path]] = {}
    for model_dir in sorted(_iter_directories(model_root, _MAX_SCAN_DEPTH)):
        if model_dir.name not in valid_ssms_models:
            continue
        relative_parents = model_dir.relative_to(model_root).parts[:-1]
        if _NON_LAN_NETWORK_TYPES.intersection(relative_parents):
            continue
        has_state = any(model_dir.glob("*state_dict*"))
        has_cfg = any(model_dir.glob("*network_config*"))
        if has_state and has_cfg:
            candidates.setdefault(model_dir.name, []).append(model_dir)

    # Prefer a directory whose artifacts pair by run; otherwise keep the first
    # candidate so resolution reports why it cannot be paired.
    return {
        model: next(
            (model_dir for model_dir in dirs if _is_pairable(model_dir)), dirs[0]
        )
        for model, dirs in candidates.items()
    }


def _is_pairable(model_dir: Path) -> bool:
    try:
        pair_run_artifacts(model_dir)
    except FileNotFoundError:
        return False
    return True


def _run_uuid(path: Path) -> str | None:
    match = _RUN_UUID_PATTERN.search(path.name)
    return match.group(0) if match else None


def pair_run_artifacts(model_dir: Path) -> tuple[Path, Path]:
    """Return the (state dict, network config) pair from the same training run.

    Artifacts are matched by the run UUID torchtrain embeds in both filenames;
    when several runs match, the most recently written checkpoint wins. Files
    without a run UUID are accepted only when they are the sole candidates.
    """
    state_candidates = sorted(model_dir.glob("*state_dict*"))
    config_candidates = sorted(model_dir.glob("*network_config*"))

    if not state_candidates:
        raise FileNotFoundError(f"No state dict file found in {model_dir}.")
    if not config_candidates:
        raise FileNotFoundError(f"No network config file found in {model_dir}.")

    configs_by_uuid = {
        run_uuid: config
        for config in config_candidates
        if (run_uuid := _run_uuid(config)) is not None
    }
    matched = [
        (state, configs_by_uuid[run_uuid])
        for state in state_candidates
        if (run_uuid := _run_uuid(state)) in configs_by_uuid
    ]
    if matched:
        return max(matched, key=lambda pair: pair[0].stat().st_mtime)

    if (
        len(state_candidates) == 1
        and len(config_candidates) == 1
        and _run_uuid(state_candidates[0]) is None
        and _run_uuid(config_candidates[0]) is None
    ):
        return state_candidates[0], config_candidates[0]

    raise FileNotFoundError(
        f"Could not match a state dict to a network config from the same "
        f"training run in {model_dir}. Keep exactly one pair of artifacts, or "
        "use torchtrain's run-UUID filenames so they can be paired."
    )


def resolve_model_paths(base_dir: str, model: str) -> tuple[Path, Path]:
    """Return the (state dict, network config) pair to load for ``model``."""
    model_dir = model_directories(base_dir).get(model)
    if model_dir is None:
        available = available_models(base_dir)
        available_str = ", ".join(available) if available else "none detected"
        raise FileNotFoundError(
            f"Model directory not found for '{model}' under {base_dir}. "
            "Set the base directory to your torch_models folder. "
            f"Detected models in this base dir: {available_str}."
        )

    return pair_run_artifacts(model_dir)
