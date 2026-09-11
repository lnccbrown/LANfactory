#!/usr/bin/env python
"""Command-line interface for deriving auxiliary-network corpora from a LAN.

Integrates a trained LAN's density over reaction time and writes a cpn, opn or
gonogo training corpus in the layout ``torchtrain`` / ``jaxtrain`` consume.

Usage:
    derive-aux --from-onnx ddm.onnx --network-type cpn --model-name ddm \\
        --output-folder data/cpn/ddm

The option names follow the Hub CLIs (``--network-type``, ``--model-name``,
``--output-folder``); ``--type``, ``--model`` and ``--out`` are accepted as
short aliases.
"""

from pathlib import Path

import typer
from ssms.config import ModelConfigBuilder

from lanfactory.derive import (
    MANIFEST_NAME,
    NETWORK_TYPES,
    IntegrationGrid,
    OnsetGrid,
    SourceLAN,
    derive_aux_corpus,
)

app = typer.Typer()


@app.command()
def main(
    from_onnx: Path = typer.Option(
        ...,
        "--from-onnx",
        help="The trained LAN, a (1, n_params + 2) ONNX artifact.",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    network_type: str = typer.Option(
        ...,
        "--network-type",
        "--type",
        help=f"Auxiliary network type to derive: one of {', '.join(NETWORK_TYPES)}.",
    ),
    model_name: str = typer.Option(
        ...,
        "--model-name",
        "--model",
        help="ssms model name the LAN was trained for (e.g., ddm, angle).",
    ),
    output_folder: Path = typer.Option(
        ...,
        "--output-folder",
        "--out",
        help="Destination folder for the pickles and the manifest.",
        file_okay=False,
        resolve_path=True,
    ),
    n_files: int = typer.Option(
        100, "--n-files", help="Number of training files to write (at least 2)."
    ),
    n_theta_per_file: int = typer.Option(
        4096,
        "--n-theta-per-file",
        help="Parameter vectors per file (rows per file for opn/gonogo; "
        "times the number of choices for cpn).",
    ),
    onset_param: str = typer.Option(
        "t",
        "--onset-param",
        help="Non-decision-time parameter the integration grid is refined around. "
        "Pass an empty string to integrate on a uniform grid instead.",
    ),
    grid_points: int = typer.Option(
        1000,
        "--grid-points",
        help="Points of the uniform grid; used only when no onset grid applies "
        "(--onset-param '' or a model without that parameter).",
    ),
    max_t: float = typer.Option(
        20.0, "--max-t", help="Upper edge of the integration grid in seconds."
    ),
    deadline_quantile_frac: float = typer.Option(
        0.7,
        "--deadline-quantile-frac",
        help="Share of opn/gonogo deadlines drawn from the LAN's own RT quantiles; "
        "the rest are uniform on the deadline bounds.",
    ),
    fallback_window: tuple[float, float] = typer.Option(
        (0.98, 1.03),
        "--fallback-window",
        help="LO HI: a parameter vector whose total LAN mass is outside this open "
        "interval is labelled by ssms simulation instead of the LAN.",
    ),
    fallback_n_sim: int = typer.Option(
        20_000,
        "--fallback-n-sim",
        help="Trials simulated per parameter vector that falls back.",
    ),
    no_fallback: bool = typer.Option(
        False,
        "--no-fallback",
        help="Label every parameter vector from the LAN, whatever its total.",
    ),
    seed: int = typer.Option(0, "--seed", help="Base random seed."),
    source_run_uuid: str = typer.Option(
        None,
        "--source-run-uuid",
        help="Training run uuid of the LAN (parsed from the filename when absent).",
    ),
    source_hf_repo: str = typer.Option(
        None, "--source-hf-repo", help="Hub repository the LAN was downloaded from."
    ),
    source_hf_revision: str = typer.Option(
        None,
        "--source-hf-revision",
        help="Hub revision (commit, tag, or branch) of that download.",
    ),
):
    """Derive a cpn / opn / gonogo training corpus from a trained LAN.

    Writes ``--n-files`` pickles plus ``derive_manifest.json`` to
    ``--output-folder`` and prints the manifest path. Train on the folder with
    ``torchtrain`` or ``jaxtrain`` using a batch size that divides the rows
    per file.

    Example:
        derive-aux --from-onnx ddm.onnx --network-type cpn --model-name ddm
        --output-folder data/cpn/ddm
    """
    if network_type not in NETWORK_TYPES:
        raise typer.BadParameter(
            f"--network-type must be one of {list(NETWORK_TYPES)}, got: {network_type}"
        )
    try:
        params = list(ModelConfigBuilder.from_model(model_name)["params"])
    except Exception as e:  # noqa: BLE001 - ssms raises several types here
        raise typer.BadParameter(f"--model-name {model_name!r}: {e}") from e
    onset = onset_param or None
    try:
        if onset is not None and onset in params:
            grid: IntegrationGrid | OnsetGrid = OnsetGrid(max_t=max_t)
        else:
            if onset is not None:
                typer.echo(
                    f"Warning: model {model_name!r} has no parameter {onset!r}; "
                    f"integrating on a uniform {grid_points}-point grid.",
                    err=True,
                )
            grid = IntegrationGrid(n_points=grid_points, max_t=max_t)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    source = SourceLAN.from_onnx(
        from_onnx,
        run_uuid=source_run_uuid,
        hf_repo=source_hf_repo,
        hf_revision=source_hf_revision,
    )
    try:
        files = derive_aux_corpus(
            from_onnx,
            model_name,
            network_type,
            output_folder,
            n_files=n_files,
            n_theta_per_file=n_theta_per_file,
            onset_param=onset,
            grid=grid,
            deadline_quantile_frac=deadline_quantile_frac,
            fallback_window=None if no_fallback else fallback_window,
            fallback_n_sim=fallback_n_sim,
            seed=seed,
            source=source,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    typer.echo(
        f"Wrote {len(files)} {network_type} files for {model_name} to {output_folder}"
    )
    typer.echo(f"Manifest: {output_folder / MANIFEST_NAME}")


if __name__ == "__main__":
    app()
