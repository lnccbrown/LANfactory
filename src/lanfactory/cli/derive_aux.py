#!/usr/bin/env python
"""Command-line interface for deriving auxiliary-network corpora from a LAN.

Integrates a trained LAN's density over reaction time and writes a cpn, opn or
gonogo training corpus in the layout ``torchtrain`` / ``jaxtrain`` consume.

Usage:
    derive-aux --from-onnx ddm.onnx --model ddm --type cpn --out data/cpn/ddm
"""

from pathlib import Path

import typer

from lanfactory.derive import (
    NETWORK_TYPES,
    IntegrationGrid,
    SourceLAN,
    derive_aux_corpus,
)
from lanfactory.derive.corpus import MANIFEST_NAME

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
    model: str = typer.Option(
        ...,
        "--model",
        help="ssms model name the LAN was trained for (e.g., ddm, angle).",
    ),
    network_type: str = typer.Option(
        ...,
        "--type",
        help=f"Auxiliary network type: one of {', '.join(NETWORK_TYPES)}.",
    ),
    out: Path = typer.Option(
        ...,
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
    grid_points: int = typer.Option(
        1000, "--grid-points", help="Reaction-time grid points per choice."
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

    Writes ``--n-files`` pickles plus ``derive_manifest.json`` to ``--out`` and
    prints the manifest path. Train on the folder with ``torchtrain`` or
    ``jaxtrain`` using a batch size that divides the rows per file.

    Example:
        derive-aux --from-onnx ddm.onnx --model ddm --type cpn --out data/cpn/ddm
    """
    if network_type not in NETWORK_TYPES:
        raise typer.BadParameter(
            f"--type must be one of {list(NETWORK_TYPES)}, got: {network_type}"
        )
    try:
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
            model,
            network_type,
            out,
            n_files=n_files,
            n_theta_per_file=n_theta_per_file,
            grid=grid,
            deadline_quantile_frac=deadline_quantile_frac,
            seed=seed,
            source=source,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    typer.echo(f"Wrote {len(files)} {network_type} files for {model} to {out}")
    typer.echo(f"Manifest: {out / MANIFEST_NAME}")


if __name__ == "__main__":
    app()
