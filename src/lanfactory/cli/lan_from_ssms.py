"""Train a bounded LAN recipe; never publish or submit a cluster job."""

import importlib
import json
from pathlib import Path

import typer

from lanfactory.recipes import train_lan_for_model

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = typer.Option(...),
    out: Path = typer.Option(...),
    preset: str = typer.Option("toy", help="toy, dev or explicitly authorized production"),
    model_config: str | None = typer.Option(
        None, help="Trusted module:attribute returning an ssms config"
    ),
    fixed_parameters: str = typer.Option(
        "{}", help="JSON mapping of fixed parameter values"
    ),
    seed: int = typer.Option(0),
    estimator: str = typer.Option("kde"),
    validate: bool = typer.Option(True, "--validate/--no-validate"),
    allow_production: bool = typer.Option(False),
):
    config = None
    if model_config:
        module, name = model_config.split(":", 1)
        config = getattr(importlib.import_module(module), name)
        config = config() if callable(config) else config
    artifact = train_lan_for_model(
        model,
        out,
        model_config=config,
        preset=preset,
        estimator=estimator,
        fixed_parameters=json.loads(fixed_parameters),
        seed=seed,
        validate=validate,
        allow_production=allow_production,
    )
    typer.echo(
        json.dumps(
            {
                "out": str(artifact.out_dir),
                "validated": artifact.validated,
                "gates": {k: v["passed"] for k, v in artifact.validation.items()},
            }
        )
    )
    if not artifact.validated:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
