"""CLI tests for ``derive-aux``."""

from __future__ import annotations

import json
import re
from importlib.metadata import entry_points

from typer.testing import CliRunner

from lanfactory.cli.derive_aux import app
from lanfactory.derive import MANIFEST_NAME, derive_aux_corpus
from tests._onnx_utils import DDM_ONNX, HUB_COMMIT

_ANSI_RE = re.compile(r"\x1b\[[\d;]*m")
RUN_UUID = "56d99936415e11f0a2bf3cecefb6d5ee"

runner = CliRunner()


def _out(result) -> str:
    return _ANSI_RE.sub("", result.output)


def test_derive_aux_is_installed_as_a_console_script():
    """The ``[project.scripts]`` entry point resolves to this app."""
    (entry,) = entry_points(group="console_scripts", name="derive-aux")
    assert entry.load() is app


def test_derive_aux_writes_what_the_library_writes(tmp_path):
    """Every option reaches ``derive_aux_corpus``: the CLI output equals the API's.

    Seed, run uuid and Hub revision all land in the pickles, so byte-identical
    pickles and equal manifests prove each option was passed through.
    """
    out = tmp_path / "opn"
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(DDM_ONNX),
            "--network-type",
            "opn",
            "--model-name",
            "ddm",
            "--output-folder",
            str(out),
            "--n-files",
            "3",
            "--n-theta-per-file",
            "16",
            "--seed",
            "7",
            "--source-run-uuid",
            RUN_UUID,
            "--source-hf-revision",
            HUB_COMMIT,
        ],
    )
    assert result.exit_code == 0, _out(result)
    assert str(out / MANIFEST_NAME) in _out(result)
    assert "Wrote 3 opn files for ddm" in _out(result)

    library = derive_aux_corpus(
        DDM_ONNX,
        "ddm",
        "opn",
        tmp_path / "lib",
        n_files=3,
        n_theta_per_file=16,
        seed=7,
        source_run_uuid=RUN_UUID,
        source_hf_revision=HUB_COMMIT,
    )
    cli_files = sorted(out.glob("*.pickle"))
    assert [f.name for f in cli_files] == [f.name for f in library]
    for cli_file, lib_file in zip(cli_files, library, strict=True):
        assert cli_file.read_bytes() == lib_file.read_bytes()
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest == json.loads((tmp_path / "lib" / MANIFEST_NAME).read_text())
    assert manifest["source"]["source_lan_run_uuid"] == RUN_UUID


def test_derive_aux_turns_a_library_error_into_a_usage_error(tmp_path):
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(DDM_ONNX),
            "--network-type",
            "lan",
            "--model-name",
            "ddm",
            "--output-folder",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "network_type must be one of" in _out(result)
    assert not list(tmp_path.glob("*.pickle"))
