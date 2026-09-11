"""CLI tests for ``derive-aux``."""

from __future__ import annotations

import json
import re
import subprocess

from typer.testing import CliRunner

from lanfactory.cli.derive_aux import app
from lanfactory.derive import (
    MANIFEST_NAME,
    IntegrationGrid,
    OnsetGrid,
    SourceLAN,
    derive_aux_corpus,
    grid_description,
)
from tests.derive.conftest import DDM_ONNX

_ANSI_RE = re.compile(r"\x1b\[[\d;]*m")
RUN_UUID = "56d99936415e11f0a2bf3cecefb6d5ee"

runner = CliRunner()


def _out(result) -> str:
    return _ANSI_RE.sub("", result.output)


def test_derive_aux_is_installed_as_a_console_script():
    """The ``[project.scripts]`` entry point resolves (the tests below run in-process)."""
    result = subprocess.run(
        ["derive-aux", "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    help_text = _ANSI_RE.sub("", result.stdout)
    for flag in ("--from-onnx", "--network-type", "--model-name", "--output-folder"):
        assert flag in help_text


def test_derive_aux_writes_files_and_manifest(tmp_path, ddm_provenance):
    """Every option reaches ``derive_aux_corpus``: the CLI output equals the API's.

    The pickles omit the source path, so a CLI run and a library call with the
    same arguments are byte-identical file by file.
    """
    out = tmp_path / "opn"
    hub_commit = ddm_provenance["Hub commit"]
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
            "--grid-points",
            "200",
            "--max-t",
            "15",
            "--deadline-quantile-frac",
            "0.3",
            "--fallback-window",
            "0.9",
            "1.1",
            "--fallback-n-sim",
            "200",
            "--seed",
            "7",
            "--source-run-uuid",
            RUN_UUID,
            "--source-hf-repo",
            "franklab/HSSM",
            "--source-hf-revision",
            hub_commit,
        ],
    )
    assert result.exit_code == 0, _out(result)
    assert str(out / MANIFEST_NAME) in _out(result)
    assert "Wrote 3 opn files for ddm" in _out(result)
    assert len(list(out.glob("*.pickle"))) == 3

    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["model"] == "ddm" and manifest["network_type"] == "opn"
    assert manifest["n_files"] == 3 and manifest["n_theta_per_file"] == 16
    assert manifest["grid"] == grid_description(OnsetGrid(max_t=15.0), "t")
    assert manifest["grid"]["kind"] == "onset" and manifest["grid"]["max_t"] == 15.0
    assert manifest["deadline_quantile_frac"] == 0.3
    assert manifest["fallback_window"] == [0.9, 1.1]
    assert manifest["fallback_n_sim"] == 200
    assert manifest["seed"] == 7
    source = manifest["source"]
    assert source["hf_repo"] == "franklab/HSSM"
    assert source["hf_revision"] == source["source_lan_hf_commit"] == hub_commit
    assert source["run_uuid"] == source["source_lan_run_uuid"] == RUN_UUID
    assert source["sha256"] == source["source_lan_sha256"] == ddm_provenance["sha256"]

    library = derive_aux_corpus(
        DDM_ONNX,
        "ddm",
        "opn",
        tmp_path / "lib",
        n_files=3,
        n_theta_per_file=16,
        grid=OnsetGrid(max_t=15.0),
        deadline_quantile_frac=0.3,
        fallback_window=(0.9, 1.1),
        fallback_n_sim=200,
        seed=7,
        source=SourceLAN.from_onnx(
            DDM_ONNX,
            run_uuid=RUN_UUID,
            hf_repo="franklab/HSSM",
            hf_revision=hub_commit,
        ),
    )
    # The pickles carry the flat provenance (uuid, sha256, Hub commit) but not
    # the source path, so the two runs are byte-identical file by file.
    for cli_file, lib_file in zip(sorted(out.glob("*.pickle")), library, strict=True):
        assert cli_file.name == lib_file.name
        assert cli_file.read_bytes() == lib_file.read_bytes()


def test_derive_aux_short_aliases_match_the_long_options(tmp_path):
    """``--type`` / ``--model`` / ``--out`` are aliases, not separate options."""
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(DDM_ONNX),
            "--type",
            "cpn",
            "--model",
            "ddm",
            "--out",
            str(tmp_path / "cpn"),
            "--n-files",
            "2",
            "--n-theta-per-file",
            "8",
        ],
    )
    assert result.exit_code == 0, _out(result)
    manifest = json.loads((tmp_path / "cpn" / MANIFEST_NAME).read_text())
    assert manifest["network_type"] == "cpn" and manifest["model"] == "ddm"
    assert manifest["n_rows_per_file"] == 16  # 8 thetas x 2 choices


def test_derive_aux_requires_type(tmp_path):
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(DDM_ONNX),
            "--model-name",
            "ddm",
            "--output-folder",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "Missing option" in _out(result) and "--network-type" in _out(result)


def test_derive_aux_rejects_unknown_type(tmp_path):
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
    assert "--network-type must be one of" in _out(result)
    assert not list(tmp_path.glob("*.pickle"))


def test_derive_aux_rejects_missing_onnx_and_bad_counts(tmp_path):
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(tmp_path / "missing.onnx"),
            "--network-type",
            "cpn",
            "--model-name",
            "ddm",
            "--output-folder",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(DDM_ONNX),
            "--network-type",
            "cpn",
            "--model-name",
            "ddm",
            "--output-folder",
            str(tmp_path),
            "--n-files",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert "n_files must be >= 2" in _out(result)


def test_derive_aux_empty_onset_param_selects_the_uniform_grid(tmp_path):
    """``--onset-param ''`` integrates on ``--grid-points`` uniform points."""
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(DDM_ONNX),
            "--network-type",
            "cpn",
            "--model-name",
            "ddm",
            "--output-folder",
            str(tmp_path / "uniform"),
            "--n-files",
            "2",
            "--n-theta-per-file",
            "8",
            "--onset-param",
            "",
            "--grid-points",
            "200",
        ],
    )
    assert result.exit_code == 0, _out(result)
    manifest = json.loads((tmp_path / "uniform" / MANIFEST_NAME).read_text())
    assert manifest["grid"] == grid_description(IntegrationGrid(n_points=200), None)
    assert manifest["grid"]["kind"] == "uniform"


def test_derive_aux_warns_and_uses_the_uniform_grid_without_the_onset_param(
    tmp_path,
):
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(DDM_ONNX),
            "--network-type",
            "cpn",
            "--model-name",
            "ddm",
            "--output-folder",
            str(tmp_path / "missing"),
            "--n-files",
            "2",
            "--n-theta-per-file",
            "8",
            "--onset-param",
            "ndt",
        ],
    )
    assert result.exit_code == 0, _out(result)
    assert "no parameter 'ndt'" in _out(result)
    manifest = json.loads((tmp_path / "missing" / MANIFEST_NAME).read_text())
    assert manifest["grid"]["kind"] == "uniform"
    assert manifest["grid"]["onset_param"] is None


def test_derive_aux_no_fallback_disables_the_window(tmp_path):
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(DDM_ONNX),
            "--network-type",
            "cpn",
            "--model-name",
            "ddm",
            "--output-folder",
            str(tmp_path / "raw"),
            "--n-files",
            "2",
            "--n-theta-per-file",
            "8",
            "--no-fallback",
        ],
    )
    assert result.exit_code == 0, _out(result)
    manifest = json.loads((tmp_path / "raw" / MANIFEST_NAME).read_text())
    assert manifest["fallback_window"] is None
    assert manifest["derive_stats"]["derive_fallback_frac"] == 0.0


def test_derive_aux_rejects_a_bad_fallback_window(tmp_path):
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(DDM_ONNX),
            "--network-type",
            "cpn",
            "--model-name",
            "ddm",
            "--output-folder",
            str(tmp_path),
            "--fallback-window",
            "1.1",
            "0.9",
        ],
    )
    assert result.exit_code == 2
    assert "fallback_window must satisfy" in _out(result)
    assert not list(tmp_path.glob("*.pickle"))
