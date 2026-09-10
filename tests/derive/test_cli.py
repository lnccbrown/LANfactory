"""CLI tests for ``derive-aux``."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from lanfactory.cli.derive_aux import app
from lanfactory.derive import MANIFEST_NAME

DDM_ONNX = Path(__file__).parent.parent / "fixtures" / "onnx" / "ddm.onnx"
_ANSI_RE = re.compile(r"\x1b\[[\d;]*m")

runner = CliRunner()


def _out(result) -> str:
    return _ANSI_RE.sub("", result.output)


def test_derive_aux_writes_files_and_manifest(tmp_path):
    out = tmp_path / "opn"
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(DDM_ONNX),
            "--model",
            "ddm",
            "--type",
            "opn",
            "--out",
            str(out),
            "--n-files",
            "3",
            "--n-theta-per-file",
            "16",
            "--grid-points",
            "200",
            "--seed",
            "7",
            "--source-hf-repo",
            "franklab/HSSM",
            "--source-hf-revision",
            "01f5d4d0fa9188940ab541a979f933550b29616a",
        ],
    )
    assert result.exit_code == 0, _out(result)
    assert str(out / MANIFEST_NAME) in _out(result)
    assert len(list(out.glob("*.pickle"))) == 3
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["n_files"] == 3 and manifest["n_theta_per_file"] == 16
    assert manifest["grid"]["n_points"] == 200 and manifest["seed"] == 7
    assert manifest["source"]["hf_repo"] == "franklab/HSSM"
    assert manifest["source"]["source_lan_hf_commit"].startswith("01f5d4d0")


def test_derive_aux_requires_type(tmp_path):
    result = runner.invoke(
        app,
        ["--from-onnx", str(DDM_ONNX), "--model", "ddm", "--out", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert "Missing option" in _out(result) and "--type" in _out(result)


def test_derive_aux_rejects_unknown_type(tmp_path):
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(DDM_ONNX),
            "--model",
            "ddm",
            "--type",
            "lan",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "--type must be one of" in _out(result)
    assert not list(tmp_path.glob("*.pickle"))


def test_derive_aux_rejects_missing_onnx_and_bad_counts(tmp_path):
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(tmp_path / "missing.onnx"),
            "--model",
            "ddm",
            "--type",
            "cpn",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    result = runner.invoke(
        app,
        [
            "--from-onnx",
            str(DDM_ONNX),
            "--model",
            "ddm",
            "--type",
            "cpn",
            "--out",
            str(tmp_path),
            "--n-files",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert "n_files must be >= 2" in _out(result)
