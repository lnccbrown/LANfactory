"""Execute the tutorial notebooks so they cannot silently drift from the code.

These tests are skipped by default. Run them with:
    uv run pytest tests/test_notebooks.py --run-notebooks
    uv run pytest tests/test_notebooks.py --run-notebooks -k lan_torch

Two tutorial surfaces are covered:
- Every committed rendered notebook is executed with ``jupyter nbconvert``.
- Every canonical marimo source under ``notebooks`` is executed by exporting it
  to ipynb (``marimo export ipynb``), which runs the whole notebook.

Every notebook is executed inside a throwaway working directory so the training
data / model artifacts they generate never land in the repository.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
BASIC_TUTORIAL_DIR = PROJECT_ROOT / "docs" / "basic_tutorial"
EXPORTED_TUTORIAL_DIR = PROJECT_ROOT / "docs" / "tutorials"
MARIMO_DIR = PROJECT_ROOT / "notebooks"

# Timeout for a single notebook (seconds). The tutorials generate a small amount
# of training data and train for a few epochs, so this is generous.
NOTEBOOK_TIMEOUT = 1200

BASIC_NOTEBOOKS = sorted(BASIC_TUTORIAL_DIR.glob("*.ipynb"))
EXPORTED_NOTEBOOKS = sorted(EXPORTED_TUTORIAL_DIR.glob("*.ipynb"))
# Fail loudly if discovery finds nothing (e.g. the dir was moved/renamed) rather
# than parametrizing over an empty set and silently skipping the whole suite.
assert BASIC_NOTEBOOKS, (
    f"No basic-tutorial notebooks discovered in {BASIC_TUTORIAL_DIR}"
)
assert EXPORTED_NOTEBOOKS, (
    f"No exported tutorial notebooks discovered in {EXPORTED_TUTORIAL_DIR}"
)
RENDERED_NOTEBOOKS = BASIC_NOTEBOOKS + EXPORTED_NOTEBOOKS
MARIMO_EXPORTS = {
    MARIMO_DIR / "basic_tutorial_lan_jax.py": BASIC_TUTORIAL_DIR
    / "basic_tutorial_lan_jax.ipynb",
    MARIMO_DIR / "exporting_sbi_to_onnx.py": EXPORTED_TUTORIAL_DIR
    / "exporting_sbi_to_onnx.ipynb",
    MARIMO_DIR / "exporting_bayesflow_to_onnx.py": EXPORTED_TUTORIAL_DIR
    / "exporting_bayesflow_to_onnx.ipynb",
}
MARIMO_NOTEBOOKS = list(MARIMO_EXPORTS)
assert all(nb.exists() for nb in MARIMO_NOTEBOOKS), (
    f"Missing marimo tutorial source(s): "
    f"{[str(nb) for nb in MARIMO_NOTEBOOKS if not nb.exists()]}"
)

LOCAL_OUTPUT_PATH = re.compile(
    r"(?:/Users/|/home/|/private/var/folders/|/var/folders/|[A-Za-z]:[\\\\/]Users[\\\\/])"
)


def _run(cmd: list[str]) -> tuple[bool, str]:
    """Run ``cmd`` in a throwaway working directory; return (success, output)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=NOTEBOOK_TIMEOUT + 60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"timed out after {NOTEBOOK_TIMEOUT + 60} seconds"
    return (
        result.returncode == 0,
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
    )


@pytest.mark.parametrize(
    "notebook_path",
    RENDERED_NOTEBOOKS,
    ids=[nb.stem for nb in RENDERED_NOTEBOOKS],
)
def test_committed_notebook_outputs_are_portable(notebook_path: Path):
    """Keep terminal progress noise and machine-local paths out of public output."""
    notebook = json.loads(notebook_path.read_text())
    for cell_index, cell in enumerate(notebook["cells"]):
        for output_index, output in enumerate(cell.get("outputs", [])):
            if output.get("output_type") != "stream":
                continue
            text = "".join(output.get("text", []))
            assert "\r" not in text, (
                f"{notebook_path}: cell {cell_index}, output {output_index} "
                "contains carriage-return progress output"
            )
            assert LOCAL_OUTPUT_PATH.search(text) is None, (
                f"{notebook_path}: cell {cell_index}, output {output_index} "
                "contains a machine-local path"
            )


@pytest.mark.notebooks
@pytest.mark.parametrize(
    "notebook_path",
    RENDERED_NOTEBOOKS,
    ids=[nb.stem for nb in RENDERED_NOTEBOOKS],
)
def test_rendered_tutorial_executes(notebook_path: Path):
    """Execute the exact committed documentation notebook via nbconvert."""
    success, output = _run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--stdout",
            f"--ExecutePreprocessor.timeout={NOTEBOOK_TIMEOUT}",
            str(notebook_path.resolve()),
        ]
    )
    if not success:
        pytest.fail(f"{notebook_path.name} failed to execute:\n{output[-4000:]}")


@pytest.mark.notebooks
@pytest.mark.parametrize(
    "notebook_path", MARIMO_NOTEBOOKS, ids=[nb.stem for nb in MARIMO_NOTEBOOKS]
)
def test_marimo_tutorial_executes(notebook_path: Path):
    """Execute a marimo export tutorial by exporting it to ipynb (runs the notebook)."""
    success, output = _run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "ipynb",
            str(notebook_path.resolve()),
            "-o",
            "exported.ipynb",
            "--include-outputs",
        ]
    )
    if not success:
        pytest.fail(f"{notebook_path.name} failed to execute:\n{output[-4000:]}")


@pytest.mark.notebooks
@pytest.mark.parametrize(
    ("source_path", "rendered_path"),
    MARIMO_EXPORTS.items(),
    ids=[path.stem for path in MARIMO_EXPORTS],
)
def test_marimo_source_matches_rendered_notebook(
    source_path: Path,
    rendered_path: Path,
):
    """Keep the rendered notebook's cells synchronized with its marimo source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exported_path = Path(tmpdir) / "exported.ipynb"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "ipynb",
                str(source_path.resolve()),
                "-o",
                str(exported_path),
            ],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-4000:]
        exported = json.loads(exported_path.read_text())

    rendered = json.loads(rendered_path.read_text())

    def cell_contract(notebook: dict) -> list[tuple[str, list[str]]]:
        return [(cell["cell_type"], cell["source"]) for cell in notebook["cells"]]

    assert cell_contract(rendered) == cell_contract(exported), (
        f"{rendered_path} is stale; regenerate it from {source_path}"
    )
