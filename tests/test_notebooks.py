"""Execute the tutorial notebooks so they cannot silently drift from the code.

These tests are skipped by default. Run them with:
    uv run pytest tests/test_notebooks.py --run-notebooks
    uv run pytest tests/test_notebooks.py --run-notebooks -k lan_torch

Two tutorial formats are covered:
- The Jupyter basic tutorials under ``docs/basic_tutorial`` are executed with
  ``jupyter nbconvert --execute``.
- The marimo export tutorials under ``notebooks`` are executed by exporting them
  to an ipynb (``marimo export ipynb``), which runs the whole notebook.

Every notebook is executed inside a throwaway working directory so the training
data / model artifacts they generate never land in the repository.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
BASIC_TUTORIAL_DIR = PROJECT_ROOT / "docs" / "basic_tutorial"
MARIMO_DIR = PROJECT_ROOT / "notebooks"

# Timeout for a single notebook (seconds). The tutorials generate a small amount
# of training data and train for a few epochs, so this is generous.
NOTEBOOK_TIMEOUT = 1200

BASIC_NOTEBOOKS = sorted(BASIC_TUTORIAL_DIR.glob("*.ipynb"))
MARIMO_NOTEBOOKS = [
    MARIMO_DIR / "exporting_sbi_to_onnx.py",
    MARIMO_DIR / "exporting_bayesflow_to_onnx.py",
]


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
            return False, f"timed out after {NOTEBOOK_TIMEOUT} seconds"
    return (
        result.returncode == 0,
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
    )


@pytest.mark.notebooks
@pytest.mark.parametrize(
    "notebook_path", BASIC_NOTEBOOKS, ids=[nb.stem for nb in BASIC_NOTEBOOKS]
)
def test_basic_tutorial_executes(notebook_path: Path):
    """Execute a basic-tutorial Jupyter notebook via nbconvert."""
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
