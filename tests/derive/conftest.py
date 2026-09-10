"""Shared fixtures for the derive tests: the production ddm LAN fixture."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "onnx"
DDM_ONNX = FIXTURE_DIR / "ddm.onnx"


@pytest.fixture(scope="session")
def ddm_provenance() -> dict[str, str]:
    """The ``ddm.onnx`` table of ``PROVENANCE.md``, so the doc is the source of truth.

    Keys are the table's row labels (``Source``, ``Hub commit``, ``sha256``,
    ...); code-span values are returned without their backticks.
    """
    doc = (FIXTURE_DIR / "PROVENANCE.md").read_text()
    table = re.search(r"^## `ddm\.onnx`\n(?P<body>.*?)(?=^## |\Z)", doc, re.M | re.S)
    assert table is not None, "PROVENANCE.md has no ddm.onnx section"
    fields: dict[str, str] = {}
    for line in table.group("body").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] and not set(cells[0]) <= {"-"}:
            fields[cells[0]] = cells[1].strip("`")
    return fields
