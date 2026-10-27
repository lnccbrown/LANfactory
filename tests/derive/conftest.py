"""Shared constants for the derive tests: the production ddm LAN fixture."""

from __future__ import annotations

from pathlib import Path

# franklab/HSSM's root ddm.onnx, copied unchanged (see fixtures/onnx/PROVENANCE.md).
DDM_ONNX = Path(__file__).parent.parent / "fixtures" / "onnx" / "ddm.onnx"
DDM_SHA256 = "09f685c18d3bdbd9b54fa89bed3b5bc0e2c76566e7ed0ae5e24df2c9b04f1b0e"
HUB_COMMIT = "01f5d4d0fa9188940ab541a979f933550b29616a"
