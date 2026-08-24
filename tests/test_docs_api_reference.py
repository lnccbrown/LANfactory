"""Keep public namespace exports represented in the rendered API reference."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _all_exports(module_path: str) -> set[str]:
    tree = ast.parse((ROOT / module_path).read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        return set(ast.literal_eval(node.value))
    raise AssertionError(f"No literal __all__ found in {module_path}")


def _assert_documented(module_path: str, docs_path: str) -> None:
    reference = (ROOT / docs_path).read_text()
    missing = sorted(
        name
        for name in _all_exports(module_path)
        if f"`{name}`" not in reference and f".{name}" not in reference
    )
    assert not missing, f"{docs_path} omits public exports: {missing}"


def test_config_exports_are_referenced() -> None:
    _assert_documented("src/lanfactory/config/__init__.py", "docs/api/config.md")


def test_hf_exports_are_referenced() -> None:
    _assert_documented("src/lanfactory/hf/__init__.py", "docs/api/hf.md")


def test_network_inspector_exports_are_referenced() -> None:
    _assert_documented(
        "src/lanfactory/network_inspectors/__init__.py",
        "docs/api/network_inspectors.md",
    )
