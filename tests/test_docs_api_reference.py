"""Keep public namespace exports represented in the rendered API reference."""

from __future__ import annotations

import ast
import re
import tomllib
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


def _literal_assignment(module_path: str, name: str) -> object:
    tree = ast.parse((ROOT / module_path).read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"No literal assignment for {name} in {module_path}")


def _cli_option_flags(module_path: str) -> set[str]:
    tree = ast.parse((ROOT / module_path).read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    arguments = [*main.args.posonlyargs, *main.args.args]
    defaults = [None] * (len(arguments) - len(main.args.defaults)) + list(
        main.args.defaults
    )
    arguments.extend(main.args.kwonlyargs)
    defaults.extend(main.args.kw_defaults)

    flags: set[str] = set()
    for argument, default in zip(arguments, defaults, strict=True):
        if not isinstance(default, ast.Call):
            continue
        is_typer_option = (
            isinstance(default.func, ast.Attribute) and default.func.attr == "Option"
        ) or (
            isinstance(default.func, ast.Name)
            and default.func.id == "option_no_default"
        )
        if not is_typer_option:
            continue
        explicit = {
            flag
            for arg in default.args
            if isinstance(arg, ast.Constant)
            and isinstance((value := arg.value), str)
            and value.startswith("-")
            for flag in value.split("/")
        }
        flags.update(explicit or {f"--{argument.arg.replace('_', '-')}"})
    return flags


def _project_scripts() -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return project["scripts"]


def _module_path(entry_point: str) -> str:
    module, separator, callable_name = entry_point.partition(":")
    assert separator == ":" and callable_name == "app"
    return f"src/{module.replace('.', '/')}.py"


def _documented_command_flags(reference: str, command: str) -> set[str]:
    match = re.search(
        rf"^## `{re.escape(command)}`\n(?P<body>.*?)(?=^## `|\Z)",
        reference,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"docs/api/cli.md has no section for {command}"
    flags: set[str] = set()
    for line in match.group("body").splitlines():
        if not line.startswith("| `"):
            continue
        option_cell = line.split("|", maxsplit=2)[1]
        for code_span in re.findall(r"`([^`]+)`", option_cell):
            flag = code_span.split(maxsplit=1)[0]
            if flag.startswith("-"):
                flags.add(flag)
    return flags


def test_config_exports_are_referenced() -> None:
    _assert_documented("src/lanfactory/config/__init__.py", "docs/api/config.md")


def test_hf_exports_are_referenced() -> None:
    _assert_documented("src/lanfactory/hf/__init__.py", "docs/api/hf.md")


def test_hf_constants_are_current() -> None:
    reference = (ROOT / "docs/api/hf.md").read_text()
    module = "src/lanfactory/hf/__init__.py"
    assert f"`{_literal_assignment(module, 'DEFAULT_REPO_ID')}`" in reference
    assert f"`{_literal_assignment(module, 'DEFAULT_LICENSE')}`" in reference
    for network_type in _literal_assignment(module, "VALID_NETWORK_TYPES"):
        assert f"`{network_type}`" in reference


def test_every_installed_cli_has_an_exact_option_reference() -> None:
    reference = (ROOT / "docs/api/cli.md").read_text()
    scripts = _project_scripts()
    documented_commands = set(re.findall(r"^## `([^`]+)`$", reference, re.MULTILINE))
    assert documented_commands == set(scripts)

    for command, entry_point in scripts.items():
        source_flags = _cli_option_flags(_module_path(entry_point))
        documented_flags = _documented_command_flags(reference, command)
        assert documented_flags == source_flags, (
            f"docs/api/cli.md flags drifted for {command}: "
            f"missing={sorted(source_flags - documented_flags)}, "
            f"extra={sorted(documented_flags - source_flags)}"
        )


def test_network_inspector_exports_are_referenced() -> None:
    _assert_documented(
        "src/lanfactory/network_inspectors/__init__.py",
        "docs/api/network_inspectors.md",
    )
