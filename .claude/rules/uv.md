---
description: Enforce uv as the package manager for all Python operations
globs:
  - "**/*.py"
  - "**/pyproject.toml"
---

- Always use `uv run` to execute commands — never bare `python`, `pytest`, `ruff`, or other tools.
- When working in this repo (local development, CI, or scripts), do not use `pip install`; use `uv sync` (with `--extra` or `--all-groups` flags) to manage dependencies. End-user installation docs may still reference `pip`.
- The `uv.lock` file is the source of truth for resolved dependency versions.
- When adding dependencies, add them to `pyproject.toml` and run `uv sync`.
