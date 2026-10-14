<div style="position: relative; width: 100%;">
  <img src="docs/images/mainlogo.png" alt="LANfactory" style="width: 175px;">
  <a href="https://ccbs.carney.brown.edu/brainstorm" style="position: absolute; right: 0; top: 50%; transform: translateY(-50%);">
    <img src="docs/images/Brain-Bolt-%2B-Circuits.gif" alt="Brown Brainstorm" style="width: 100px;">
  </a>
</div>

# LANfactory

[![DOI](https://zenodo.org/badge/386076271.svg)](https://doi.org/10.5281/zenodo.17137303)
![PyPI](https://img.shields.io/pypi/v/lanfactory)
![PyPI downloads](https://img.shields.io/pypi/dm/lanfactory)
[![Run tests](https://github.com/lnccbrown/LANfactory/actions/workflows/run_tests.yml/badge.svg)](https://github.com/lnccbrown/LANfactory/actions/workflows/run_tests.yml)
[![codecov](https://codecov.io/gh/lnccbrown/LANfactory/branch/main/graph/badge.svg)](https://codecov.io/gh/lnccbrown/LANfactory)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LANfactory trains likelihood approximation networks (LANs), choice probability
networks (CPNs), and option probability networks (OPNs) with PyTorch or
JAX/Flax. It exports concrete-shape ONNX artifacts for use as likelihoods in
[HSSM](https://lnccbrown.github.io/HSSM/).

[Documentation](https://lnccbrown.github.io/LANfactory/) ·
[Start with the PyTorch tutorial](https://lnccbrown.github.io/LANfactory/basic_tutorial/basic_tutorial_lan_torch/) ·
[API reference](https://lnccbrown.github.io/LANfactory/api/trainers/) ·
[HSSM ecosystem map](https://lnccbrown.github.io/HSSM/ecosystem/)

## Install

```bash
pip install lanfactory
```

Optional integrations are available through the `mlflow`, `hf`, `sbi`, and
`bayesflow` extras, or together through `lanfactory[all]`. The documentation is
the canonical source for training, export, configuration, and sharing guidance.

## Contributor bootstrap

Install the development environment and run the package gates:

```bash
uv sync --all-groups
uv run pytest tests/
uv run ruff check src/lanfactory
uv run ruff format --check .
```

Build or serve the documentation through the repository-owned entry point:

```bash
./scripts/docs.sh build
./scripts/docs.sh serve
```

Notebook execution remains a separate package test workflow; the documentation
build renders the committed notebook outputs without executing them.

## Citation

Use the archived LANfactory release DOI:
[10.5281/zenodo.17137303](https://doi.org/10.5281/zenodo.17137303).
