<div>
    <a href="https://ccbs.carney.brown.edu/brainstorm" style="display: block; float: right; padding: 10px">
        <img src="images/Brain-Bolt-%2B-Circuits.gif" style="width: 100px;">
    </a>
    <img src="images/mainlogo.png" style="width: 175px;">
</div>

## LANfactory

![PyPI](https://img.shields.io/pypi/v/lanfactory)
![PyPI_dl](https://img.shields.io/pypi/dm/lanfactory)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`lanfactory` is a lightweight Python package for training
[likelihood approximation networks](https://elifesciences.org/articles/65074)
(LANs) — and their choice-probability siblings — for sequential sampling models
(SSMs), using PyTorch or JAX/Flax. Starting from simulator-generated training
data, it provides dataloaders, network factories, and training loops, and
exports the trained networks to ONNX so they can serve as likelihoods in
[HSSM](https://lnccbrown.github.io/HSSM/).

---

## Ecosystem fit

LANfactory is the network-training layer of the HSSM ecosystem: it trains
LAN/CPN/OPN networks on simulated data and exports them to ONNX, in the form
HSSM consumes as likelihoods.

For the full map — what each package owns, how artifacts flow between them,
and which versions work together — see
[The HSSM ecosystem](https://lnccbrown.github.io/HSSM/ecosystem/).

---

## Installation

```bash
pip install lanfactory
```

Optional integrations ship as extras: `lanfactory[mlflow]` (experiment
tracking), `lanfactory[hf]` (HuggingFace Hub upload/download), `lanfactory[sbi]`
and `lanfactory[bayesflow]` (ONNX export of externally trained networks), or
`lanfactory[all]` for everything.

---

## Quickstart

Given a folder of training data files generated with
[ssm-simulators](https://lnccbrown.github.io/ssm-simulators/), the minimal
PyTorch training loop is:

```python
from pathlib import Path
import lanfactory

file_list = list(Path("training_data").glob("*.pickle"))

train_dl, valid_dl, input_dim = lanfactory.trainers.make_train_valid_dataloaders(
    file_ids=file_list, batch_size=128, network_type="lan"
)
net = lanfactory.trainers.TorchMLPFactory(
    network_config=lanfactory.config.network_configs.network_config_mlp,
    input_dim=input_dim,
    network_type="lan",
)
trainer = lanfactory.trainers.ModelTrainerTorchMLP(
    model=net,
    train_config=lanfactory.config.network_configs.train_config_mlp,
    train_dl=train_dl,
    valid_dl=valid_dl,
)
trainer.train_and_evaluate(output_folder="torch_models/ddm", output_file_id="ddm")
```

For the full walkthrough — data generation, configuration, training, and
inspecting the learned likelihood — see the
[training tutorial](basic_tutorial/basic_tutorial_lan_torch.ipynb).

---

## Export to ONNX

Trained PyTorch networks convert to ONNX with the `transform-onnx` CLI:

```bash
transform-onnx --network-config-file <network_config.pickle> \
  --state-dict-file <state_dict.pt> --input-shape <input_dim> \
  --output-onnx-file <model.onnx>
```

The resulting file can be used directly with
[HSSM](https://lnccbrown.github.io/HSSM/) — see
[The ONNX likelihood contract](https://lnccbrown.github.io/HSSM/how_to/custom_onnx_likelihoods/) for the artifact rules. Networks trained outside LANfactory
can be exported the same way — see the
[sbi](exporting_sbi_models.md) and [bayesflow](exporting_bayesflow_models.md)
export guides.

---

## Where to go next

- **Tutorials**
    - [Train a network (PyTorch LAN)](basic_tutorial/basic_tutorial_lan_torch.ipynb) — the canonical end-to-end training walkthrough.
    - [How to train with the JAX backend](basic_tutorial/basic_tutorial_lan_jax.ipynb) — the same workflow on JAX/Flax.
    - [Exporting sbi → ONNX](tutorials/exporting_sbi_to_onnx.ipynb) and [exporting bayesflow → ONNX](tutorials/exporting_bayesflow_to_onnx.ipynb) — runnable export notebooks.
- **Guides**
    - [Network types: LAN, CPN, OPN](network_types.md) — what each network learns and how their configs differ.
    - [MLflow integration](using_mlflow.md) — track and compare training runs.
    - [HuggingFace Hub](using_huggingface.md) — upload and download trained networks.
    - [Exporting sbi models](exporting_sbi_models.md) and [exporting bayesflow models](exporting_bayesflow_models.md) — bring externally trained networks into HSSM.
- **API reference** — [config](api/config.md), [trainers](api/trainers.md), [onnx](api/onnx.md), [hf](api/hf.md), [utils](api/utils.md).

We hope this package may be helpful in case you attempt to train
[LANs](https://elifesciences.org/articles/65074) for your own research.
