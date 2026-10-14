# Network types: LAN, CPN, OPN

LANfactory trains three kinds of networks. All of them share the same MLP
architecture and the same training workflow — the one shown in the
[training tutorial](basic_tutorial/basic_tutorial_lan_torch.ipynb) — and differ
only in what they learn, which loss they train with, and a handful of config
values. This page carries those deltas; everything else (data generation,
dataloaders, training loop, inference) is identical to the tutorial.

| | LAN | CPN | OPN |
| --- | --- | --- | --- |
| Full name | Likelihood Approximation Network | Choice Probability Network | Option Probability Network |
| Learns | log-likelihood of `(rt, choice)` given parameters | choice probability given parameters | response-before-deadline probability given parameters |
| Training output | `logprob` | `logits` | `logits` |
| Loss | Huber | BCE with logits | BCE with logits |
| Rows in training data | many `(rt, choice)` rows per parameter set | one row per parameter set | one row per parameter set |
| Configs | `network_config_mlp` / `train_config_mlp` | `network_config_cpn` / `train_config_cpn` | `network_config_opn` / `train_config_opn` |
| `network_type` | `"lan"` | `"cpn"` | `"opn"` |
| Typical use | likelihood for RT + choice inference in [HSSM](https://lnccbrown.github.io/HSSM/) | choice-probability-only models | deadline / omission models |

The CPN and OPN configs are aliases of a shared `choice_prob` config — the two
network types are distinguished by the training data you feed them and the
`network_type` you pass, not by architecture.

## LAN (the tutorial default)

The [training tutorial](basic_tutorial/basic_tutorial_lan_torch.ipynb) trains a
LAN end-to-end; its config delta is the baseline the other two types deviate
from:

```python
network_config = lanfactory.config.network_configs.network_config_mlp  # logprob
train_config = lanfactory.config.network_configs.train_config_mlp  # huber loss

train_dl, valid_dl, input_dim = lanfactory.trainers.make_train_valid_dataloaders(
    file_ids=file_list, batch_size=1000, network_type="lan"
)
net = lanfactory.trainers.TorchMLPFactory(
    network_config=network_config, input_dim=input_dim, network_type="lan"
)
```

The network input is `model parameters + (rt, choice)`; the output approximates
the log-likelihood of that trial.

## CPN: choice probabilities

A CPN predicts the probability of a choice given a parameter set — no reaction
times involved. The training data (generated exactly as in the tutorial)
contributes **one row per parameter set**, so pick a `batch_size` that divides
`n_parameter_sets`. The delta from the tutorial:

```python
network_config = lanfactory.config.network_configs.network_config_cpn  # logits
train_config = lanfactory.config.network_configs.train_config_cpn  # bcelogit loss

train_dl, valid_dl, input_dim = lanfactory.trainers.make_train_valid_dataloaders(
    file_ids=file_list, batch_size=100, network_type="cpn"
)
net = lanfactory.trainers.TorchMLPFactory(
    network_config=network_config, input_dim=input_dim, network_type="cpn"
)

# At inference time, pass the network type so logits are handled correctly:
network = lanfactory.trainers.LoadTorchMLPInfer(
    model_file_path=network_file_path,
    network_config=network_config,
    input_dim=input_dim,
    network_type="cpn",
)
```

The network input is just the model parameters; the output (after the logit
transform applied at inference) is a log choice probability.

## OPN: option probabilities under a deadline

An OPN predicts the probability that a response happens at all — e.g. before a
deadline — which makes it the right companion for deadline model variants such
as `ddm_deadline`. It is a CPN with a deadline-aware generative model; the
extra delta on top of the CPN block above is the model choice:

```python
MODEL = "ddm_deadline"
generator_config["model"] = MODEL
# ddm_deadline is a runtime-derived variant; build its model config with
# ModelConfigBuilder (it is not stored in ssms.config.model_config).
model_config = ssms.config.ModelConfigBuilder.from_model(MODEL)

network_config = lanfactory.config.network_configs.network_config_opn  # logits
train_config = lanfactory.config.network_configs.train_config_opn  # bcelogit loss

train_dl, valid_dl, input_dim = lanfactory.trainers.make_train_valid_dataloaders(
    file_ids=file_list, batch_size=100, network_type="opn"
)
```

The network input is the model parameters including the deadline; training
data again contributes one row per parameter set.

## Where the variants came from

The CPN and OPN walkthroughs used to be standalone notebook tutorials that
repeated the LAN workflow verbatim. They were consolidated into this page —
follow the [training tutorial](basic_tutorial/basic_tutorial_lan_torch.ipynb)
with the deltas above to train either type, and see the
[JAX how-to](basic_tutorial/basic_tutorial_lan_jax.ipynb) for the alternative
backend.
