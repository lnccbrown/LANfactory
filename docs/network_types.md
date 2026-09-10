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

## Deriving auxiliary networks from a trained LAN

A CPN, OPN, or go/no-go network learns a quantity that is already implied by
the LAN of the same model: the choice probability is the LAN's density
integrated over reaction time for one choice, and the omission probability
under a deadline is one minus the density integrated up to that deadline.
`lanfactory.derive` computes those integrals from a trained LAN and writes a
training corpus for the auxiliary network, so the corpus takes minutes on a
laptop instead of a fresh cluster simulation — and the trainers consume it
unchanged.

**From which LAN.** Any `(1, n_params + 2)` LAN for the model — a production
network from the Hub (`download-hf --network-type lan --model-name ddm`) or
your own `torchtrain` / `jaxtrain` export. The LAN's file hash, training run
uuid (parsed from the trainer's filename when present) and Hub revision are
recorded in every pickle and in `derive_manifest.json`.

**Tail policy.** The density is integrated with the trapezoid rule on a
uniform grid over `[t_min, max_t]` with `max_t = 20` s, ssms' default and the
LANs' training support; nothing is extrapolated past it and the masses are
never renormalised. ssms does not censor a base model at `max_t` (an
un-terminated trial is returned at `rt ≈ max_t + t` with its sign-implied
choice), so a simulated CPN label sums to one over choices while the derived
masses fall short by the density the LAN places past 20 s. The total mass is
recorded per file (`generator_config["derive_stats"]`) and per corpus (the
manifest) so that deficit stays visible.

**Row layouts.** Every row is `n_params + 1` wide, in ssms parameter order;
labels are probabilities in `[0, 1]` for the `bcelogit` loss.

| type | training row | label |
| --- | --- | --- |
| `cpn` | `[theta..., choice]`, one row per choice code | `P(choice | theta)` |
| `opn` | `[theta..., deadline]`, deadline last | `P(no response before deadline | theta)` |
| `gonogo` | `[theta..., deadline]` | `P(nogo or omission | theta)`; nogo = every choice but the largest code, as in ssms' `nogo_p` |

A CPN corpus carries the choice as an *input* (one row per choice code), so no
choice category is assumed. For `opn` and `gonogo`, deadlines are sampled per
parameter vector: a `--deadline-quantile-frac` share (default 0.7) from the
LAN's own reaction-time quantiles under that theta, so the training deadlines
sit where the omission probability is informative, and the rest uniform on
ssms' deadline bounds `(0.001, 10)` so the corpus still covers the box.

**Deriving and training.**

```bash
derive-aux --from-onnx ddm.onnx --model ddm --type cpn --out data/cpn/ddm
torchtrain --config-path cpn.yaml --training-data-folder data/cpn/ddm \
  --networks-path-base networks
```

`cpn.yaml` is the CPN config from above with `NETWORK_TYPE: "cpn"` and a batch
size that divides the rows per file: with the default `--n-theta-per-file
4096` a file holds 4096 rows for `opn` / `gonogo` and `4096 × n_choices` for
`cpn` (8192 for a two-choice model), so `CPU_BATCH_SIZE: 512` works for all
three. The Python entry point is `lanfactory.derive.derive_aux_corpus`; see the
[API reference](api/derive.md#derived-corpora) for the pickle contract and the
provenance keys.

## Where the variants came from

The CPN and OPN walkthroughs used to be standalone notebook tutorials that
repeated the LAN workflow verbatim. They were consolidated into this page —
follow the [training tutorial](basic_tutorial/basic_tutorial_lan_torch.ipynb)
with the deltas above to train either type, and see the
[JAX how-to](basic_tutorial/basic_tutorial_lan_jax.ipynb) for the alternative
backend.
