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
transform applied at inference) is a log choice probability. A CPN
[derived from a trained LAN](#deriving-auxiliary-networks-from-a-trained-lan)
uses a different, one-column-wider input layout — see that section before
mixing the two.

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

A CPN or OPN learns a quantity that is already implied by
the LAN of the same model: the choice probability is the LAN's density
integrated over reaction time for one choice, and the omission probability
under a deadline is one minus the density integrated up to that deadline.
`lanfactory.derive` computes those integrals from a trained LAN and writes a
training corpus for the auxiliary network on one laptop core instead of a
fresh cluster simulation — and the trainers consume it unchanged.

**From which LAN.** Any `(1, n_params + 2)` LAN for the model — a production
network from the Hub (`download-hf --network-type lan --model-name ddm`) or
your own `torchtrain` / `jaxtrain` export. Its identity is recorded in every
pickle's `generator_config["source"]` under the keys downstream tools read by
name — `derivation_method` (`"derived-from-lan"`), `aux_category` (`choice` /
`omission`), `source_lan_run_uuid` (`--source-run-uuid`, else parsed from the
trainer's file name), `source_lan_sha256`, `source_lan_hf_commit`
(`--source-hf-revision`), `integration_grid` and `integration_max_t` — and
again in `derive_manifest.json`.

**Grid.** The density is integrated over `[t_min, 20 s]` on a grid refined
per parameter vector around the onset: the non-decision time `t`, or
`t - st` for the models whose ssms config declares an `ndt_edge_shift`. The
[API reference](api/derive.md) states the grid and tail policies. The LAN
leaks a little mass below its onset (mean 0.0035, p99 0.052 on the Hub ddm LAN);
that leak is part of what it predicts, so it is kept, and recorded as
`derive_leak_below_onset_p99`.

**Row layouts.** Every row is `n_params + 1` wide, in ssms parameter order;
labels are probabilities in `[0, 1]` for the `bcelogit` loss.

| type | training row | label |
| --- | --- | --- |
| `cpn` | `[theta..., choice]`, one row per choice code | `P(choice | theta)` |
| `opn` | `[theta..., deadline]`, deadline last | `P(no response before deadline | theta)` |

A derived CPN corpus carries the choice as an *input* (one row per choice
code), so no choice category is assumed and the network is `n_params + 1`
wide (`[1, 5]` for `ddm`). This differs from a simulated ssms CPN corpus,
whose rows are `[theta...]` alone with a single `P(choice = 1)` label
(`n_params` wide, `[1, 4]` for `ddm`, the layout of the CPN section above):
the two CPN artifacts are not interchangeable, and a consumer must know which
one it is loading. The `opn` rows match ssms' simulated layout.

For `opn`, deadlines are sampled per parameter vector: a 0.7 share from
the LAN's own
reaction-time quantiles under that theta, so the training deadlines sit where
the omission probability is informative, and the rest uniform on ssms'
deadline bounds `(0.001, 10)` so the corpus still covers the box. Every label
is the LAN's mass as a fraction of the LAN's own total (see *Renormalisation*
below); the float32 clip to `[0, 1]` is a safety net.

**Deriving and training.**

```bash
derive-aux --from-onnx ddm.onnx --network-type cpn --model-name ddm \
  --output-folder data/cpn/ddm
torchtrain --config-path cpn.yaml --training-data-folder data/cpn/ddm \
  --networks-path-base networks
```

`cpn.yaml` is the CPN config from above with `NETWORK_TYPE: "cpn"` and a batch
size that divides the rows per file: with the default `--n-theta-per-file
4096` a file holds 4096 rows for `opn` and `4096 × n_choices` for `cpn`
(8192 for a two-choice model), so `CPU_BATCH_SIZE: 512` works for both. The
Python entry point is `lanfactory.derive.derive_aux_corpus`; see the
[API reference](api/derive.md#derived-corpora) for the pickle contract.

### Renormalisation, the window and the fallback

A LAN's integrated mass is not one: over 20 000 draws from the ddm box the
Hub ddm LAN's total has median `|total − 1|` 0.0037 (p99 0.080, range
0.877–1.219). Against simulation that error is mostly *scale* — dividing by
the total takes the CPN label error from 0.035 mean / 0.17 max to
0.004 / 0.033 and the OPN error from 0.026 / 0.22 to 0.012 / 0.18 — so
**labels are renormalised by the network's own total, and the total is
recorded, never hidden**: `derive_total_mass_{mean,min,max}` in the
manifest's corpus-wide `derive_stats`.

What remains sits where the total is off. A parameter vector whose total lies
outside the window `[0.98, 1.03]` (endpoints trusted; 3.7 % of the ddm box),
or is not a number, is labelled by **ssms simulation** instead, from 20 000
trials: the CPN label is the choice frequency among trials that respond
before `max_t`, the OPN label the share that does not respond before the
row's deadline, both from one base-model run. That bounds the unflagged residual against
simulation at 0.023 (CPN) / 0.033 (OPN). `derive_fallback_frac` records the
share that fell back and `derive_sim_past_max_t_max` the simulation's own
blind spot: the largest share of base-model trials still running at `max_t`,
which ssms returns with a choice or censors as `rt = −999` (`None` when
nothing fell back). ssms at `Δt = 10⁻³` is itself biased at `a < 0.5` (about
0.02 in `P(choice = 1)`), so a simulated label is a reference, not ground
truth.

Integrating a file of 4096 θ takes about a second; nearly all of a file's
cost is the fallback, whose thetas are the slow ones to simulate (large `a`,
small `|v|`: long reaction times).

## Where the variants came from

The CPN and OPN walkthroughs used to be standalone notebook tutorials that
repeated the LAN workflow verbatim. They were consolidated into this page —
follow the [training tutorial](basic_tutorial/basic_tutorial_lan_torch.ipynb)
with the deltas above to train either type, and see the
[JAX how-to](basic_tutorial/basic_tutorial_lan_jax.ipynb) for the alternative
backend.
