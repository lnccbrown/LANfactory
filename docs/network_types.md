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

**Grid.** The density is integrated with the trapezoid rule over
`[t_min, max_t]` with `max_t = 20` s, ssms' default and the LANs' training
support; nothing is extrapolated past it. The grid is refined per parameter
vector around the non-decision time `--onset-param` (default `t`): 32 points
below `t − 0.05`, 128 on the knee `[t − 0.05, t]`, 600 on `[t, t + 1]` and
400 on the tail, ~1160 points that match a 16 000-point uniform grid to
5 × 10⁻⁵. A uniform 1000-point grid was measured to carry 15–25 % quadrature
error at `a < 0.5` on the Hub ddm LAN, so it is only the fallback for a model
without that parameter (`--onset-param ''` selects it explicitly, sized by
`--grid-points`). The LAN leaks a little mass below `t` (mean 0.0035, p99
0.052 on the Hub ddm LAN); integrating from `t` only was tested and rejected,
and the leak is recorded per file as `derive_leak_below_onset_p99` — on the
onset grid only (`None` on a uniform grid, where it would mostly be
quadrature error).

**Row layouts.** Every row is `n_params + 1` wide, in ssms parameter order;
labels are probabilities in `[0, 1]` for the `bcelogit` loss.

| type | training row | label |
| --- | --- | --- |
| `cpn` | `[theta..., choice]`, one row per choice code | `P(choice | theta)` |
| `opn` | `[theta..., deadline]`, deadline last | `P(no response before deadline | theta)` |
| `gonogo` | `[theta..., deadline]` | `P(nogo or omission | theta)`; nogo = every choice but the largest code, as in ssms' `nogo_p` |

A derived CPN corpus carries the choice as an *input* (one row per choice
code), so no choice category is assumed and the network is `n_params + 1`
wide (`[1, 5]` for `ddm`). This differs from a simulated ssms CPN corpus,
whose rows are `[theta...]` alone with a single `P(choice = 1)` label
(`n_params` wide, `[1, 4]` for `ddm`, the layout of the CPN section above):
the two CPN artifacts are not interchangeable, and a consumer must know which
one it is loading. The `opn` and `gonogo` rows match ssms' simulated layout.

For `opn` and `gonogo`, deadlines are sampled per parameter vector: a
`--deadline-quantile-frac` share (default 0.7) from the LAN's own
reaction-time quantiles under that theta, so the training deadlines sit where
the omission probability is informative, and the rest uniform on ssms'
deadline bounds `(0.001, 10)` so the corpus still covers the box. Every label
is the LAN's mass as a fraction of the LAN's own total (see *Renormalisation*
below); the float32 clip to `[0, 1]` is a safety net, and the omission term is
formed once, before it enters either label, so `gonogo == nogo_before /
total + opn` holds row by row in the written corpus.

**Deriving and training.**

```bash
derive-aux --from-onnx ddm.onnx --network-type cpn --model-name ddm \
  --output-folder data/cpn/ddm
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

### Renormalisation, the window and the fallback

A LAN's integrated mass is not one. Over 20 000 uniform draws from the ddm box
the Hub ddm LAN's total has median `|total − 1|` 0.0037 and p99 0.080, with
min 0.877 / max 1.219 and 1.9 % of the box beyond 0.05; two regions carry the
error — `a > 2.2 & |v| < 0.5` over-estimates the tail (+0.07 mean, +0.22 max)
and `v > 2.5, z > 0.8, a > 2.2` halves the early peak (−0.06). Against
simulation that error is mostly *scale*: labelling a CPN with the raw
`mass(c)` gives a mean error of 0.035 (max 0.17) while `mass(c) / total`
gives 0.004 (max 0.033); for an OPN `1 − F(d)` gives 0.026 / 0.22 and
`1 − F(d) / F(max_t)` 0.012 / 0.18. So **labels are renormalised by the
network's own total; the total is recorded, never hidden** — per file in
`generator_config["derive_stats"]` (`derive_total_mass_{mean,min,max}`), per
corpus in the manifest, and the source LAN's whole-box `survey` rides in the
manifest under `lan_survey`.

What remains after renormalising sits where the total is off. A parameter
vector whose total lies outside `--fallback-window` (default the open
interval `0.98 1.03`, which flags 3.7 % of the ddm box) is therefore labelled
by **ssms simulation** instead of the LAN (`--fallback-n-sim` trials, default
20 000): the CPN label is the choice frequency conditional on responding
within `max_t` — what the renormalised LAN label estimates — and the OPN /
go-no-go labels come from the `{model}_deadline` simulator with the row's
deadline. With the fallback on, the unflagged residual against simulation is
≤ 0.023 (CPN) / ≤ 0.033 (OPN). `--no-fallback` labels everything from the
LAN. Every file and the manifest record the share that fell back
(`derive_fallback_frac`) and the simulation's own blind spot,
`derive_sim_past_max_t_max` — the largest share of base-model trials at or
beyond `max_t` among the fallback thetas (ssms returns those with a choice,
which no label conditioned on `[0, max_t]` can carry; `None` when nothing fell
back). Note that ssms at `Δt = 10⁻³` is itself biased at `a < 0.5` (about
0.02 in `P(choice = 1)`), so a simulated label is a reference, not ground
truth.

**Cost.** Measured on the Hub ddm LAN with `n_files = 2`, `n_theta_per_file
= 4096` and the defaults (laptop, single-threaded ssms):

| type | fallback fraction | seconds per file | share of file time in simulation | 100 files (extrapolated) |
| --- | --- | --- | --- | --- |
| `cpn` | 4.0 % | 39 | 97 % | ≈ 65 min |
| `opn` | 3.9 % | 62 | 98 % | ≈ 105 min |

The one-off survey of the LAN (20 000 θ) adds about 5 s per corpus.
Integrating a file takes about a second; the rest is the fallback, whose
thetas are the slow ones to simulate (large `a`, small `|v|`: long reaction
times). `opn` / `gonogo` simulate each fallback theta twice — the base model
for `derive_sim_past_max_t_max` and the deadline model for the label — but
the deadline run stops at the deadline and is cheap.

## Where the variants came from

The CPN and OPN walkthroughs used to be standalone notebook tutorials that
repeated the LAN workflow verbatim. They were consolidated into this page —
follow the [training tutorial](basic_tutorial/basic_tutorial_lan_torch.ipynb)
with the deltas above to train either type, and see the
[JAX how-to](basic_tutorial/basic_tutorial_lan_jax.ipynb) for the alternative
backend.
