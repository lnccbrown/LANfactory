# What LANs are, and where LANfactory fits

Many sequential sampling models (SSMs) have no closed-form likelihood. You can
*simulate* from them cheaply, but you cannot write down the density that
Bayesian inference needs. **Likelihood approximation networks** (LANs,
[Fengler et al., 2021](https://elifesciences.org/articles/65074)) close that
gap: a neural network is trained, once and offline, to approximate the
log-likelihood of a trial given the model's parameters. Inference then calls
the network instead of an analytical formula.

That has two consequences worth internalising:

- **Training is amortised.** The expensive step happens once per model, not
  once per dataset. A trained network is an artifact you keep, share, and
  reuse across studies.
- **The network is only valid where it was trained.** Its parameter bounds are
  the training bounds. Outside them the approximation is unconstrained, which
  is why bounds travel with the artifact rather than being an inference-time
  choice.

## The three network types

LANfactory trains three variants, which differ in what they learn rather than
in architecture:

| | Learns | Used for |
|---|---|---|
| **LAN** | log-likelihood of `(rt, choice)` | RT + choice inference |
| **CPN** | choice probability | choice-only models |
| **OPN** | probability of responding before a deadline | deadline / omission models |

The [network types reference](network_types.md) carries the exact config
deltas for each.

## Where LANfactory sits

LANfactory is the **training** layer of the HSSM ecosystem — it neither
simulates nor performs inference:

| Package | Owns |
|---|---|
| [ssm-simulators](https://lnccbrown.github.io/ssm-simulators/) | the generative models and the training data |
| **LANfactory** | dataloaders, network factories, training loops, ONNX export |
| [HSSM](https://lnccbrown.github.io/HSSM/) | Bayesian inference using the exported networks as likelihoods |

The handoff between the last two is a file: an ONNX graph obeying
[the ONNX likelihood contract](https://lnccbrown.github.io/HSSM/how_to/custom_onnx_likelihoods/).
Because that contract is framework-agnostic, LANfactory also exports networks
trained *elsewhere* — sbi and BayesFlow — into the same consumable form, which
is what the two export guides cover.

## Where to go next

- [Train your first LAN (PyTorch)](basic_tutorial/basic_tutorial_lan_torch.ipynb) — the end-to-end walkthrough
- [Network types: LAN, CPN, OPN](network_types.md) — config deltas for the variants
- [Share trained networks on HuggingFace Hub](using_huggingface.md) — publishing the artifact
