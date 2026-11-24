# Train a LAN for an ssms model

`train_lan_for_model` joins ssms data generation, the PyTorch trainer, concrete
single-trial ONNX export and five validation gates. It accepts built-in and
runtime-registered RT/choice models. It does not publish a network or submit jobs.

```python
from lanfactory.recipes import train_lan_for_model

artifact = train_lan_for_model("angle", "lan/angle-toy", preset="toy", seed=0)
print(artifact.validation)  # toy statistical gates can fail
if artifact.validated:
    import hssm
    model = hssm.HSSM(
        data, model="angle", loglik=str(artifact.onnx_path),
        loglik_kind="approx_differentiable",
        model_config=artifact.hssm_model_config,
    )
```

The CLI twin is `lan-from-ssms --model angle --out lan/angle-toy --preset toy`.
It exits nonzero unless all mandatory gates ran and passed. `--no-validate`
produces diagnostic artifacts only. An existing nonempty output directory is
refused so that a failed run cannot leave a stale certificate.

## Presets

| Preset | Sets/file | Simulations/set | Labels/set | Files | Epochs | Fresh samples/check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| toy | 32 | 2,000 | 128 | 2 | 6 | 8,000 |
| dev | 256 | 10,000 | 512 | 4 | 40 | 20,000 |
| production | 5,000 | 200,000 | 2,000 | 300 | 100 | 200,000 |

Toy is a plumbing check. Dev supports a bounded development experiment. Neither
is a production likelihood certificate. Production data settings match the order
of magnitude of LAN_pipeline_minimal's production configs; epoch count is a
recipe default, not a measured convergence guarantee. Production additionally
requires `allow_production=True` / `--allow-production`. Run it only on an
explicitly authorized cluster. This recipe uses one generation CPU; the existing
cluster pipeline remains the scalable training/recovery/publishing route.

## Custom models and conditional designs

Pass `model_config=<ssms dict>` for a custom simulator. Numeric training bounds
are copied and normalized in `params` order; the registry is unchanged. Dependent
symbolic bounds are rejected. `generator_config` accepts nested ssms overrides;
`network_config` and `train_config` override the usual LANfactory dictionaries.
The file-level train/validation split uses distinct parameter draws. Each part's
hash, row count and theta-index offset are saved in the artifact metadata. The
batch size is adjusted to the largest divisor of file row counts no greater than
the request, with a warning and the actual value in the manifest.

Use `fixed_parameters={"a": 1.0, "z": 0.5, "t": 0.2}` for a conditional
experiment. Fixed coordinates remain in the network's input order and become
numeric fixed priors in the HSSM config. They are not free parameters. Preserve
these fixed values when fitting; the network was not trained away from them.

The CLI accepts `--model-config trusted.module:attribute` and JSON
`--fixed-parameters '{"a": 1.0}'`. Import only your own trusted model code.

## Guards and gates

ssms 0.16.0's KDE filters read the wrong config key and use non-strict boundary
comparisons ([#392](https://github.com/lnccbrown/ssm-simulators/issues/392),
[#432](https://github.com/lnccbrown/ssm-simulators/pull/432)). The recipe injects a
local guarded pipeline; it does not monkeypatch ssms. It reads
`pipeline.simulation_filters` and applies strict mode/std/choice-count thresholds.
RTs that are nonfinite/nonpositive, omissions, unknown choices and sampled
horizon hits are rejected before KDE construction. A long sequence of rejections
aborts with an actionable error. This is a conservative guard, not a resolution
of ssms' differing max_t policies ([#404](https://github.com/lnccbrown/ssm-simulators/issues/404)).
Displacement by `t` is disabled explicitly: not every model's support starts at
`t`. PyDDM labeling is optional and limited to single-particle two-choice
diffusions; it does not provide an exact oracle for arbitrary models.

| Gate | Requirement |
| --- | --- |
| G1 | Concrete single-trial ONNX, correct width, one input and one scalar output; ONNX Runtime loads/runs it. |
| G2 | Up to 1,000 training-file rows agree between the trainer and ONNX within 1e-4 absolute error. |
| G3 | HSSM accepts the explicit model config and produces finite initial logp on fresh simulated data. Missing HSSM is a skipped, nonpassing gate. |
| G4 | At five inset parameter draws, choice-weighted fresh KDE comparison: total mass in [0.9, 1.1], Hellinger <= 0.15 and distance/sampling-floor <= 3. Two independent simulations measure the floor. |
| G5 | HSSM's JAX ONNX derivative agrees with finite differences of the same graph (scaled max error <= 0.01) and with paired-seed finite differences of fresh simulated log-RT KDEs. |

G5 scales derivatives by each free parameter's training span. It uses +/- 4% of
that span and two independent paired-seed references at two RT quantiles per
choice, at five parameter draws. Reference RMS noise must be <= 0.5; the network's
RMS error must be <= 0.5 + 0.35 * reference RMS. These explicit development
thresholds are not proof of posterior calibration, and a noisy reference cannot
silently pass. Production validation needs more points, more simulation, boundary
and tail checks, and ensemble recovery in LAN_pipeline_minimal.

## Outputs

All runs write ONNX, trainer files, data parts, `manifest.json` and
`validation.json`. The manifest records configs, seeds, dependency versions,
source revision/dirty state, model-config hash, ONNX/data hashes, runtime and
rejection counts. Only a fully passing run writes `model_config.json` and sets
`artifact.hssm_model_config`. A failed/disabled/skipped gate leaves it `None`.
The HSSM bounds are exactly the training box, with fixed priors recorded explicitly.
No artifact from this recipe is marked `production_validated`.
