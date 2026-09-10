# Test fixtures: real LANs

Fixtures live here, not under `tests/test_data/`: that tree is the suite's
scratch output and the session-scoped `cleanup_afters_tests` fixture in
`tests/conftest.py` removes it after every run.

## `ddm.onnx`

The production DDM likelihood approximation network HSSM downloads by default,
copied unchanged from the Hugging Face Hub.

| | |
| --- | --- |
| Source | https://huggingface.co/franklab/HSSM/blob/main/ddm.onnx |
| Hub commit | `01f5d4d0fa9188940ab541a979f933550b29616a` |
| sha256 | `09f685c18d3bdbd9b54fa89bed3b5bc0e2c76566e7ed0ae5e24df2c9b04f1b0e` |
| Copied | 2026-09-10 |
| Size | ~85 KB |
| Input contract | `(1, 6)` float32 = `[v, a, z, t, rt, choice]` (the single-trial contract, see `lanfactory.onnx.contract`) |
| Output | `(1, 1)` log-likelihood |
| Model | ssms `ddm`; params `v, a, z, t`; choices `[-1, 1]` |

Used by `tests/derive` to check that the masses `lanfactory.derive` integrates
from a real LAN match ssms simulations, and to exercise `derive_aux_corpus` and
the `derive-aux` CLI on a production network. The file is a fixture: never
modify it in place — replace it and update this table.
