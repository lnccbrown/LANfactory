"""ModelTrainerJaxMLP uses a supplied lr_dict, else the default."""

from types import SimpleNamespace as NS

import jax
from lanfactory.trainers.jax_mlp import JaxMLPFactory, ModelTrainerJaxMLP, optax

NET = {
    "layer_sizes": [4, 1],
    "activations": ["tanh", "linear"],
    "train_output_type": "logprob",
}
DEFAULT = {"init_value": 2e-4, "peak_value": 0.02, "end_value": 0.0, "exponent": 1.0}
LR = {"init_value": 1e-5, "peak_value": 1e-3, "end_value": 1e-6, "exponent": 1.0}
TC = {"loss": "huber", "n_epochs": 2}
DL = NS(dataset=NS(input_dim=9, __len__=lambda: 200))  # all create_train_state reads
KEYS = ("init_value", "peak_value", "end_value")


def test_lr_dict_default_and_supplied(monkeypatch):
    net = JaxMLPFactory(network_config=NET, train=True)
    make = lambda tc: ModelTrainerJaxMLP(tc, net, DL, None)  # noqa: E731
    real, seen = optax.warmup_cosine_decay_schedule, []
    monkeypatch.setattr(
        optax,
        "warmup_cosine_decay_schedule",
        lambda **kw: seen.append(kw) or real(**kw),
    )
    for tc, want in ((TC, DEFAULT), ({**TC, "lr_dict": LR}, LR)):
        assert make(tc).lr_dict == want
        make(tc).create_train_state(jax.random.PRNGKey(0))
        assert {k: seen[-1][k] for k in KEYS} == {k: want[k] for k in KEYS}
