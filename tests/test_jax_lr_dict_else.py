"""ModelTrainerJaxMLP uses a supplied lr_dict, else the default."""

from lanfactory.trainers.jax_mlp import JaxMLPFactory, ModelTrainerJaxMLP

NET = {"layer_sizes": [4, 1], "activations": ["tanh", "linear"],
       "train_output_type": "logprob"}
DEFAULT = {"init_value": 2e-4, "peak_value": 0.02, "end_value": 0.0, "exponent": 1.0}
LR = {"init_value": 1e-5, "peak_value": 1e-3, "end_value": 0.0, "exponent": 1.0}


def test_lr_dict_default_and_supplied():
    net = JaxMLPFactory(network_config=NET, train=True)
    make = lambda tc: ModelTrainerJaxMLP(tc, net, None, None)  # noqa: E731
    assert make({"loss": "huber"}).lr_dict == DEFAULT
    assert make({"loss": "huber", "lr_dict": LR}).lr_dict == LR
