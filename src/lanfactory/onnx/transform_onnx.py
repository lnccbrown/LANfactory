import pickle
from typing import Any
import argparse

import torch
from lanfactory.trainers.torch_mlp import TorchMLP

"""This module contains the function to transform Torch/Jax models to ONNX format.
Can be run as a script.
"""


def transform_to_onnx(
    network_config_file: str,
    state_dict_file: str,
    input_shape: int,
    output_onnx_file: str,
) -> None:
    """
    Transforms a TorchMLP model to ONNX format.

    Arguments
    ---------
        network_config_file (str):
            Path to the pickle file containing the network configuration.
        state_dict_file (str):
            Path to the file containing the state dictionary of the model.
        input_shape (int):
            The size of the input tensor for the model.
        output_onnx_file (str):
            Path to the output ONNX file.
    """
    with open(network_config_file, "rb") as f:
        network_config_mlp: Any = pickle.load(f)

    mynet = TorchMLP(
        network_config=network_config_mlp,
        input_shape=input_shape,
        generative_model_id=None,
    )

    mynet.load_state_dict(
        torch.load(state_dict_file, map_location=torch.device("cpu")),
    )

    x = torch.randn(1, input_shape, requires_grad=True)
    torch.onnx.export(mynet, x, output_onnx_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a TorchMLP model to ONNX format."
    )
    parser.add_argument(
        "network_config_file", help="Path to the network configuration file (pickle)."
    )
    parser.add_argument("state_dict_file", help="Path to the state dictionary file.")
    parser.add_argument(
        "input_shape", type=int, help="Size of the input tensor for the model."
    )
    parser.add_argument("output_onnx_file", help="Path to the output ONNX file.")

    args = parser.parse_args()

    transform_to_onnx(
        args.network_config_file,
        args.state_dict_file,
        args.input_shape,
        args.output_onnx_file,
    )
