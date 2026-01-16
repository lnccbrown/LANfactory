"""This module contains the function to transform Torch/Jax models to ONNX format.
Can be run as a script.
"""

import pickle

import torch
import typer

from lanfactory.trainers.torch_mlp import TorchMLP


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
        network_config_mlp = pickle.load(f)

    mynet = TorchMLP(
        network_config=network_config_mlp,
        input_shape=input_shape,
    )

    mynet.load_state_dict(
        torch.load(state_dict_file, map_location=torch.device("cpu")),
    )

    x = torch.randn(1, input_shape, requires_grad=True)
    torch.onnx.export(
        mynet,
        x,
        output_onnx_file,
        dynamo=False,  # disable dynamo for now to support older ONNX versions
    )


app = typer.Typer()


def option_no_default(help: str) -> typer.Option:
    return typer.Option(..., help=help, show_default=False)


@app.command()
def main(
    network_config_file: str = option_no_default(
        "Path to the network configuration file (pickle)."
    ),
    state_dict_file: str = option_no_default("Path to the state dictionary file."),
    input_shape: int = option_no_default("Size of the input tensor for the model."),
    output_onnx_file: str = option_no_default("Path to the output ONNX file."),
):
    """
    Convert a TorchMLP model to ONNX format.
    """
    transform_to_onnx(
        network_config_file,
        state_dict_file,
        input_shape,
        output_onnx_file,
    )


if __name__ == "__main__":
    app()
