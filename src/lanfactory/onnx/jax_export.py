"""Export jaxtrain networks to ONNX. Can be run as a script.

Closes the gap that made jaxtrain a dead end for HSSM: the jax trainer saves
flax parameter bytes (``*_train_state.jax``) which nothing downstream could
convert — ``transform-onnx`` only reads torch state dicts, and HSSM consumes
ONNX exclusively.

The export follows the ecosystem's single-trial ONNX contract; the canonical
statement lives in HSSM's docs:
https://lnccbrown.github.io/HSSM/how_to/custom_onnx_likelihoods/
The graph is traced with a **concrete** ``(1, input_dim)`` dummy
and no dynamic axes, exactly like the torch MLP exporter and the production
networks on franklab/HSSM. Every dim is static, so HSSM's load-time check
passes, and HSSM's rank-1-per-trial + ``jax.vmap`` consumption works because
the resulting graph is pure ``Gemm`` + elementwise activations (verified
end-to-end against ``hssm.make_jax_func``). Rank-1 ``(input_dim,)`` tracing is
NOT used here: jax2onnx lowers ``nn.Dense`` to ``Gemm``, whose ONNX spec
requires rank-2 inputs — a rank-1-traced graph is rejected by onnxruntime.
(The sbi/bayesflow exporters trace rank-1 because torch lowers Linear to
rank-agnostic MatMul+Add; different tracer, different constraint.)

The exported graph is the **eval-mode** forward: for LANs (``logprob``) this
equals the raw head, and for CPN/OPN (``logits``) it applies the logsigmoid
transform ``-log(1 + exp(-x))`` so the network emits log choice probabilities.
This matches every torch export path (``_save_onnx`` calls ``.eval()``, and
``transform-onnx`` exports under ``torch.onnx``'s EVAL default) and is what
HSSM consumes as element-wise log-likelihood. Exporting the raw training head
for logits networks would silently corrupt every downstream logp by
``+log(1 + exp(-logit))``.
"""

import pickle

import typer

# Opset matching the sbi/bayesflow exporters. The MLPs only need Gemm +
# elementwise activations, all ancient; a newer default (jax2onnx uses 23)
# would just narrow runtime compatibility (jaxonnxruntime) for no benefit.
DEFAULT_OPSET = 17


def export_forward_to_onnx(
    forward,
    input_shape: int,
    output_onnx_file: str,
    model_name: str = "jax_mlp",
    opset: int = DEFAULT_OPSET,
) -> None:
    """Export a jax forward function to ONNX under the single-trial contract.

    The shared core for both the file-based CLI (``transform_jax_to_onnx``)
    and the jax trainer's post-training export. ``forward`` must accept an
    input of shape ``(1, input_shape)`` (flax Dense handles arbitrary leading
    dims, so the trainers' forward functions qualify unchanged).
    """
    import jax.numpy as jnp
    from jax2onnx import to_onnx

    import onnx as onnx_lib

    model_proto = to_onnx(
        forward,
        # Concrete (1, D) dummy: every dim static — the load-bearing line of
        # the ecosystem contract. See the module docstring for why (1, D)
        # rather than rank-1 (Gemm requires rank 2).
        inputs=[jnp.zeros((1, input_shape), dtype=jnp.float32)],
        model_name=model_name,
        opset=opset,
    )
    onnx_lib.save_model(model_proto, output_onnx_file)


def transform_jax_to_onnx(
    network_config_file: str,
    state_file: str,
    input_shape: int,
    output_onnx_file: str,
    opset: int = DEFAULT_OPSET,
) -> None:
    """Transform a trained JaxMLP to ONNX format.

    Arguments
    ---------
        network_config_file (str):
            Path to the pickle file containing the network configuration
            (``layer_sizes``, ``activations``, ``train_output_type``).
        state_file (str):
            Path to the ``*_train_state.jax`` file written by the jax trainer
            (flax ``to_bytes`` serialization of the parameters).
        input_shape (int):
            The size of the single-trial input vector for the model
            (``n_params + 2`` for LANs).
        output_onnx_file (str):
            Path to the output ONNX file.
        opset (int):
            ONNX opset version to target.
    """
    from lanfactory.trainers import JaxMLPFactory

    with open(network_config_file, "rb") as f:
        network_config = pickle.load(f)

    # train=False: export the EVAL head. For logprob networks this equals the
    # raw head; for logits networks it applies logsigmoid, matching the torch
    # exporters and HSSM's log-likelihood consumption (see module docstring).
    net = JaxMLPFactory(network_config=network_config, train=False)
    forward, _ = net.make_forward_partial(
        input_dim=input_shape,
        state=state_file,
        add_jitted=False,
    )

    export_forward_to_onnx(
        forward,
        input_shape=input_shape,
        output_onnx_file=output_onnx_file,
        model_name=str(network_config.get("network_type", "jax_mlp")),
        opset=opset,
    )


app = typer.Typer()


def option_no_default(help: str) -> typer.Option:
    return typer.Option(..., help=help, show_default=False)


@app.command()
def main(
    network_config_file: str = option_no_default(
        "Path to the network configuration file (pickle)."
    ),
    state_file: str = option_no_default(
        "Path to the *_train_state.jax file (flax parameter bytes)."
    ),
    input_shape: int = option_no_default("Size of the input tensor for the model."),
    output_onnx_file: str = option_no_default("Path to the output ONNX file."),
    opset: int = typer.Option(DEFAULT_OPSET, help="ONNX opset version to target."),
):
    """Convert a jaxtrain-produced JaxMLP to ONNX format."""
    transform_jax_to_onnx(
        network_config_file,
        state_file,
        input_shape,
        output_onnx_file,
        opset=opset,
    )


if __name__ == "__main__":
    app()
