# marimo source for the "Train with the JAX backend" documentation how-to.
#
# Regenerate the rendered docs notebook (with outputs) after editing:
#   uv run marimo export ipynb notebooks/basic_tutorial_lan_jax.py \
#     -o docs/basic_tutorial/basic_tutorial_lan_jax.ipynb --include-outputs
#
# Edit interactively with: uv run marimo edit notebooks/basic_tutorial_lan_jax.py

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train with the JAX backend

    This is the JAX/Flax companion to [Train your first LAN
    (PyTorch)](../basic_tutorial_lan_torch/). Complete that learning tutorial
    first: it owns the explanation of LAN training data, configuration, and the
    end-to-end workflow. Here we repeat a deliberately tiny data fixture so this
    how-to remains executable, then focus on the JAX-specific factory, trainer,
    saved state, and inference call.

    The fixture uses [`ssm-simulators`](https://lnccbrown.github.io/ssm-simulators/)
    to generate DDM data. LANfactory can train on compatible data from other
    generators as well.
    """)
    return


@app.cell
def _():
    from copy import deepcopy
    from pathlib import Path

    import lanfactory
    import numpy as np
    import ssms

    return Path, deepcopy, lanfactory, np, ssms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate a small training fixture

    These settings make two small files so the example runs quickly. For real
    training, choose the simulator budget and parameter coverage described in
    the PyTorch learning tutorial and the
    [`ssm-simulators` data-generation documentation](https://lnccbrown.github.io/ssm-simulators/).
    """)
    return


@app.cell
def _(Path, deepcopy, ssms):
    MODEL = "ddm"
    OUT_FOLDER = Path("jax_nb_data") / "training_data"
    MODEL_FOLDER = Path("jax_nb_data") / "jax_models" / "lan"
    N_DATA_FILES = 2
    BATCH_SIZE = 1000
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    generator_config = ssms.config.get_default_generator_config("lan")
    generator_config["model"] = MODEL
    generator_config["pipeline"]["n_parameter_sets"] = 100
    generator_config["pipeline"]["n_cpus"] = 1
    generator_config["simulator"]["n_samples"] = 200
    generator_config["training"]["n_samples_per_param"] = 200
    generator_config["output"]["folder"] = str(OUT_FOLDER)

    model_config = deepcopy(ssms.config.model_config[MODEL])
    return (
        BATCH_SIZE,
        MODEL,
        MODEL_FOLDER,
        N_DATA_FILES,
        OUT_FOLDER,
        generator_config,
        model_config,
    )


@app.cell
def _(N_DATA_FILES, generator_config, model_config, ssms):
    for _i in range(N_DATA_FILES):
        print(f"Generating data file {_i + 1} / {N_DATA_FILES}")
        _generator = ssms.dataset_generators.lan_mlp.TrainingDataGenerator(
            config=generator_config,
            model_config=model_config,
        )
        _generator.generate_data_training(save=True)
    return


@app.cell
def _(deepcopy, lanfactory):
    network_config = deepcopy(lanfactory.config.network_configs.network_config_mlp)
    network_config["layer_sizes"] = [100, 100, 100, 1]
    network_config["activations"] = ["tanh", "tanh", "tanh", "linear"]
    print("Network config:")
    print(network_config)

    train_config = deepcopy(lanfactory.config.network_configs.train_config_mlp)
    train_config["learning_rate"] = 2e-6
    train_config["cpu_batch_size"] = 4096
    train_config["gpu_batch_size"] = 4096
    train_config["n_epochs"] = 2
    print("Train config:")
    print(train_config)
    return network_config, train_config


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reuse the shared data loader

    `make_train_valid_dataloaders` is backend-neutral at this boundary. It:

    - splits your data files into training and validation sets;
    - creates the appropriate `DatasetTorch` objects; and
    - wraps them in PyTorch `DataLoader` objects with sensible defaults.

    The loaders yield NumPy-compatible batches that the JAX trainer consumes.
    """)
    return


@app.cell
def _(BATCH_SIZE, OUT_FOLDER, lanfactory):
    file_list_ = sorted(OUT_FOLDER.glob("*.pickle"))

    # num_workers=0 keeps data loading in-process: ssm-simulators sets the
    # multiprocessing start method to "spawn" at import, which is unsafe for
    # DataLoader worker processes inside a notebook.
    jax_training_dataloader, jax_validation_dataloader, input_dim = lanfactory.trainers.make_train_valid_dataloaders(
        file_ids=file_list_,
        batch_size=BATCH_SIZE,
        network_type="lan",
        train_val_split=0.5,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Training batches: {len(jax_training_dataloader)}")
    print(f"Validation batches: {len(jax_validation_dataloader)}")
    print(f"Input dimension: {input_dim}")
    return jax_training_dataloader, jax_validation_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create the JAX/Flax network

    `JaxMLPFactory` materializes the configured Flax MLP. Passing `train=True`
    prepares it for the training-state lifecycle used by `ModelTrainerJaxMLP`.
    """)
    return


@app.cell
def _(lanfactory, network_config):
    jax_net = lanfactory.trainers.JaxMLPFactory(
        network_config=network_config,
        train=True,
    )
    return (jax_net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train and save the JAX state

    The JAX trainer takes the same configuration and loaders as the PyTorch
    path, but writes a Flax training-state artifact rather than a Torch model.
    """)
    return


@app.cell
def _(
    jax_net,
    jax_training_dataloader,
    jax_validation_dataloader,
    lanfactory,
    train_config,
):
    jax_trainer = lanfactory.trainers.ModelTrainerJaxMLP(
        train_config=train_config,
        model=jax_net,
        train_dl=jax_training_dataloader,
        valid_dl=jax_validation_dataloader,
        pin_memory=False,
    )
    return (jax_trainer,)


@app.cell
def _(MODEL, MODEL_FOLDER, jax_trainer):
    train_state = jax_trainer.train_and_evaluate(
        output_folder=MODEL_FOLDER,
        output_file_id=MODEL,
        run_id="jax",
        mlflow_on=False,
        verbose=1,
        save_outputs=True,
    )
    return (train_state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reload the state for inference

    Recreate the factory with `train=False`, load the saved state, and request a
    JIT-compiled forward function. The input width is the model parameter count
    plus reaction time and response.
    """)
    return


@app.cell
def _(lanfactory, network_config):
    jax_infer = lanfactory.trainers.JaxMLPFactory(
        network_config=network_config,
        train=False,
    )
    return (jax_infer,)


@app.cell
def _(MODEL, MODEL_FOLDER, jax_infer, model_config, train_state):
    # Establish the reactive dependency before reloading the saved state.
    _ = train_state
    _forward_pass, forward_pass_jitted = jax_infer.make_forward_partial(
        seed=42,
        input_dim=model_config["n_params"] + 2,
        state=str(MODEL_FOLDER / f"jax_lan_{MODEL}__train_state.jax"),
        add_jitted=True,
    )
    return (forward_pass_jitted,)


@app.cell
def _(MODEL, deepcopy, forward_pass_jitted, np, ssms):
    import jax.numpy as jnp

    theta = deepcopy(ssms.config.model_config[MODEL]["default_params"])
    sim_out = ssms.basic_simulators.simulator.simulator(
        model=MODEL,
        theta=theta,
        n_samples=50_000,
    )
    input_mat = jnp.zeros((2000, len(theta) + 2))
    for _i in range(len(theta)):
        input_mat = input_mat.at[:, _i].set(jnp.ones(2000) * theta[_i])
    input_mat = input_mat.at[:, len(theta)].set(
        jnp.array(
            np.concatenate(
                [
                    np.linspace(5, 0, 1000, dtype=np.float32),
                    np.linspace(0, 5, 1000, dtype=np.float32),
                ]
            )
        )
    )
    input_mat = input_mat.at[:, len(theta) + 1].set(
        jnp.array(
            np.concatenate(
                [np.repeat(-1.0, 1000), np.repeat(1.0, 1000)]
            ).astype(np.float32)
        )
    )
    net_out = forward_pass_jitted(input_mat)
    return net_out, sim_out


@app.cell
def _(net_out, np, sim_out):
    from matplotlib import pyplot as plt

    plt.plot(
        np.linspace(-5, 5, 2000, dtype=np.float32),
        np.exp(net_out),
        label="JAX LAN likelihood",
    )
    plt.hist(
        sim_out["rts"] * sim_out["choices"],
        bins=100,
        histtype="step",
        fill=None,
        density=True,
        label="simulated choices × RT",
    )
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
