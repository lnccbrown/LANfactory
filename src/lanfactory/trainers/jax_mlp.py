# from ast import Module
import numpy as np
import pandas as pd
import pickle
from functools import partial
from frozendict import frozendict
from typing import Sequence

from lanfactory.utils import try_gen_folder
from time import time

import jax
from jax import numpy as jnp

import flax
from flax.training import train_state
from flax import linen as nn
import optax

try:
    import wandb
except ImportError:
    print("wandb not available")

"""This module contains the JaxMLP class and the ModelTrainerJaxMLP class which 
   are used to train Jax based LANs and CPNs.
"""


def MLPJaxFactory(network_config={}, train=True):
    """Factory function to create a MLPJax object.
    Arguments
    ---------
        network_config (dict):
            Dictionary containing the network configuration.
        train (bool):
            Whether the model should be trained or not.
    Returns
    -------
        MLPJax class initialized with the correct network configuration.
    """

    if isinstance(network_config, str):
        network_config_internal = pickle.load(open(network_config, "rb"))
    elif isinstance(network_config, dict):
        network_config_internal = network_config
    else:
        raise ValueError(
            "network_config argument is not passed as "
            + "either a dictionary or a string (path to a file)!"
        )

    return MLPJax(
        layer_sizes=network_config_internal["layer_sizes"],
        activations=network_config_internal["activations"],
        train_output_type=network_config_internal["train_output_type"],
        train=train,
    )


class MLPJax(nn.Module):
    """JaxMLP class.
    Arguments
    ---------
        layer_sizes (Sequence[int]):
            Sequence of integers containing the sizes of the layers.
        activations (Sequence[str]):
            Sequence of strings containing the activation functions.
        train (bool):
            Whether the model should be set to training mode or not.
        train_output_type (str):
            The output type of the model during training.
    """

    network_type_dict: dict = frozendict({"logprob": "lan", "logits": "cpn"})
    layer_sizes: Sequence[int] = (100, 90, 80, 1)
    activations: Sequence[str] = ("tanh", "tanh", "tanh", "linear")
    train: bool = True
    # if train = False, output applies transform f
    # such that: f(train_output_type) = logprob
    train_output_type: str = "logprob"
    activations_dict = frozendict(
        {"relu": nn.relu, "tanh": nn.tanh, "sigmoid": nn.sigmoid}
    )
    # network_type: Optional[str] = "none"

    # Define network type
    # network_type = "lan" if train_output_type == "logprob" else "cpn"

    def setup(self):
        """Setup function for the JaxMLP class.
        Initializes the layers and activation functions.
        """
        # TODO: Warn if unknown activation string used
        # TODO: Warn if linear activation is used before final layer
        self.layers = [nn.Dense(layer_size) for layer_size in self.layer_sizes]
        self.activation_funs = [
            self.activations_dict[activation]
            for activation in self.activations
            if (activation != "linear")
        ]

        # Identification
        self.network_type = self.network_type_dict[self.train_output_type]

    def __call__(self, inputs):
        """Call function for the JaxMLP class.
        Performs forward pass through the network.

        Arguments
        ---------
            inputs (jax.numpy.ndarray):
                Input tensor.
        Returns
        -------
            jax.numpy.ndarray:
                Output tensor.
        """
        x = inputs

        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != (len(self.layers) - 1):
                x = self.activation_funs[i](x)
            else:
                if self.activations[i] == "linear":
                    pass
                else:
                    x = self.activation_funs[i](x)

        if (not self.train) and (self.train_output_type == "logprob"):
            print("passing through identity")
            x = x  # just for pedagogy
        elif (not self.train) and (self.train_output_type == "logits"):
            print("passing through transform")
            x = -jnp.log((1 + jnp.exp(-x)))
        elif not self.train:
            print("passing through identity 2")
            x = x  # just for pedagogy

        return x

    def load_state_from_file(self, seed=42, input_dim=6, file_path=None):
        """Loads the state dictionary from a file.

        Arguments
        ---------
            seed (int):
                Seed for the random number generator.
            input_dim (int):
                Dimension of the input tensor.
            file_path (str):
                Path to the file containing the state dictionary.

        Returns
        -------
            flax.core.frozen_dict.FrozenDict:
                The state dictionary.
        """

        if file_path is None:
            raise ValueError(
                "file_path argument needs to be speficied! "
                + "(Currently Set to its default: None)"
            )

        with open(file_path, "rb") as file_:
            loaded_state_bytes = file_.read()

        if input_dim is None:
            # flax.serialization.from_bytes wants a reference state,
            # but also works without ....
            loaded_state = flax.serialization.from_bytes(None, loaded_state_bytes)

        else:
            # potentially safer since we provide a reference
            # to flax.serialization.from_bytes
            rng_, key1_ = jax.random.split(jax.random.PRNGKey(42), 2)
            rng_, key2_ = jax.random.split(rng_)

            x = jax.random.uniform(key1_, (1, input_dim))
            state = self.init(key2_, x)

            loaded_state = flax.serialization.from_bytes(state, loaded_state_bytes)
        return loaded_state

    def make_forward_partial(
        self,
        seed=42,
        input_dim=6,
        state=None,
        add_jitted=False,
    ):
        """Creates a partial function for the forward pass of the network.

        Arguments
        ---------
            seed (int):
                Seed for the random number generator.
            input_dim (int):
                Dimension of the input tensor.
            state (flax.core.frozen_dict.FrozenDict):
                The state dictionary (if not loaded from file).
            add_jitted (bool):
                Whether the partial function should be jitted or not.

        Returns
        -------
            Callable:
                The partial function for the forward pass of the network.
        """

        # Load state
        if isinstance(state, str):
            loaded_state = self.load_state_from_file(
                seed=seed, input_dim=input_dim, file_path=state
            )
        elif isinstance(state, dict):
            loaded_state = state
        else:
            raise ValueError("state argument has to be a dictionary or a string!")

        # Make forward pass
        net_forward = partial(self.apply, loaded_state)

        # Jit forward pass
        if add_jitted:
            net_forward_jitted = jax.jit(net_forward)
        else:
            net_forward_jitted = None

        return net_forward, net_forward_jitted


class ModelTrainerJaxMLP:
    def __init__(
        self,
        train_config=None,
        model=None,
        train_dl=None,
        valid_dl=None,
        allow_abs_path_folder_generation=False,
        pin_memory=False,
        seed=None,
    ):
        """Class for training JaxMLP models.

        Arguments
        ---------
            train_config (dict):
                Dictionary containing the training configuration.
            model (MLPJax):
                The MLPJax model to be trained.
            train_dl (torch.utils.data.DataLoader):
                The training data loader.
            valid_dl (torch.utils.data.DataLoader):
                The validation data loader.
            allow_abs_path_folder_generation (bool):
                Whether the folder for the output files should be created or not.
            pin_memory (bool):
                Whether the data loader should pin memory or not.
            seed (int):
                Seed for the random number generator.

        Returns
        -------
            ModelTrainerJaxMLP:
                The ModelTrainerJaxMLP object.

        """
        if "loss_dict" not in train_config.keys():
            self.loss_dict = {
                "huber": {"fun": optax.huber_loss, "kwargs": {"delta": 1}},
                "mse": {"fun": optax.l2_loss, "kwargs": {}},
                "bcelogit": {"fun": optax.sigmoid_binary_cross_entropy, "kwargs": {}},
            }
        else:
            self.loss_dict = train_config["loss_dict"]

        if "lr_dict" not in train_config.keys():
            # Todo: Add more schedules (for now warmup_cosine_decay_schedule)
            self.lr_dict = {
                "init_value": 0.0002,
                "peak_value": 0.02,
                "end_value": 0.0,
                "exponent": 1.0,  # note, exponent currently not used (optax bug)
            }

        self.train_config = train_config
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.pin_memory = pin_memory

        if seed is None:
            self.seed = int(np.random.choice(4000000000))
        else:
            self.seed = seed
        self.allow_abs_path_folder_generation = allow_abs_path_folder_generation
        self.wandb_on = 0

        self.__get_loss()
        self.apply_model_train = self.__make_apply_model(train=True)
        self.apply_model_eval = self.__make_apply_model(train=False)
        self.update_model = self.__make_update_model()

        self.training_history = (
            "Please run training for this attribute to be specified!"
        )
        self.state = "Please run training for this attribute to be specified!"

    def __get_loss(self):
        """Define loss function."""
        self.loss = partial(
            self.loss_dict[self.train_config["loss"]]["fun"],
            **self.loss_dict[self.train_config["loss"]]["kwargs"],
        )

    def __make_apply_model(self, train=True):
        """Compile forward pass with loss aplication"""

        @jax.jit
        def apply_model_core(state, features, labels):
            def loss_fn(params):
                pred = state.apply_fn(params, features)
                loss = self.loss(pred, labels)
                loss = jnp.mean(loss)
                return loss, pred

            if train:
                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss, pred), grads = grad_fn(state.params)
                return grads, loss
            else:
                loss, pred = loss_fn(state.params)
                return loss

        return apply_model_core

    def __make_update_model(self):
        """Compile gradient application"""

        @jax.jit
        def update_model(state, grads):
            return state.apply_gradients(grads=grads)

        return update_model

    def __try_wandb(
        self, wandb_project_id="projectid", file_id="fileid", run_id="runid"
    ):
        """Helper function to initialize wandb

        Arguments
        ---------
            wandb_project_id (str):
                The wandb project id.
            file_id (str):
                The file id.
            run_id (str):
                The run id.

        """
        try:
            wandb.init(
                project=wandb_project_id,
                name=(
                    "wd_"
                    + str(self.train_config["weight_decay"])
                    + "_optim_"
                    + str(self.train_config["optimizer"])
                    + "_"
                    + run_id
                ),
                config=self.train_config,
            )
            self.wandb_on = 1
        except ModuleNotFoundError:
            print("No wandb found, proceeding without logging")

    def create_train_state(self, rng):
        """Create initial train state"""
        params = self.model.init(
            rng, jnp.ones((1, self.train_dl.dataset.input_dim))
        )  # self.train_config['input_size'])))
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.lr_dict["init_value"],
            peak_value=self.lr_dict["peak_value"],
            warmup_steps=self.train_dl.dataset.__len__(),
            decay_steps=self.train_dl.dataset.__len__() * self.train_config["n_epochs"],
            end_value=self.lr_dict["end_value"],
        )
        tx = optax.adam(learning_rate=lr_schedule)
        return train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx
        )

    def run_epoch(self, state, train=True, verbose=1, epoch=0, max_epochs=0):
        """Run one epoch of training or validation
        Arguments
        ---------
            state (flax.core.frozen_dict.FrozenDict):
                The state dictionary.
            train (bool):
                Whether the model should is in training mode or not.
            verbose (int):
                The verbosity level.
            epoch (int):
                The current epoch.
            max_epochs (int):
                The maximum number of epochs.

        Returns
        -------
            tuple (flax.core.frozen_dict.FrozenDict, float):
                The state dictionary and the mean epoch loss.
        """
        if train:
            tmp_dataloader = self.train_dl
            train_str = "Training"
        else:
            tmp_dataloader = self.valid_dl
            train_str = "Validation"

        epoch_loss = []

        cnt_max = tmp_dataloader.__len__()  # total steps per epoch

        # Run training for one epoch
        start_time = time()
        step = 0
        for X, y in tmp_dataloader:
            X_jax = jnp.array(X)
            y_jax = jnp.array(y)

            if train:
                grads, loss = self.apply_model_train(state, X_jax, y_jax)
                state = self.update_model(state, grads)
            else:
                loss = self.apply_model_eval(state, X_jax, y_jax)

            epoch_loss.append(loss)

            # Log wandb and print progress if verbose
            if (step % 100) == 0:
                if self.wandb_on:
                    try:
                        wandb.log({"loss": loss}, step=int(state.step))
                    except ModuleNotFoundError:
                        pass
                if verbose == 2:
                    print(
                        train_str
                        + " - Step: "
                        + str(step)
                        + " of "
                        + str(cnt_max)
                        + " - Loss: "
                        + str(loss)
                    )
                elif verbose == 1:
                    if (step % 1000) == 0:
                        print(
                            train_str
                            + " - Step: "
                            + str(step)
                            + " of "
                            + str(cnt_max)
                            + " - Loss: "
                            + str(loss)
                        )
                else:
                    pass

            step += 1

        end_time = time()
        print(
            "Epoch "
            + str(epoch)
            + "/"
            + str(max_epochs)
            + " time: "
            + str(end_time - start_time)
            + "s"
        )

        mean_epoch_loss = np.mean(epoch_loss)
        return state, mean_epoch_loss

    def train_and_evaluate(
        self,
        output_folder="data/",
        output_file_id="fileid",
        run_id="runid",
        wandb_on=True,
        wandb_project_id="projectid",
        save_history=True,
        save_model=True,
        save_config=True,
        save_all=True,
        save_data_details=True,
        verbose=1,
    ):
        """Train and evaluate JAXMLP model.
        Arguments
        ---------

            output_folder (str):
                Path to the output folder.
            output_file_id (str):
                The file id.
            run_id (str):
                The run id.
            wandb_on (bool):
                Whether to use wandb or not.
            wandb_project_id (str):
                Project id for wandb.
            save_history (bool):
                Whether to save the training history or not.
            save_model (bool):
                Whether to save the model or not.
            save_config (bool):
                Whether to save the training configuration or not.
            save_all (bool):
                Whether to save all files or not.
            save_data_details (bool):
                Whether to save the data details or not.
            verbose (int):
                The verbosity level.
        Returns
        -------
            flax.core.frozen_dict.FrozenDict:
                The final state dictionary (model state).
        """
        try_gen_folder(
            folder=output_folder,
            allow_abs_path_folder_generation=self.allow_abs_path_folder_generation,
        )  # AF-TODO import folder

        if wandb_on:
            self.__try_wandb(
                wandb_project_id=wandb_project_id, file_id=output_file_id, run_id=run_id
            )

        # Identify network type:
        if self.model.train_output_type == "logprob":
            network_type = "lan"
        elif self.model.train_output_type == "logits":
            network_type = "cpn"
        else:
            network_type = "unknown"
            print(
                'Model type identified as "unknown" because '
                "the training_output_type attribute"
                ' of the supplied jax model is neither "logprob", nor "logits"'
            )

        # Initialize Training history
        training_history = pd.DataFrame(
            np.zeros((self.train_config["n_epochs"], 2)), columns=["epoch", "val_loss"]
        )

        # Initialize network
        if not isinstance(self.seed, int):
            raise ValueError(
                "seed argument is not an integer, "
                + "please specift a valid seed to make this code reproducible!"
            )
        else:
            rng = jax.random.PRNGKey(self.seed)

        rng, init_rng = jax.random.split(rng)
        state = self.create_train_state(init_rng)

        # Training loop over epochs
        for epoch in range(self.train_config["n_epochs"]):
            print("Epoch: " + str(epoch) + " of " + str(self.train_config["n_epochs"]))
            state, train_loss = self.run_epoch(
                state,
                train=True,
                verbose=verbose,
                epoch=epoch,
                max_epochs=self.train_config["n_epochs"],
            )

            state, test_loss = self.run_epoch(
                state,
                train=False,
                verbose=verbose,
                epoch=epoch,
                max_epochs=self.train_config["n_epochs"],
            )

            # Collect loss in training history
            training_history.values[epoch, :] = [int(epoch), float(test_loss)]

            print(
                "Epoch: {} / {}, test_loss: {}".format(
                    epoch, self.train_config["n_epochs"], test_loss
                )
            )

        # Set some final attributes
        self.training_history = training_history
        self.state = state

        # Saving
        full_path = (
            output_folder
            + "/"
            + run_id
            + "_"
            + network_type
            + "_"
            + output_file_id
            + "_"
        )

        if save_history or save_all:
            training_history_path = full_path + "_jax_training_history.csv"
            training_history.to_csv(training_history_path)
            print("Saving training history to: " + training_history_path)

        if save_model or save_all:
            # Serialize parameter state
            byte_output = flax.serialization.to_bytes(state.params)

            # Write to file
            train_state_path = full_path + "_train_state.jax"
            file = open(train_state_path, "wb")
            file.write(byte_output)
            file.close()
            print("Saving model parameters to: " + train_state_path)

        if save_config or save_all:
            config_path = full_path + "_train_config.pickle"
            pickle.dump(self.train_config, open(config_path, "wb"))
            print("Saving training config to: " + config_path)

        if save_data_details or save_all:
            data_details_path = full_path + "_data_details.pickle"
            pickle.dump(
                {
                    "train_data_generator_config": self.train_dl.dataset.data_generator_config,
                    "train_data_file_ids": self.train_dl.dataset.file_ids,
                    "valid_data_generator_config": self.valid_dl.dataset.data_generator_config,
                    "valid_data_file_ids": self.valid_dl.dataset.file_ids,
                },
                open(data_details_path, "wb"),
            )
            print("Saving training data details to: " + data_details_path)

        return state
