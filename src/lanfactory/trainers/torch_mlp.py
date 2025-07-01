"""This module contains the classes for training TorchMLP models."""

import numpy as np
import pandas as pd
import pickle
from typing import Callable
from time import time
import logging
from pathlib import Path


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    print("wandb not available")

logger = logging.getLogger(__name__)


class DatasetTorch(torch.utils.data.Dataset):
    """Dataset class for TorchMLP training.

    Arguments
    ----------
        file_ids (list[str]):
            List of paths to the data files.
        batch_size (int):
            Batch size.
        label_lower_bound (float | None):
            Lower bound for the labels.
        label_upper_bound (float | None):
            Upper bound for the labels.
        features_key (str):
            Key for the features in the data files.
        label_key (str):
            Key for the labels in the data files.
        out_framework (str):
            Output framework.
    """

    def __init__(
        self,
        file_ids: list[str] | list[Path],
        batch_size: int = 32,
        label_lower_bound: float | None = None,
        label_upper_bound: float | None = None,
        features_key: str = "data",
        label_key: str = "labels",
        out_framework: str = "torch",
    ) -> None:
        # AF-TODO: Take device into account at this level,
        # this currently happens only in the training loop
        # Initialization
        self.batch_size = batch_size
        self.file_ids = file_ids
        self.indexes = np.arange(len(self.file_ids))
        self.label_upper_bound = label_upper_bound
        self.label_lower_bound = label_lower_bound
        self.features_key = features_key
        self.label_key = label_key
        self.out_framework = out_framework
        self.data_generator_config: str = "None"

        self.tmp_data: dict = {}

        # get metadata from loading a test file
        self.__init_file_shape()

    def __len__(self) -> int:
        # Number of batches per epoch
        return (
            len(self.file_ids) * ((self.file_shape_dict["inputs"][0] // self.batch_size) * self.batch_size)
        ) // self.batch_size

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        # Check if it is time to load the next file
        if ((index % self.batches_per_file) == 0) or (self.tmp_data == {}):
            self.__load_file(file_index=self.indexes[index // self.batches_per_file])

        # Generate and return a batch
        batch_ids = np.arange(
            ((index % self.batches_per_file) * self.batch_size),
            ((index % self.batches_per_file) + 1) * self.batch_size,
            1,
        )
        X, y = self.__data_generation(batch_ids)
        return X, y

    def __load_file(self, file_index: int) -> None:
        # Load file and shuffle the indices
        self.tmp_data = pickle.load(open(self.file_ids[file_index], "rb"))
        shuffle_idx = np.random.choice(
            self.tmp_data[self.features_key].shape[0],
            size=self.tmp_data[self.features_key].shape[0],
            replace=True,
        )
        self.tmp_data[self.features_key] = self.tmp_data[self.features_key][shuffle_idx, :]
        self.tmp_data[self.label_key] = self.tmp_data[self.label_key][shuffle_idx]
        return

    def __init_file_shape(self) -> None:
        # Function gets dimensionalities form a test data file
        init_file = pickle.load(open(self.file_ids[0], "rb"))
        self.file_shape_dict = {
            "inputs": init_file[self.features_key].shape,
            "labels": init_file[self.label_key].shape,
        }
        self.batches_per_file = int(self.file_shape_dict["inputs"][0] / self.batch_size)
        self.input_dim = self.file_shape_dict["inputs"][1]

        if "generator_config" in init_file.keys():
            self.data_generator_config = init_file["generator_config"]

        if len(self.file_shape_dict["labels"]) > 1:
            self.label_dim = self.file_shape_dict["labels"][1]
        else:
            self.label_dim = 1
        return

    def __data_generation(self, batch_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        # Generates data containing batch_size samples
        X = self.tmp_data[self.features_key][batch_ids, :]
        if self.tmp_data[self.label_key].ndim == 1:
            y = np.expand_dims(self.tmp_data[self.label_key][batch_ids], axis=1)
        elif self.tmp_data[self.label_key].ndim == 2:
            y = self.tmp_data[self.label_key][batch_ids]
        else:
            bad_shape = str(self.tmp_data[self.label_key].shape)
            raise ValueError(f"Label data has unexpected shape: {bad_shape}")

        if self.label_lower_bound is not None:
            y[y < self.label_lower_bound] = self.label_lower_bound

        if self.label_upper_bound is not None:
            y[y > self.label_upper_bound] = self.label_upper_bound

        return X, y


class TorchMLP(nn.Module):
    """TorchMLP class.

    Arguments
    ----------
        network_config (dict):
            Network configuration.
        input_shape (int):
            Input shape.
        network_type (str):
            Network type.
    """

    # AF-TODO: Potentially split this via super-class
    # In the end I want 'eval', but differentiable
    # w.r.t to input ...., might be a problem
    def __init__(
        self,
        network_config: dict,
        input_shape: int = 10,
        network_type: str | None = None,
        **kwargs,
    ) -> None:
        super(TorchMLP, self).__init__()

        self.input_shape = input_shape
        self.network_config = network_config

        if "train_output_type" in self.network_config.keys():
            self.train_output_type = self.network_config["train_output_type"]
        else:
            self.train_output_type = "logprob"

        if network_type is not None:
            self.network_type = network_type
        else:
            self.network_type = "lan" if self.train_output_type == "logprob" else "cpn"
            print(
                'Setting network type to "lan" or "cpn" based on train_output_type. \n'
                + "Note: This is only a default setting, and can be overwritten by the network_type argument."
            )

        self.activations: dict[str, nn.Module] = {
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "sigmoid": torch.nn.Sigmoid(),
        }

        # Build the network ------
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_shape, self.network_config["layer_sizes"][0]))
        self.layers.append(self.activations[self.network_config["activations"][0]])
        print(self.network_config["activations"][0])
        for i in range(len(self.network_config["layer_sizes"]) - 1):
            self.layers.append(
                nn.Linear(
                    self.network_config["layer_sizes"][i],
                    self.network_config["layer_sizes"][i + 1],
                )
            )
            print(self.network_config["activations"][i + 1])
            if i < (len(self.network_config["layer_sizes"]) - 2):
                # activations until last hidden layer are always applied
                self.layers.append(self.activations[self.network_config["activations"][i + 1]])
            elif len(self.network_config["activations"]) >= len(self.network_config["layer_sizes"]) - 1:
                # apply output activation if supplied
                # e.g. classification network
                if self.network_config["activations"][i + 1] != "linear":
                    self.layers.append(self.activations[self.network_config["activations"][i + 1]])
                else:
                    pass
            else:
                # skip output activation if no activation for last layer is provided
                # e.g. regression network
                pass

        self.len_layers = len(self.layers)
        # -----------------------

    # Define forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.

        Arguments
        ---------
            x (torch.Tensor):
                Input tensor.

        Returns
        -------
            torch.Tensor:
                Output tensor.
        """
        for i in range(self.len_layers - 1):
            x = self.layers[i](x)
        if self.training or self.train_output_type == "logprob":
            return self.layers[-1](x)
        elif self.train_output_type == "logits":
            return -torch.log(
                (1 + torch.exp(-self.layers[-1](x)))
            )  # log ( 1 / (1 + exp(-x))), where x = log(p / (1 - p))
        else:
            return self.layers[-1](x)


class ModelTrainerTorchMLP:
    def __init__(
        self,
        model: TorchMLP,
        train_config: dict | str | Path,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        allow_abs_path_folder_generation: bool = False,
        pin_memory: bool = True,
        seed: int | None = None,
    ) -> None:
        """Class to train Torch Models.
        Arguments
        ---------
            train_config (dict):
                Training configuration.
            model (TorchMLP):
                TorchMLP model.
            train_dl (DatasetTorch):
                Training dataloader.
            valid_dl (DatasetTorch):
                Validation dataloader.
            allow_abs_path_folder_generation (bool):
                Whether to allow absolute path folder generation.
            pin_memory (bool):
                Whether to pin memory (dataloader). Can affect speed.
            seed (int):
                Random seed.
        """
        torch.backends.cudnn.benchmark = True
        self.dev: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Torch Device: {self.dev}")
        if train_config is None:
            raise ValueError("train_config is passed as None")
        elif isinstance(train_config, str | Path):
            print("train_config is passed as string or path: \n", train_config)
            try:
                logger.info("Trying to load string as path to pickle file: ")
                self.train_config: dict = pickle.load(open(train_config, "rb"))
            except (OSError, pickle.PickleError) as e:
                logger.error(f"Error loading training config from file {train_config}: {str(e)}")
                raise
        elif isinstance(train_config, dict):
            print("train_config is passed as dictionary: \n")
            print(train_config)
            self.train_config: dict = train_config

        self.model: TorchMLP = model.to(self.dev)
        self.allow_abs_path_folder_generation: bool = allow_abs_path_folder_generation
        self.train_dl: DataLoader = train_dl
        self.valid_dl: DataLoader = valid_dl
        self.pin_memory: bool = pin_memory

        # Initialize training parameters with defaults
        self.loss_fun: Callable = F.mse_loss
        self.optimizer: optim.Optimizer = optim.Adam(
            self.model.parameters(),
            weight_decay=self.train_config["weight_decay"],
            lr=self.train_config["learning_rate"],
        )
        self.scheduler: optim.lr_scheduler.ExponentialLR | optim.lr_scheduler.ReduceLROnPlateau | None = None

        # Override default training parameters with user-provided values
        self.__get_loss()
        self.__get_optimizer()
        self.__get_scheduler()
        # self.__load_weights()

    def __try_wandb(
        self,
        wandb_project_id: str = "projectid",
        run_id: str = "runid",
    ) -> None:
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
            logger.info("Succefully initialized wandb!")
        except Exception as e:
            logger.error(f"Unexpected wandb error: {e}")
            logger.info("wandb not available, not storing results there")

    def __get_loss(self) -> None:
        if self.train_config["loss"] == "huber":
            self.loss_fun = F.huber_loss
        elif self.train_config["loss"] == "mse":
            self.loss_fun = F.mse_loss
        elif self.train_config["loss"] == "bce":
            self.loss_fun = F.binary_cross_entropy
        elif self.train_config["loss"] == "bcelogit":
            self.loss_fun = F.binary_cross_entropy_with_logits

    def __get_optimizer(self) -> None:
        if self.train_config["optimizer"] not in ["adam", "sgd"]:
            raise ValueError(f"Optimizer {self.train_config['optimizer']} not supported yet")
        elif self.train_config["optimizer"] == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                weight_decay=self.train_config["weight_decay"],
                lr=self.train_config["learning_rate"],
            )

    def __get_scheduler(self) -> None:
        # Add scheduler if scheduler option supplied
        if self.train_config["lr_scheduler"] is not None:
            if self.train_config["lr_scheduler"] == "reduce_on_plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=(
                        self.train_config["lr_scheduler_params"]["factor"]
                        if "factor" in self.train_config["lr_scheduler_params"].keys()
                        else 0.1
                    ),
                    patience=(
                        self.train_config["lr_scheduler_params"]["patience"]
                        if "patience" in self.train_config["lr_scheduler_params"].keys()
                        else 2
                    ),
                    threshold=(
                        self.train_config["lr_scheduler_params"]["threshold"]
                        if "threshold" in self.train_config["lr_scheduler_params"].keys()
                        else 0.001
                    ),
                    threshold_mode="rel",
                    cooldown=0,
                    min_lr=(
                        self.train_config["lr_scheduler_params"]["min_lr"]
                        if "min_lr" in self.train_config["lr_scheduler_params"].keys()
                        else 0.00000001
                    ),
                )
            elif self.train_config["lr_scheduler"] == "multiply":
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=(
                        self.train_config["lr_scheduler_params"]["factor"]
                        if "factor" in self.train_config["lr_scheduler_params"].keys()
                        else 0.1
                    ),
                    last_epoch=-1,
                )

    def __load_weights(self) -> None:
        # raise NotImplementedError
        # for warmstart, not implemented at the moment
        return

    def train_and_evaluate(
        self,
        output_folder: str | Path = "data/",
        output_file_id: str = "fileid",
        run_id: str = "runid",
        wandb_on: bool = True,
        wandb_project_id: str = "projectid",
        save_outputs: bool = True,
        verbose: int = 1,
    ) -> None:
        """Train and evaluate the model.

        Arguments
        ---------
            output_folder (str):
                Output folder.
            output_file_id (str):
                Output file ID.
            run_id (str):
                Run ID.
            wandb_on (bool):
                Whether to use wandb.
            wandb_project_id (str):
                Wandb project ID.
            save_outputs (bool):
                Whether to save all outputs.
            verbose (int):
                Verbosity level.
        """

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # try_gen_folder(
        #     folder=output_folder,
        #     allow_abs_path_folder_generation=self.allow_abs_path_folder_generation,
        # )  # AF-TODO import folder

        if wandb_on:
            self.__try_wandb(wandb_project_id=wandb_project_id, run_id=run_id)

        training_history: pd.DataFrame = pd.DataFrame(
            np.zeros((self.train_config["n_epochs"], 2)), columns=["epoch", "val_loss"]
        )

        if wandb_on:
            try:
                wandb.watch(self.model, criterion=None, log="all", log_freq=1000)
            except Exception as e:
                print(e)

        step_cnt = 0
        self.model.train()
        for epoch in range(self.train_config["n_epochs"]):
            cnt = 0
            epoch_s_t = time()

            # Training loop
            for xb, yb in self.train_dl:
                # Shift data to device
                if self.pin_memory and str(self.dev) == "cuda":
                    xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
                else:
                    xb, yb = xb.to(self.dev), yb.to(self.dev)

                # Forward pass
                pred = self.model(xb)
                loss = self.loss_fun(pred, yb)

                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Log training progress
                self._log_training_progress(epoch, cnt, loss, verbose)

                cnt += 1
                step_cnt += 1

            print(f"Epoch took {epoch} / {self.train_config['n_epochs']}, took {time() - epoch_s_t} seconds")

            # Start validation
            with torch.no_grad():
                val_loss: torch.Tensor = sum(
                    self.loss_fun(self.model(xb.to(self.dev)), yb.to(self.dev)) for xb, yb in self.valid_dl
                ) / len(self.valid_dl)

            # Print validation loss
            print(f"epoch {epoch} / {self.train_config['n_epochs']}, validation_loss: {val_loss:2.4}")

            # Scheduler step:
            if self.train_config["lr_scheduler"] is not None:
                if self.train_config["lr_scheduler"] == "reduce_on_plateau":
                    self.scheduler.step(val_loss)
                elif self.train_config["lr_scheduler"] == "multiply":
                    self.scheduler.step()

            # Append training history
            training_history.values[epoch, :] = [epoch, val_loss.cpu()]

            # Log wandb if possible
            self._log_wandb(wandb_on, loss, val_loss, step_cnt)

        if save_outputs:
            partial_path_str = str(output_folder / f"{output_file_id}_{self.model.network_type}_{run_id}")

            self._save_training_history(training_history, partial_path_str + "_training_history.csv")
            self._save_model(self.model, partial_path_str + "_train_state_dict.pt")
            self._save_config(self.train_config, partial_path_str + "_train_config.pickle")
            self._save_data_details(self.train_dl, self.valid_dl, partial_path_str + "_data_details.pickle")
            self._save_onnx(self.model, self.dev, partial_path_str + "_model.onnx")

        self._close_wandb(wandb_on)
        logger.info("Training finished successfully...")

    def _log_training_progress(self, epoch: int, cnt: int, loss: torch.Tensor, verbose: int) -> None:
        """Log training progress based on verbosity level."""
        if verbose == 0:
            return

        # Define logging intervals based on verbosity
        log_intervals = {
            1: 100,  # Log every 100 batches
            2: 1000,  # Log every 1000 batches
        }

        interval = log_intervals.get(verbose, 0)
        if interval > 0 and cnt % interval == 0:
            message = (
                f"epoch: {epoch}/{self.train_config['n_epochs']}, "
                f"batch: {cnt}/{self.train_dl.__len__()}, "
                f"batch_loss: {loss:.4f}"
            )
            print(message)

    @staticmethod
    def _close_wandb(wandb_on: bool) -> None:
        if wandb_on:
            try:
                wandb.finish()
                logger.info("wandb uploaded")
            except Exception as e:
                logger.error("Unexpected wandb error: {}".format(e))
        else:
            logger.info("wandb not available, nothing to close")

    @staticmethod
    def _log_wandb(wandb_on: bool, loss: torch.Tensor, val_loss: torch.Tensor, step_cnt: int) -> None:
        if wandb_on:
            try:
                wandb.log({"loss": loss, "val_loss": val_loss}, step=step_cnt)
            except Exception as e:
                logger.error("Unexpected wandb error: {}".format(e))

    @staticmethod
    def _save_training_history(training_history: pd.DataFrame, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(training_history).to_csv(path)
        logger.info(f"Saving training history to: {path}")

    @staticmethod
    def _save_model(model: TorchMLP, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        logger.info(f"Saving model to: {path}")

    @staticmethod
    def _save_config(train_config: dict, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(train_config, f)
        logger.info(f"Saving config to: {path}")

    @staticmethod
    def _save_data_details(train_dl: DataLoader, valid_dl: DataLoader, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "train_data_generator_config": train_dl.dataset.data_generator_config,
                    "train_datafile_ids": train_dl.dataset.file_ids,
                    "valid_data_generator_config": valid_dl.dataset.data_generator_config,
                    "valid_datafile_ids": valid_dl.dataset.file_ids,
                },
                f,
            )
        logger.info(f"Saving data details to: {path}")

    @staticmethod
    def _save_onnx(model: TorchMLP, dev: torch.device, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model.eval()
        torch.onnx.export(
            model,
            torch.randn(1, model.input_shape).to(dev),
            path,
        )
        logger.info(f"Saving model to ONNX format to: {path}")


class LoadTorchMLPInfer:
    """Class to load TorchMLP models for inference. (This
    was originally useful directly for application in the
    HDDM toolbox).

    Arguments
    ---------
        model_file_path (str):
            Path to the model file.
        network_config (dict):
            Network configuration.
        input_dim (int):
            Input dimension.

    """

    def __init__(
        self,
        model_file_path: str | None = None,
        network_config: dict | str | None = None,
        input_dim: int | None = None,
        network_type: str | None = None,
    ) -> None:
        if input_dim is None:
            raise ValueError("input_dim is required")
        if model_file_path is None:
            raise ValueError("model_file_path is required")

        self.model_file_path = model_file_path
        torch.backends.cudnn.benchmark = True
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if isinstance(network_config, str):
            with open(network_config, "rb") as f:
                self.network_config = pickle.load(f)
        elif isinstance(network_config, dict):
            self.network_config = network_config
        else:
            raise ValueError("network config is neither a string nor a dictionary")

        self.input_dim = input_dim

        self.net = TorchMLP(
            network_config=self.network_config,
            input_shape=self.input_dim,
            generative_model_id=None,
            network_type=network_type,
        )
        if not torch.cuda.is_available():
            self.net.load_state_dict(torch.load(self.model_file_path, map_location=torch.device("cpu")))
        else:
            self.net.load_state_dict(torch.load(self.model_file_path))
        self.net.to(self.dev)
        self.net.eval()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_on_batch(self, x: np.ndarray | None = None) -> np.ndarray:
        """
        Intended as function that computes trial wise log-likelihoods
        from a matrix input.
        To be used primarily through the HDDM toolbox.

        Arguments
        ---------
            x (numpy.ndarray(dtype=numpy.float32)):
                Matrix which will be passed through the network.
                LANs expect the matrix columns to follow a specific order.
                When used in HDDM, x will be passed as follows.
                The first few columns are trial wise model parameters
                (order specified in the model_config file under the 'params' key).
                The last two columns are filled with trial wise
                reaction times and choices.
                When not used via HDDM, no such restriction applies.
        Output
        ------
            numpy.ndarray(dtype = numpy.float32):
                Output of the network. When called through HDDM,
                this is expected as trial-wise log likelihoods
                of a given generative model.

        """
        return self.net(torch.from_numpy(x).to(self.dev)).cpu().numpy()


class LoadTorchMLP:
    """Class to load TorchMLP models.

    Arguments
    ---------
        model_file_path (str):
            Path to the model file.
        network_config (dict):
            Network configuration.
        input_dim (int):
            Input dimension."""

    def __init__(
        self,
        model_file_path: str,
        network_config: dict | str,
        input_dim: int,
    ) -> None:
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_file_path = model_file_path

        # Load network config from pickle file if string path provided
        if isinstance(network_config, str):
            with open(network_config, "rb") as f:
                self.network_config = pickle.load(f)
        else:
            self.network_config = network_config

        self.input_dim = input_dim

        self.net = TorchMLP(
            network_config=self.network_config,
            input_shape=self.input_dim,
            generative_model_id=None,
        )
        if not torch.cuda.is_available():
            self.net.load_state_dict(torch.load(self.model_file_path, map_location=torch.device("cpu")))
        else:
            self.net.load_state_dict(torch.load(self.model_file_path))
        self.net.to(self.dev)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_on_batch(self, x: np.ndarray | None = None) -> np.ndarray:
        """Makes predictions on a batch of data.

        Args:
            x: Input data as numpy array.

        Returns:
            numpy.ndarray: Model predictions as numpy array.

        Note:
            Input data is automatically converted to torch tensor and moved to the
            appropriate device (CPU/GPU). Output is converted back to numpy array
            on CPU.
        """
        return self.net(torch.from_numpy(x).to(self.dev)).cpu().numpy()
