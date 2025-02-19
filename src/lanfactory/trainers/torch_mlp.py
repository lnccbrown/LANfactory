import numpy as np
import pandas as pd
import pickle

from lanfactory.utils import try_gen_folder
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    print("wandb not available")

"""This module contains the classes for training TorchMLP models."""


class DatasetTorch(torch.utils.data.Dataset):
    """Dataset class for TorchMLP training.

    Arguments
    ----------
        file_ids (list):
            List of paths to the data files.
        batch_size (int):
            Batch size.
        label_lower_bound (float):
            Lower bound for the labels.
        label_upper_bound (float):
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
        file_ids,
        batch_size=32,
        label_lower_bound=None,
        label_upper_bound=None,
        features_key="data",
        label_key="labels",
        out_framework="torch",
    ):
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
        self.data_generator_config = "None"

        self.tmp_data = None

        # get metadata from loading a test file
        self.__init_file_shape()

    def __len__(self):
        # Number of batches per epoch
        return int(
            np.floor(
                (
                    len(self.file_ids)
                    * (
                        (self.file_shape_dict["inputs"][0] // self.batch_size)
                        * self.batch_size
                    )
                )
                / self.batch_size
            )
        )

    def __getitem__(self, index):
        # Check if it is time to load the next file
        if ((index % self.batches_per_file) == 0) or (self.tmp_data is None):
            self.__load_file(file_index=self.indexes[index // self.batches_per_file])

        # Generate and return a batch
        batch_ids = np.arange(
            ((index % self.batches_per_file) * self.batch_size),
            ((index % self.batches_per_file) + 1) * self.batch_size,
            1,
        )
        X, y = self.__data_generation(batch_ids)
        return X, y

    def __load_file(self, file_index):
        # Load file and shuffle the indices
        self.tmp_data = pickle.load(open(self.file_ids[file_index], "rb"))
        shuffle_idx = np.random.choice(
            self.tmp_data[self.features_key].shape[0],
            size=self.tmp_data[self.features_key].shape[0],
            replace=True,
        )
        self.tmp_data[self.features_key] = self.tmp_data[self.features_key][
            shuffle_idx, :
        ]
        self.tmp_data[self.label_key] = self.tmp_data[self.label_key][shuffle_idx]
        return

    def __init_file_shape(self):
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

    def __data_generation(self, batch_ids=None):
        # Generates data containing batch_size samples
        X = self.tmp_data[self.features_key][batch_ids, :]
        if self.tmp_data[self.label_key].ndim == 1:
            y = np.expand_dims(self.tmp_data[self.label_key][batch_ids], axis=1)
        elif self.tmp_data[self.label_key].ndim == 2:
            y = self.tmp_data[self.label_key][batch_ids]
        else:
            raise ValueError(
                "Label data has unexpected shape: "
                + str(self.tmp_data[self.label_key].shape)
            )

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
        network_config=None,
        input_shape=10,
        network_type=None,
        **kwargs,
    ):
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

        self.activations = {
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "sigmoid": torch.nn.Sigmoid(),
        }

        # Build the network ------
        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Linear(input_shape, self.network_config["layer_sizes"][0])
        )
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
                self.layers.append(
                    self.activations[self.network_config["activations"][i + 1]]
                )
            elif (
                len(self.network_config["activations"])
                >= len(self.network_config["layer_sizes"]) - 1
            ):
                # apply output activation if supplied
                # e.g. classification network
                if self.network_config["activations"][i + 1] != "linear":
                    self.layers.append(
                        self.activations[self.network_config["activations"][i + 1]]
                    )
                else:
                    pass
            else:
                # skip output activation if no activation for last layer is provided
                # e.g. regression network
                pass

        self.len_layers = len(self.layers)
        # -----------------------

    # Define forward pass
    def forward(self, x):
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
        train_config=None,
        model=None,
        train_dl=None,
        valid_dl=None,
        allow_abs_path_folder_generation=False,
        pin_memory=True,
        seed=None,
    ):
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
        self.dev = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("Torch Device: ", self.dev)
        if train_config is None:
            ValueError("train_config is passed as None")
        elif isinstance(train_config, str):
            print("train_config is passed as string: ", train_config)
            try:
                print("Trying to load string as path to pickle file: ")
                self.train_config = pickle.load(open(train_config, "rb"))
            except Exception as e:
                print(e)
        elif isinstance(train_config, dict):
            print("train_config is passed as dictionary: ")
            print(train_config)
            self.train_config = train_config

        self.model = model.to(self.dev)
        self.allow_abs_path_folder_generation = allow_abs_path_folder_generation
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.pin_memory = pin_memory

        self.__get_loss()
        self.__get_optimizer()
        self.__load_weights()

    def __try_wandb(
        self, wandb_project_id="projectid", file_id="fileid", run_id="runid"
    ):
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
            print("Succefully initialized wandb!")
        except Exception as e:
            print(e)
            print("wandb not available, not storing results there")

    def __get_loss(self):
        print(self.train_config)
        if self.train_config["loss"] == "huber":
            self.loss_fun = F.huber_loss
        elif self.train_config["loss"] == "mse":
            self.loss_fun = F.mse_loss
        elif self.train_config["loss"] == "bce":
            self.loss_fun = F.binary_cross_entropy
        elif self.train_config["loss"] == "bcelogit":
            self.loss_fun = F.binary_cross_entropy_with_logits

    def __get_optimizer(self):
        if self.train_config["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                weight_decay=self.train_config["weight_decay"],
                lr=self.train_config["learning_rate"],
            )
        elif self.train_config["optimizer"] == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                weight_decay=self.train_config["weight_decay"],
                lr=self.train_config["learning_rate"],
            )

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
                        if "threshold"
                        in self.train_config["lr_scheduler_params"].keys()
                        else 0.001
                    ),
                    threshold_mode="rel",
                    cooldown=0,
                    min_lr=(
                        self.train_config["lr_scheduler_params"]["min_lr"]
                        if "min_lr" in self.train_config["lr_scheduler_params"].keys()
                        else 0.00000001
                    ),
                    verbose=(
                        self.train_config["lr_scheduler_params"]["verbose"]
                        if "verbose" in self.train_config["lr_scheduler_params"].keys()
                        else True
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
                    verbose=(
                        self.train_config["lr_scheduler_params"]["verbose"]
                        if "verbose" in self.train_config["lr_scheduler_params"].keys()
                        else True
                    ),
                )

    def __load_weights(self):
        # raise NotImplementedError
        # for warmstart, not implemented at the moment
        return

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
        save_onnx=True,
        save_all=True,
        save_data_details=True,
        verbose=1,
    ):
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
            save_history (bool):
                Whether to save the training history.
            save_model (bool):
                Whether to save the model.
            save_config (bool):
                Whether to save the training configuration.
            save_onnx (bool):
                Whether to save the model to ONNX format.
            save_all (bool):
                Whether to save all.
            save_data_details (bool):
                Whether to save the data details.
            verbose (int):
                Verbosity level.
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
        # if self.model.train_output_type == "logprob":
        #     network_type = "lan"
        # elif self.model.train_output_type == "logits":
        #     network_type = "cpn"
        # else:
        #     network_type = "unknown"
        #     print(
        #         'Model type identified as "unknown" because the '
        #         "training_output_type attribute"
        #         + ' of the supplied jax model is neither "logprob", nor "logits"'
        #     )

        training_history = pd.DataFrame(
            np.zeros((self.train_config["n_epochs"], 2)), columns=["epoch", "val_loss"]
        )

        if wandb_on:
            try:
                wandb.watch(self.model, criterion=None, log="all", log_freq=1000)
            except Exception as e:
                print("passing 4")
                print(e)

        step_cnt = 0
        for epoch in range(self.train_config["n_epochs"]):
            self.model.train()
            cnt = 0
            epoch_s_t = time()

            # Training loop
            for xb, yb in self.train_dl:
                # Shift data to device
                if self.pin_memory and self.dev.__str__() == "cuda":
                    xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
                else:
                    xb, yb = xb.to(self.dev), yb.to(self.dev)

                pred = self.model(xb)
                loss = self.loss_fun(pred, yb)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (cnt % 100) == 0 and verbose == 1:
                    print(
                        "epoch: {} / {}, batch: {} / {}, batch_loss: {}".format(
                            epoch,
                            self.train_config["n_epochs"],
                            cnt,
                            self.train_dl.__len__(),
                            loss,
                        )
                    )
                elif (cnt % 1000) == 0 and verbose == 2:
                    print(
                        "epoch: {} / {}, batch: {} / {}, batch_loss: {}".format(
                            epoch,
                            self.train_config["n_epochs"],
                            cnt,
                            self.train_dl.__len__(),
                            loss,
                        )
                    )

                cnt += 1
                step_cnt += 1

            print(
                "Epoch took {} / {},  took {} seconds".format(
                    epoch, self.train_config["n_epochs"], time() - epoch_s_t
                )
            )

            # Start validation
            # self.model.eval()
            with torch.no_grad():
                val_loss = (
                    sum(
                        self.loss_fun(self.model(xb.to(self.dev)), yb.to(self.dev))
                        for xb, yb in self.valid_dl
                    )
                    / self.valid_dl.__len__()
                )
            print(
                "epoch {} / {}, validation_loss: {:2.4}".format(
                    epoch, self.train_config["n_epochs"], val_loss
                )
            )

            # Scheduler step:
            if self.train_config["lr_scheduler"] is not None:
                if self.train_config["lr_scheduler"] == "reduce_on_plateau":
                    self.scheduler.step(val_loss)
                elif self.train_config["lr_scheduler"] == "multiply":
                    self.scheduler.step()

            training_history.values[epoch, :] = [epoch, val_loss.cpu()]

            # Log wandb if possible
            if wandb_on:
                try:
                    wandb.log({"loss": loss, "val_loss": val_loss}, step=step_cnt)
                # print('logged loss')
                except Exception as e:
                    print("passing 5")
                    print(e)

        # Saving
        full_path = (
            output_folder
            + "/"
            + output_file_id
            + "_"
            + self.model.network_type
            + "_"
            + run_id
        )

        if save_history or save_all:
            print("Saving training history")
            training_history_path = full_path + "_torch_training_history.csv"
            pd.DataFrame(training_history).to_csv(training_history_path)
            print("Saving training history to: " + training_history_path)
            self.file_path_training_history = training_history_path

        if save_model or save_all:
            print("Saving model state dict")
            train_state_path = full_path + "_train_state_dict_torch.pt"
            torch.save(
                self.model.state_dict(),
                train_state_path,
            )
            print("Saving model parameters to: " + train_state_path)
            self.file_path_model = train_state_path

        if save_config or save_all:
            config_path = full_path + "_train_config.pickle"
            pickle.dump(self.train_config, open(config_path, "wb"))
            print("Saving training config to: " + config_path)
            self.file_path_config = config_path

        if save_data_details or save_all:
            data_details_path = full_path + "_data_details.pickle"
            pickle.dump(
                {
                    "train_data_generator_config": self.train_dl.dataset.data_generator_config,
                    "train_datafile_ids": self.train_dl.dataset.file_ids,
                    "valid_data_generator_config": self.valid_dl.dataset.data_generator_config,
                    "valid_datafile_ids": self.valid_dl.dataset.file_ids,
                },
                open(data_details_path, "wb"),
            )

            print("Saving training data details to: " + data_details_path)
            self.file_path_data_details = data_details_path

        if save_onnx or save_all:
            print("Saving model to ONNX format")
            onnx_path = full_path + "_torch_model.onnx"
            # Put model in eval mode
            self.model.eval()
            torch.onnx.export(
                self.model,
                torch.randn(1, self.model.input_shape).to(self.dev),
                onnx_path,
            )
            print("Saving model to ONNX format to: " + onnx_path)
            self.file_path_onnx = onnx_path

        # Upload wandb data
        if wandb_on:
            try:
                wandb.finish()
                print("wandb uploaded")
            except Exception as e:
                print("passing 6")
                print(e)

        print("Training finished successfully...")


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

    def __init__(self, model_file_path=None, network_config=None, input_dim=None):
        torch.backends.cudnn.benchmark = True
        self.dev = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model_file_path = model_file_path

        if isinstance(network_config, str):
            self.network_config = pickle.load(open(network_config, "rb"))
        elif isinstance(network_config, dict):
            self.network_config = network_config
        else:
            raise ValueError("network config is neither a string nor a dictionary")

        # self.network_config = network_config
        self.input_dim = input_dim

        self.net = TorchMLP(
            network_config=self.network_config,
            input_shape=self.input_dim,
            generative_model_id=None,
        )
        # self.net.load_state_dict(torch.load(self.model_file_path))
        if not torch.cuda.is_available():
            self.net.load_state_dict(
                torch.load(self.model_file_path, map_location=torch.device("cpu"))
            )
        else:
            self.net.load_state_dict(torch.load(self.model_file_path))
        self.net.to(self.dev)
        self.net.eval()

    # AF-TODO: Seemingly LoadTorchMLPInfer is still not callable !
    @torch.no_grad()
    def __call__(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict_on_batch(self, x=None):
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

    def __init__(self, model_file_path=None, network_config=None, input_dim=None):
        ##torch.backends.cudnn.benchmark = True
        self.dev = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model_file_path = model_file_path
        self.network_config = network_config
        self.input_dim = input_dim

        self.net = TorchMLP(
            network_config=self.network_config,
            input_shape=self.input_dim,
            generative_model_id=None,
        )
        # self.net.load_state_dict(torch.load(self.model_file_path))
        if not torch.cuda.is_available():
            self.net.load_state_dict(
                torch.load(self.model_file_path, map_location=torch.device("cpu"))
            )
        else:
            self.net.load_state_dict(torch.load(self.model_file_path))
        self.net.to(self.dev)

    # AF-TODO: Seemingly LoadTorchMLPInfer is still not callable !
    @torch.no_grad()
    def __call__(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict_on_batch(self, x=None):
        return self.net(torch.from_numpy(x).to(self.dev)).cpu().numpy()
