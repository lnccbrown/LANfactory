import ssms
import lanfactory
from pathlib import Path
import numpy as np
from copy import deepcopy
import torch
import tempfile
import shutil


def make_configs():
    # MAKE CONFIGS
    model = "angle"
    # Initialize the generator config (for MLP LANs)
    generator_config = ssms.config.data_generator_config["lan"]
    # Specify generative model (one from the list of included models mentioned above)
    generator_config["model"] = model
    # Specify number of parameter sets to simulate
    generator_config["n_parameter_sets"] = 100
    # Specify how many samples a simulation run should entail
    generator_config["n_samples"] = 1000
    # Specify folder in which to save generated data
    generator_config["output_folder"] = "data/lan_mlp/" + model + "/"

    # Make model config dict
    model_config = ssms.config.model_config[model]

    # MAKE DATA
    my_dataset_generator = ssms.dataset_generators.lan_mlp.data_generator(
        generator_config=generator_config, model_config=model_config
    )

    training_data = my_dataset_generator.generate_data_training_uniform(save=True)

    return training_data
