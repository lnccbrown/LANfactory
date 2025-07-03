# LANfactory

![PyPI](https://img.shields.io/pypi/v/lanfactory)
![PyPI_dl](https://img.shields.io/pypi/dm/lanfactory)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Lightweight python package to help with training [LANs](https://elifesciences.org/articles/65074) (Likelihood approximation networks). 

Please find the original [documentation here](https://alexanderfengler.github.io/LANfactory/).

### Quick Start

The `LANfactory` package is a light-weight convenience package for training `likelihood approximation networks` (LANs) in torch (or jaxtrain), starting from supplied training data.

[LANs](https://elifesciences.org/articles/65074), although more general in potential scope of applications, were conceived in the context of sequential sampling modeling
to account for cognitive processes giving rise to *choice* and *reaction time* data in *n-alternative forced choice experiments* commonly encountered in the cognitive sciences.

For a basic tutorial on how to use the `LANfactory` package, please refer to the [basic tutorial notebook](docs/basic_tutorial/basic_tutorial.ipynb)..

#### Install

To install the `LANfactory` package type,

`pip install lanfactory`

Necessary dependency should be installed automatically in the process.

### Basic Tutorial

Check the basic tutorial [here](docs/basic_tutorial/basic_tutorial.ipynb).

### Command Line Interface

LANfactory includes a command line interface with the commands `jaxtrain` and `torchtrain`, which train neural networks using `jax` and `torch` as backends, respectively.

**Examples**
```bash
jaxtrain --config-path config.yaml --training-data-folder my_generated_data --network-id 0 --dl-workers 3 --network-path-base my_trained_network
```
```bash
torchtrain --config-path config.yaml --training-data-folder my_generated_data --network-id 0 --dl-workers 3 --network-path-base my_trained_network
```

`jaxtrain` and `torchtrain` have the same 6 arguments 
* `--config-path`: Path to the YAML config file
* `--training-data-folder`: Path to folder with data to train the neural network on
* `--networks-path-base`: Path to the output folder for trained neural network
* `--network-id`: ID for the neural network to train (default: 0)
* `--dl-workers`: Number of cores to use with the dataloader class (default: 1)
* `--log-level`: Set the logging level (default: WARNING)

You can also view the help to see further documentation.

Below is a sample configuration file you can use with `jaxtrain` or `torchtrain`. 

```yaml
NETWORK_TYPE: "lan" 
CPU_BATCH_SIZE: 1000
GPU_BATCH_SIZE: 50000
GENERATOR_APPROACH: "lan"
OPTIMIZER_: "adam"
N_EPOCHS: 20
MODEL: "ddm"
SHUFFLE: True
LAYER_SIZES: [[100, 100, 100, 1], [100, 100, 100, 100, 1], [100, 100, 100, 100, 100, 1],
              [120, 120, 120, 1], [120, 120, 120, 120, 1], [120, 120, 120, 120, 120, 1]]
ACTIVATIONS: [['tanh', 'tanh', 'tanh'],
              ['tanh', 'tanh', 'tanh', 'tanh'],
              ['tanh', 'tanh', 'tanh', 'tanh', 'tanh'],
              ['tanh', 'tanh', 'tanh'],
              ['tanh', 'tanh', 'tanh', 'tanh'],
              ['tanh', 'tanh', 'tanh', 'tanh', 'tanh']] # specifies all but output layer activation (output layer activation is determined by)
WEIGHT_DECAY: 0.0
TRAIN_VAL_SPLIT: 0.5
N_TRAINING_FILES: 10000 # can be list
LABELS_LOWER_BOUND: np.log(1e-7)
LEARNING_RATE: 0.001
LR_SCHEDULER: 'reduce_on_plateau'
LR_SCHEDULER_PARAMS:
  factor: 0.1
  patience: 2
  threshold: 0.001
  min_lr: 0.00000001
  verbose: True
```

Configuration file parameter details follow:

| Option | Definition |
| ------ | ---------- |
| `NETWORK_TYPE` | The type of network you want to train. Other options include "cpn", "opn", "gonogo" and "cpn_bce" |
| `CPU_BATCH_SIZE` | Number of samples to work through before updating internal model parameters, when CPU is being used |
| `GPU_BATCH_SIZE` | Number of samples to work through before updating internal model parameters, when GPU is being used |
| `GENERATOR_APPROACH` | Compatible training data generator to train the respective LAN |
| `OPTIMIZER` | Optimization algorithm used to train the network |
| `N_EPOCHS` | Number of passes through the entire training dataset |
| `MODEL` | Type of model that was simulated |
| `SHUFFLE` | Boolean that represents whether training data is shuffled before each epoch |
| `LAYER_SIZES` | Number of neurons in each layer of the neural network. Contains multiple vectors of layer sizes to choose the best network after iterating through all networks |
| `ACTIVATIONS` | Type of function that decides whether a neuron should be activated or not, depending on the weighted sum of the inputs it receives. Contains multiple options due to iteration through multiple networks |
| `WEIGHT_DECAY` | Controls the amount of regularization to prevent overfitting, also known as L2 regularization |
| `TRAIN_VAL_SPLIT` | Percentage of files used for training data vs. validation |
| `N_TRAINING_FILES` | Max number of training files to use for training and validation |
| `LABELS_LOWER_BOUND` | Minimum value for training labels to prevent extreme or undefined values |
| `LEARNING_RATE` | A hyperparameter that controls how much the model weights are adjusted during training. A smaller learning rate means slower training but potentially more accurate results |
| `LR_SCHEDULER` | The learning rate scheduler used to adapt the learning rate during training. `reduce_on_plateau` reduces the learning rate when the validation loss stops improving. |
| `LR_SCHEDULER_PARAMS` | A dictionary specifying the parameters for the learning rate scheduler. It includes: `factor` (multiplier applied to reduce the LR), `patience` (number of epochs with no improvement before reducing LR), `threshold` (minimum change to qualify as improvement), `min_lr` (minimum LR allowed), and `verbose` (whether to print updates). |

To make your own configuration file, you can copy the example above into a new `.yaml` file and modify it with your preferences.

If you are using `uv`, you can also use the `uv run` command to run `jaxtrain` or `torchtrain` from the command line
    
### TorchMLP to ONNX Converter

Once you have trained your model, you can convert it to the ONNX format using the `transform_onnx.py` script.

The `transform_onnx.py` script converts a TorchMLP model to the ONNX format. It takes a network configuration file (in pickle format), a state dictionary file (Torch model weights), the size of the input tensor, and the desired output ONNX file path.

### Usage

```python onnx/transform_onnx.py <network_config_file> <state_dict_file> <input_shape> <output_onnx_file>```

Replace the placeholders with the appropriate values:

- <network_config_file>: Path to the pickle file containing the network configuration.
- <state_dict_file>: Path to the file containing the state dictionary of the model.
- <input_shape>: The size of the input tensor for the model (integer).
- <output_onnx_file>: Path to the output ONNX file.

For example:

```
python onnx/transform_onnx.py '0d9f0e94175b11eca9e93cecef057438_lca_no_bias_4_torch__network_config.pickle' '0d9f0e94175b11eca9e93cecef057438_lca_no_bias_4_torch_state_dict.pt' 11 'lca_no_bias_4_torch.onnx'
```
This onnx file can be used directly with the [`HSSM`](https://github.com/lnccbrown/HSSM) package. 

We hope this package may be helpful in case you attempt to train [LANs](https://elifesciences.org/articles/65074) for your own research.

#### END

