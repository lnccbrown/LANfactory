"""Test configuration constants."""

from dataclasses import dataclass, field


@dataclass
class TestGeneratorConstants:
    """Test configuration constants."""

    MODEL: str = "angle"
    N_PARAMETER_SETS: int = 256
    N_SAMPLES: int = 2000
    N_TRAINING_SAMPLES: int = 2000
    N_SAMPLES_BY_PARAMETER_SET: int = 2000
    TEST_FOLDER: str = "tests/test_data"
    OUT_FOLDER: str = "tests/test_data/lan_mlp/training_data"
    MODEL_FOLDER: str = "tests/test_data/jax_models/lan"
    N_DATA_FILES: int = 2
    DEVICE: str = "cpu"


@dataclass
class TestTrainConstants:
    """Test training constants."""

    N_EPOCHS: int = 2
    CPU_BATCH_SIZE: int = 4196
    GPU_BATCH_SIZE: int = 4196
    OPTIMIZER: str = "adam"
    LEARNING_RATE: float = 2e-06
    LR_SCHEDULER: str = "reduce_on_plateau"
    LR_SCHEDULER_PARAMS: dict = field(default_factory=dict)
    WEIGHT_DECAY: float = 0.0
    LOSS: str = "huber"
    SAVE_HISTORY: bool = False


@dataclass
class TestNetworkConstantsLAN:
    """Test network constants for LANs."""

    LAYER_SIZES: list = field(default_factory=lambda: [100, 100, 100, 1])
    ACTIVATIONS: list = field(
        default_factory=lambda: ["tanh", "tanh", "tanh", "linear"]
    )
    TRAIN_OUTPUT_TYPE: str = "logprob"


TEST_GENERATOR_CONSTANTS = TestGeneratorConstants()
TEST_TRAIN_CONSTANTS = TestTrainConstants()
TEST_NETWORK_CONSTANTS_LAN = TestNetworkConstantsLAN()
