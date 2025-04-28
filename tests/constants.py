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
    LAN_MODEL_FOLDER: str = "tests/test_data/jax_models/lan"
    CPN_MODEL_FOLDER: str = "tests/test_data/jax_models/cpn"
    OPN_MODEL_FOLDER: str = "tests/test_data/jax_models/opn"
    N_DATA_FILES: int = 2
    DEVICE: str = "cpu"


@dataclass
class TestTrainConstantsLAN:
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
class TestTrainConstantsCPN:
    """Test training constants."""

    N_EPOCHS: int = 2
    CPU_BATCH_SIZE: int = 32
    GPU_BATCH_SIZE: int = 32
    OPTIMIZER: str = "adam"
    LEARNING_RATE: float = 2e-06
    LR_SCHEDULER: str = "reduce_on_plateau"
    LR_SCHEDULER_PARAMS: dict = field(default_factory=dict)
    WEIGHT_DECAY: float = 0.0
    LOSS: str = "bcelogit"
    SAVE_HISTORY: bool = False


@dataclass
class TestTrainConstantsOPN:
    """Test training constants."""

    N_EPOCHS: int = 2
    CPU_BATCH_SIZE: int = 32
    GPU_BATCH_SIZE: int = 32
    OPTIMIZER: str = "adam"
    LEARNING_RATE: float = 2e-06
    LR_SCHEDULER: str = "reduce_on_plateau"
    LR_SCHEDULER_PARAMS: dict = field(default_factory=dict)
    WEIGHT_DECAY: float = 0.0
    LOSS: str = "bcelogit"
    SAVE_HISTORY: bool = False


@dataclass
class TestNetworkConstantsLAN:
    """Test network constants for LANs."""

    LAYER_SIZES: list = field(default_factory=lambda: [100, 100, 100, 1])
    ACTIVATIONS: list = field(
        default_factory=lambda: ["tanh", "tanh", "tanh", "linear"]
    )
    TRAIN_OUTPUT_TYPE: str = "logprob"


@dataclass
class TestNetworkConstantsCPN:
    """Test network constants for CPNs."""

    LAYER_SIZES: list = field(default_factory=lambda: [100, 100, 1])
    ACTIVATIONS: list = field(default_factory=lambda: ["tanh", "tanh", "linear"])
    TRAIN_OUTPUT_TYPE: str = "logits"


@dataclass
class TestNetworkConstantsOPN:
    """Test network constants for OPNs."""

    LAYER_SIZES: list = field(default_factory=lambda: [100, 100, 1])
    ACTIVATIONS: list = field(default_factory=lambda: ["tanh", "tanh", "linear"])
    TRAIN_OUTPUT_TYPE: str = "logits"


TEST_GENERATOR_CONSTANTS = TestGeneratorConstants()
TEST_TRAIN_CONSTANTS_LAN = TestTrainConstantsLAN()
TEST_TRAIN_CONSTANTS_CPN = TestTrainConstantsCPN()
TEST_TRAIN_CONSTANTS_OPN = TestTrainConstantsOPN()
TEST_NETWORK_CONSTANTS_LAN = TestNetworkConstantsLAN()
TEST_NETWORK_CONSTANTS_CPN = TestNetworkConstantsCPN()
TEST_NETWORK_CONSTANTS_OPN = TestNetworkConstantsOPN()
