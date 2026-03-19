"""Model card utilities for HuggingFace Hub.

This module reads user-provided model_card.yaml files and generates
HuggingFace-compatible README.md files with proper frontmatter.
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelCardConfig:
    """Configuration for model card generation.

    Attributes
    ----------
    tags : list[str]
        HuggingFace tags for discoverability.
    library_name : str
        Library name for HuggingFace (default: "onnx").
    license : str
        License identifier (default: "mit").
    title : str
        Model title.
    description : str
        Model description.
    architecture : dict | None
        Network architecture details.
    training : dict | None
        Training configuration details.
    usage_example : str | None
        Usage example code.
    """

    tags: list[str] = field(default_factory=lambda: ["lan", "ssm", "hssm"])
    library_name: str = "onnx"
    license: str = "mit"
    title: str = "LAN Model"
    description: str = "Likelihood Approximation Network trained with LANfactory."
    architecture: dict | None = None
    training: dict | None = None
    usage_example: str | None = None


def load_model_card_yaml(model_folder: Path) -> ModelCardConfig:
    """Load model card configuration from YAML file.

    Parameters
    ----------
    model_folder : Path
        Path to the model folder containing model_card.yaml.

    Returns
    -------
    ModelCardConfig
        Parsed model card configuration.

    Raises
    ------
    FileNotFoundError
        If model_card.yaml is not found in the model folder.
    """
    yaml_path = model_folder / "model_card.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"model_card.yaml not found in {model_folder}. "
            "Please create a model_card.yaml file with model metadata."
        )

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Extract fields with defaults
    config = ModelCardConfig(
        tags=data.get("tags", ["lan", "ssm", "hssm"]),
        library_name=data.get("library_name", "onnx"),
        license=data.get("license", "mit"),
        title=data.get("title", "LAN Model"),
        description=data.get(
            "description", "Likelihood Approximation Network trained with LANfactory."
        ),
        architecture=data.get("architecture"),
        training=data.get("training"),
        usage_example=data.get("usage_example"),
    )

    # Try to fill in missing architecture/training from pickle configs
    config = _fill_from_pickle_configs(model_folder, config)

    return config


def _fill_from_pickle_configs(
    model_folder: Path, config: ModelCardConfig
) -> ModelCardConfig:
    """Fill in missing config details from pickle files if available.

    Parameters
    ----------
    model_folder : Path
        Path to the model folder.
    config : ModelCardConfig
        Partially filled configuration.

    Returns
    -------
    ModelCardConfig
        Configuration with filled-in details from pickle files.
    """
    # Try to find and load network config
    if config.architecture is None:
        network_config_files = list(model_folder.glob("*network_config.pickle"))
        if network_config_files:
            try:
                with open(network_config_files[0], "rb") as f:
                    network_config = pickle.load(f)
                config.architecture = {
                    "layer_sizes": network_config.get("layer_sizes"),
                    "activations": network_config.get("activations"),
                    "network_type": network_config.get("network_type"),
                }
                logger.info(f"Loaded architecture from {network_config_files[0]}")
            except Exception as e:
                logger.warning(f"Could not load network config: {e}")

    # Try to find and load train config
    if config.training is None:
        train_config_files = list(model_folder.glob("*train_config.pickle"))
        if train_config_files:
            try:
                with open(train_config_files[0], "rb") as f:
                    train_config = pickle.load(f)
                config.training = {
                    "epochs": train_config.get("n_epochs"),
                    "optimizer": train_config.get("optimizer"),
                    "learning_rate": train_config.get("learning_rate"),
                    "loss": train_config.get("loss"),
                }
                logger.info(f"Loaded training config from {train_config_files[0]}")
            except Exception as e:
                logger.warning(f"Could not load train config: {e}")

    return config


def generate_readme(config: ModelCardConfig, model_name: str | None = None) -> str:
    """Generate HuggingFace-compatible README.md content.

    Parameters
    ----------
    config : ModelCardConfig
        Model card configuration.
    model_name : str | None
        Model name to include in usage example.

    Returns
    -------
    str
        README.md content with YAML frontmatter.
    """
    # Build YAML frontmatter
    frontmatter_dict: dict[str, Any] = {
        "tags": config.tags,
        "library_name": config.library_name,
        "license": config.license,
    }

    frontmatter = yaml.dump(frontmatter_dict, default_flow_style=False, sort_keys=False)

    # Build README content
    lines = [
        "---",
        frontmatter.strip(),
        "---",
        "",
        f"# {config.title}",
        "",
        config.description,
        "",
    ]

    # Add architecture section if available
    if config.architecture:
        lines.extend(
            [
                "## Architecture",
                "",
            ]
        )
        if config.architecture.get("network_type"):
            lines.append(f"- **Network Type:** {config.architecture['network_type']}")
        if config.architecture.get("layer_sizes"):
            lines.append(f"- **Layer Sizes:** {config.architecture['layer_sizes']}")
        if config.architecture.get("activations"):
            lines.append(f"- **Activations:** {config.architecture['activations']}")
        lines.append("")

    # Add training section if available
    if config.training:
        lines.extend(
            [
                "## Training",
                "",
            ]
        )
        if config.training.get("epochs"):
            lines.append(f"- **Epochs:** {config.training['epochs']}")
        if config.training.get("optimizer"):
            lines.append(f"- **Optimizer:** {config.training['optimizer']}")
        if config.training.get("learning_rate"):
            lines.append(f"- **Learning Rate:** {config.training['learning_rate']}")
        if config.training.get("loss"):
            lines.append(f"- **Loss:** {config.training['loss']}")
        lines.append("")

    # Add usage example
    lines.extend(
        [
            "## Usage with HSSM",
            "",
            "```python",
        ]
    )

    if config.usage_example:
        lines.append(config.usage_example.strip())
    else:
        # Default usage example
        model_str = model_name or "ddm"
        lines.extend(
            [
                "import hssm",
                f'model = hssm.HSSM(data=my_data, model="{model_str}", loglik_kind="approx_differentiable")',
            ]
        )

    lines.extend(
        [
            "```",
            "",
            "## Citation",
            "",
            "If you use this model, please cite:",
            "",
            "- [LANfactory](https://github.com/lnccbrown/LANfactory)",
            "- [HSSM](https://github.com/lnccbrown/HSSM)",
            "",
        ]
    )

    return "\n".join(lines)


def write_readme(
    model_folder: Path, config: ModelCardConfig, model_name: str | None = None
) -> Path:
    """Generate and write README.md to model folder.

    Parameters
    ----------
    model_folder : Path
        Path to the model folder.
    config : ModelCardConfig
        Model card configuration.
    model_name : str | None
        Model name to include in usage example.

    Returns
    -------
    Path
        Path to the written README.md file.
    """
    readme_content = generate_readme(config, model_name)
    readme_path = model_folder / "README.md"

    with open(readme_path, "w") as f:
        f.write(readme_content)

    logger.info(f"Generated README.md at {readme_path}")
    return readme_path
