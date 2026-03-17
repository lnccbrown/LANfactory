"""Tests for model_card.py module."""

import pickle

import pytest
import yaml

from lanfactory.hf.model_card import (
    ModelCardConfig,
    generate_readme,
    load_model_card_yaml,
    write_readme,
)


class TestModelCardConfig:
    """Tests for ModelCardConfig dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = ModelCardConfig()
        assert config.tags == ["lan", "ssm", "hssm"]
        assert config.library_name == "onnx"
        assert config.license == "mit"
        assert config.title == "LAN Model"
        assert config.architecture is None
        assert config.training is None
        assert config.usage_example is None

    def test_custom_values(self):
        """Test custom values are set correctly."""
        config = ModelCardConfig(
            tags=["lan", "ssm", "ddm"],
            title="DDM LAN Model",
            description="Custom description",
            architecture={"layer_sizes": [100, 100, 1]},
        )
        assert config.tags == ["lan", "ssm", "ddm"]
        assert config.title == "DDM LAN Model"
        assert config.description == "Custom description"
        assert config.architecture == {"layer_sizes": [100, 100, 1]}


class TestLoadModelCardYaml:
    """Tests for load_model_card_yaml function."""

    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid model_card.yaml file."""
        yaml_content = {
            "tags": ["lan", "ssm", "ddm", "hssm"],
            "library_name": "onnx",
            "license": "mit",
            "title": "LAN Model for DDM",
            "description": "Test description",
        }

        yaml_path = tmp_path / "model_card.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_model_card_yaml(tmp_path)

        assert config.tags == ["lan", "ssm", "ddm", "hssm"]
        assert config.library_name == "onnx"
        assert config.title == "LAN Model for DDM"
        assert config.description == "Test description"

    def test_load_yaml_with_defaults(self, tmp_path):
        """Test loading a minimal YAML file uses defaults."""
        yaml_content = {"title": "Minimal Model"}

        yaml_path = tmp_path / "model_card.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_model_card_yaml(tmp_path)

        assert config.title == "Minimal Model"
        assert config.tags == ["lan", "ssm", "hssm"]  # Default
        assert config.library_name == "onnx"  # Default

    def test_load_yaml_not_found(self, tmp_path):
        """Test FileNotFoundError when YAML doesn't exist."""
        with pytest.raises(FileNotFoundError, match="model_card.yaml not found"):
            load_model_card_yaml(tmp_path)

    def test_load_yaml_fills_from_pickle(self, tmp_path):
        """Test that architecture is filled from pickle config."""
        # Create minimal YAML
        yaml_content = {"title": "Test Model"}
        yaml_path = tmp_path / "model_card.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        # Create network config pickle
        network_config = {
            "layer_sizes": [100, 100, 1],
            "activations": ["tanh", "tanh", "linear"],
            "network_type": "lan",
        }
        pickle_path = tmp_path / "test_network_config.pickle"
        with open(pickle_path, "wb") as f:
            pickle.dump(network_config, f)

        config = load_model_card_yaml(tmp_path)

        assert config.architecture is not None
        assert config.architecture["layer_sizes"] == [100, 100, 1]
        assert config.architecture["network_type"] == "lan"


class TestGenerateReadme:
    """Tests for generate_readme function."""

    def test_generates_valid_frontmatter(self):
        """Test that generated README has valid YAML frontmatter."""
        config = ModelCardConfig(
            tags=["lan", "ssm", "ddm"],
            title="Test Model",
        )

        readme = generate_readme(config)

        # Check frontmatter markers
        assert readme.startswith("---\n")
        assert "\n---\n" in readme

        # Extract and parse frontmatter
        parts = readme.split("---")
        frontmatter = yaml.safe_load(parts[1])

        assert frontmatter["tags"] == ["lan", "ssm", "ddm"]
        assert frontmatter["library_name"] == "onnx"
        assert frontmatter["license"] == "mit"

    def test_includes_title_and_description(self):
        """Test that README includes title and description."""
        config = ModelCardConfig(
            title="My Model",
            description="My description",
        )

        readme = generate_readme(config)

        assert "# My Model" in readme
        assert "My description" in readme

    def test_includes_architecture_section(self):
        """Test that architecture section is included when provided."""
        config = ModelCardConfig(
            architecture={
                "layer_sizes": [100, 100, 1],
                "activations": ["tanh", "tanh", "linear"],
                "network_type": "lan",
            }
        )

        readme = generate_readme(config)

        assert "## Architecture" in readme
        assert "**Network Type:** lan" in readme
        assert "[100, 100, 1]" in readme

    def test_includes_training_section(self):
        """Test that training section is included when provided."""
        config = ModelCardConfig(
            training={
                "epochs": 20,
                "optimizer": "adam",
                "learning_rate": 0.001,
            }
        )

        readme = generate_readme(config)

        assert "## Training" in readme
        assert "**Epochs:** 20" in readme
        assert "**Optimizer:** adam" in readme

    def test_includes_usage_example(self):
        """Test that usage example section is included."""
        config = ModelCardConfig()

        readme = generate_readme(config, model_name="ddm")

        assert "## Usage with HSSM" in readme
        assert 'model="ddm"' in readme

    def test_custom_usage_example(self):
        """Test that custom usage example is used when provided."""
        config = ModelCardConfig(usage_example="custom_code_here()")

        readme = generate_readme(config)

        assert "custom_code_here()" in readme


class TestWriteReadme:
    """Tests for write_readme function."""

    def test_writes_readme_file(self, tmp_path):
        """Test that README.md is written to disk."""
        config = ModelCardConfig(title="Test Model")

        readme_path = write_readme(tmp_path, config)

        assert readme_path.exists()
        assert readme_path.name == "README.md"

        content = readme_path.read_text()
        assert "# Test Model" in content
