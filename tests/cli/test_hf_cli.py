"""CLI smoke tests for upload-hf and download-hf commands."""

import os
import subprocess

import yaml

_PLAIN_TEXT_ENV = {**os.environ, "NO_COLOR": "1", "COLUMNS": "200"}


class TestUploadHfCliHelp:
    """Tests for upload-hf CLI help and argument validation."""

    def test_help_command(self):
        """Test that --help works."""
        result = subprocess.run(
            ["upload-hf", "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=_PLAIN_TEXT_ENV,
        )
        assert result.returncode == 0
        assert "Upload a trained LANfactory model" in result.stdout
        assert "--model-folder" in result.stdout
        assert "--network-type" in result.stdout
        assert "--model-name" in result.stdout

    def test_missing_required_args(self):
        """Test that missing required args causes error."""
        result = subprocess.run(
            ["upload-hf"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0

    def test_invalid_network_type(self, tmp_path):
        """Test that invalid network type causes error."""
        # Create a dummy model_card.yaml
        yaml_path = tmp_path / "model_card.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump({"title": "Test"}, f)

        result = subprocess.run(
            [
                "upload-hf",
                "--model-folder",
                str(tmp_path),
                "--network-type",
                "invalid",
                "--model-name",
                "ddm",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0


class TestDownloadHfCliHelp:
    """Tests for download-hf CLI help and argument validation."""

    def test_help_command(self):
        """Test that --help works."""
        result = subprocess.run(
            ["download-hf", "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=_PLAIN_TEXT_ENV,
        )
        assert result.returncode == 0
        assert "Download a LANfactory model" in result.stdout
        assert "--network-type" in result.stdout
        assert "--model-name" in result.stdout
        assert "--output-folder" in result.stdout

    def test_missing_required_args(self):
        """Test that missing required args causes error."""
        result = subprocess.run(
            ["download-hf"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0

    def test_invalid_network_type(self, tmp_path):
        """Test that invalid network type causes error."""
        result = subprocess.run(
            [
                "download-hf",
                "--network-type",
                "invalid",
                "--model-name",
                "ddm",
                "--output-folder",
                str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0


class TestUploadHfDryRun:
    """Tests for upload-hf dry run functionality."""

    def test_dry_run_with_model_card(self, tmp_path):
        """Test dry run with valid model_card.yaml."""
        # Create model_card.yaml
        yaml_content = {
            "tags": ["lan", "ssm", "ddm", "hssm"],
            "library_name": "onnx",
            "title": "Test Model",
            "description": "Test description",
        }
        yaml_path = tmp_path / "model_card.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f)

        # Create a dummy model file
        (tmp_path / "model.onnx").write_text("dummy content")

        result = subprocess.run(
            [
                "upload-hf",
                "--model-folder",
                str(tmp_path),
                "--network-type",
                "lan",
                "--model-name",
                "test-model",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

    def test_dry_run_missing_model_card(self, tmp_path):
        """Test dry run fails when model_card.yaml is missing."""
        result = subprocess.run(
            [
                "upload-hf",
                "--model-folder",
                str(tmp_path),
                "--network-type",
                "lan",
                "--model-name",
                "test-model",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert (
            "model_card.yaml not found" in result.stderr
            or "model_card.yaml not found" in result.stdout
        )
