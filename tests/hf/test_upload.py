"""Tests for upload.py module."""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from lanfactory.hf.upload import (
    DEFAULT_INCLUDE_PATTERNS,
    DEFAULT_REPO_ID,
    _collect_files,
    upload_model,
)

try:
    import huggingface_hub  # noqa: F401

    HAS_HF = True
except ImportError:
    HAS_HF = False

requires_hf = pytest.mark.skipif(not HAS_HF, reason="huggingface_hub not installed")


class TestCollectFiles:
    """Tests for _collect_files function."""

    def test_collects_matching_files(self, tmp_path):
        """Test collecting files matching patterns."""
        # Create test files
        (tmp_path / "model.onnx").write_text("onnx content")
        (tmp_path / "model.pt").write_text("pytorch content")
        (tmp_path / "config.pickle").write_text("config content")
        (tmp_path / "other.txt").write_text("other content")

        files = _collect_files(
            tmp_path,
            include_patterns=["*.onnx", "*.pt"],
            exclude_patterns=None,
        )

        filenames = [f.name for f in files]
        assert "model.onnx" in filenames
        assert "model.pt" in filenames
        assert "other.txt" not in filenames

    def test_excludes_files(self, tmp_path):
        """Test excluding files matching patterns."""
        (tmp_path / "model.onnx").write_text("content")
        (tmp_path / "backup.onnx").write_text("content")

        files = _collect_files(
            tmp_path,
            include_patterns=["*.onnx"],
            exclude_patterns=["backup*"],
        )

        filenames = [f.name for f in files]
        assert "model.onnx" in filenames
        assert "backup.onnx" not in filenames

    def test_returns_empty_for_no_matches(self, tmp_path):
        """Test returns empty list when no files match."""
        (tmp_path / "other.txt").write_text("content")

        files = _collect_files(
            tmp_path,
            include_patterns=["*.onnx"],
            exclude_patterns=None,
        )

        assert files == []


class TestUploadModel:
    """Tests for upload_model function."""

    def test_raises_if_folder_not_exists(self, tmp_path):
        """Test raises FileNotFoundError if folder doesn't exist."""
        non_existent = tmp_path / "non_existent"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            upload_model(
                model_folder=non_existent,
                network_type="lan",
                model_name="ddm",
            )

    def test_raises_if_invalid_network_type(self, tmp_path):
        """Test raises ValueError for invalid network_type."""
        with pytest.raises(ValueError, match="network_type must be one of"):
            upload_model(
                model_folder=tmp_path,
                network_type="invalid",
                model_name="ddm",
            )

    def test_raises_if_model_card_missing(self, tmp_path):
        """Test raises FileNotFoundError if model_card.yaml is missing."""
        with pytest.raises(FileNotFoundError, match="model_card.yaml not found"):
            upload_model(
                model_folder=tmp_path,
                network_type="lan",
                model_name="ddm",
            )

    def test_raises_if_no_matching_files(self, tmp_path):
        """Test raises FileNotFoundError when no files match patterns."""
        yaml_content = {"title": "Test Model"}
        with open(tmp_path / "model_card.yaml", "w") as f:
            yaml.dump(yaml_content, f)

        with pytest.raises(FileNotFoundError, match="No files matching patterns"):
            upload_model(
                model_folder=tmp_path,
                network_type="lan",
                model_name="ddm",
                include_patterns=["*.nonexistent"],
            )

    def test_dry_run_does_not_upload(self, tmp_path):
        """Test dry_run shows files but doesn't upload."""
        # Create model_card.yaml
        yaml_content = {
            "tags": ["lan", "ssm", "ddm"],
            "title": "Test Model",
        }
        yaml_path = tmp_path / "model_card.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        # Create a model file
        (tmp_path / "model.onnx").write_text("onnx content")

        result = upload_model(
            model_folder=tmp_path,
            network_type="lan",
            model_name="ddm",
            dry_run=True,
        )

        assert result is None

    @requires_hf
    @patch("huggingface_hub.HfApi")
    @patch("huggingface_hub.create_repo")
    def test_creates_repo_when_requested(
        self, mock_create_repo, mock_api_class, tmp_path
    ):
        """Test repository is created when create_repo=True."""
        # Create model_card.yaml
        yaml_content = {"tags": ["lan", "ssm"], "title": "Test"}
        with open(tmp_path / "model_card.yaml", "w") as f:
            yaml.dump(yaml_content, f)
        (tmp_path / "model.onnx").write_text("content")

        # Mock API
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        upload_model(
            model_folder=tmp_path,
            network_type="lan",
            model_name="ddm",
            create_repo=True,
            token="fake_token",
        )

        mock_create_repo.assert_called_once()

    @requires_hf
    @patch("huggingface_hub.HfApi")
    @patch("huggingface_hub.create_repo")
    def test_uploads_to_correct_path(self, mock_create_repo, mock_api_class, tmp_path):
        """Test files are uploaded to correct path in repo."""
        # Create model_card.yaml
        yaml_content = {"tags": ["lan", "ssm"], "title": "Test"}
        with open(tmp_path / "model_card.yaml", "w") as f:
            yaml.dump(yaml_content, f)
        (tmp_path / "model.onnx").write_text("content")

        # Mock API
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        upload_model(
            model_folder=tmp_path,
            network_type="lan",
            model_name="ddm",
            repo_id="test/repo",
        )

        # Check upload_folder was called with correct path
        mock_api.upload_folder.assert_called_once()
        call_kwargs = mock_api.upload_folder.call_args[1]
        assert call_kwargs["path_in_repo"] == "lan/ddm"
        assert call_kwargs["repo_id"] == "test/repo"


class TestDefaults:
    """Tests for default values."""

    def test_default_repo_id(self):
        """Test default repo ID is franklab/HSSM."""
        assert DEFAULT_REPO_ID == "franklab/HSSM"

    def test_default_include_patterns(self):
        """Test default include patterns."""
        assert "*.onnx" in DEFAULT_INCLUDE_PATTERNS
        assert "*.pt" in DEFAULT_INCLUDE_PATTERNS
        assert "model_card.yaml" in DEFAULT_INCLUDE_PATTERNS
