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

    def test_raises_if_model_card_missing_and_required(self, tmp_path):
        """A missing model_card.yaml is only fatal on request.

        It used to be fatal always, which blocked automated publishing for a
        file the library can generate from the artifacts; the strict behavior
        now lives behind require_model_card (see tests/hf/test_dual_layout.py).
        """
        with pytest.raises(FileNotFoundError, match="model_card.yaml not found"):
            upload_model(
                model_folder=tmp_path,
                network_type="lan",
                model_name="ddm",
                require_model_card=True,
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

        # Artifact named the way the trainers name them (model in the filename)
        (tmp_path / "abc_lan_ddm__model.onnx").write_text("onnx content")

        result = upload_model(
            model_folder=tmp_path,
            network_type="lan",
            model_name="ddm",
            dry_run=True,
        )

        assert result is None

    def test_generically_named_onnx_needs_an_explicit_choice(self, tmp_path):
        """A filename that does not name the model must be opted into.

        The root filename is what HSSM downloads, so an artifact that cannot
        corroborate --model-name is not published by guesswork.
        """
        (tmp_path / "model_card.yaml").write_text(yaml.dump({"title": "T"}))
        onnx = tmp_path / "model.onnx"
        onnx.write_text("onnx content")

        with pytest.raises(ValueError, match="No ONNX artifact in this folder"):
            upload_model(
                model_folder=tmp_path,
                network_type="lan",
                model_name="ddm",
                dry_run=True,
            )

        # ...and the override works
        assert (
            upload_model(
                model_folder=tmp_path,
                network_type="lan",
                model_name="ddm",
                dry_run=True,
                canonical_onnx=onnx,
            )
            is None
        )

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
        (tmp_path / "abc_lan_ddm__model.onnx").write_text("content")

        # Mock API
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.list_repo_files.return_value = []

        with patch("lanfactory.hf.upload.fetch_existing_manifest", return_value=None):
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
        """Files land under {network_type}/{model} in one commit.

        The upload is a single create_commit rather than upload_folder, so the
        folder, the root alias and the manifest cannot land separately (see
        tests/hf/test_dual_layout.py for the layout assertions).
        """
        # Create model_card.yaml
        yaml_content = {"tags": ["lan", "ssm"], "title": "Test"}
        with open(tmp_path / "model_card.yaml", "w") as f:
            yaml.dump(yaml_content, f)
        (tmp_path / "abc_lan_ddm__model.onnx").write_text("content")

        # Mock API
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.list_repo_files.return_value = []

        with patch("lanfactory.hf.upload.fetch_existing_manifest", return_value=None):
            upload_model(
                model_folder=tmp_path,
                network_type="lan",
                model_name="ddm",
                repo_id="test/repo",
            )

        mock_api.create_commit.assert_called_once()
        call_kwargs = mock_api.create_commit.call_args[1]
        assert call_kwargs["repo_id"] == "test/repo"
        destinations = {op.path_in_repo for op in call_kwargs["operations"]}
        assert "lan/ddm/abc_lan_ddm__model.onnx" in destinations
        assert "ddm.onnx" in destinations  # root alias HSSM resolves


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
