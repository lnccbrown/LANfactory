"""Tests for download.py module."""

from unittest.mock import patch

import pytest

from lanfactory.hf.download import (
    DEFAULT_REPO_ID,
    download_model,
)


class TestDownloadModel:
    """Tests for download_model function."""

    def test_raises_if_invalid_network_type(self, tmp_path):
        """Test raises ValueError for invalid network_type."""
        with pytest.raises(ValueError, match="network_type must be one of"):
            download_model(
                network_type="invalid",
                model_name="ddm",
                output_folder=tmp_path / "output",
            )

    def test_raises_if_output_exists_without_force(self, tmp_path):
        """Test raises FileExistsError if output folder exists and force=False."""
        output_folder = tmp_path / "output"
        output_folder.mkdir()

        with pytest.raises(FileExistsError, match="already exists"):
            download_model(
                network_type="lan",
                model_name="ddm",
                output_folder=output_folder,
                force=False,
            )

    @patch("huggingface_hub.list_repo_files")
    @patch("huggingface_hub.hf_hub_download")
    def test_downloads_files_to_output_folder(
        self, mock_download, mock_list_files, tmp_path
    ):
        """Test files are downloaded to output folder."""
        output_folder = tmp_path / "output"

        # Mock list of files in repo
        mock_list_files.return_value = [
            "lan/ddm/model.onnx",
            "lan/ddm/config.pickle",
            "cpn/angle/model.onnx",  # Should not be downloaded
        ]

        # Mock download - return a temp file path
        temp_file = tmp_path / "temp_download.onnx"
        temp_file.write_text("onnx content")
        mock_download.return_value = str(temp_file)

        result = download_model(
            network_type="lan",
            model_name="ddm",
            output_folder=output_folder,
        )

        assert result == output_folder
        assert output_folder.exists()

        # Check that download was called for correct files
        assert mock_download.call_count == 2

    @patch("huggingface_hub.list_repo_files")
    def test_raises_if_no_files_found(self, mock_list_files, tmp_path):
        """Test raises FileNotFoundError if no files match."""
        output_folder = tmp_path / "output"

        # Mock empty file list for the path
        mock_list_files.return_value = [
            "cpn/angle/model.onnx",  # Wrong network type
        ]

        with pytest.raises(FileNotFoundError, match="No files found"):
            download_model(
                network_type="lan",
                model_name="ddm",
                output_folder=output_folder,
            )

    @patch("huggingface_hub.list_repo_files")
    @patch("huggingface_hub.hf_hub_download")
    def test_applies_include_patterns(self, mock_download, mock_list_files, tmp_path):
        """Test include patterns filter downloaded files."""
        output_folder = tmp_path / "output"

        mock_list_files.return_value = [
            "lan/ddm/model.onnx",
            "lan/ddm/history.csv",
        ]

        temp_file = tmp_path / "temp.onnx"
        temp_file.write_text("content")
        mock_download.return_value = str(temp_file)

        download_model(
            network_type="lan",
            model_name="ddm",
            output_folder=output_folder,
            include_patterns=["*.onnx"],
        )

        # Only .onnx file should be downloaded
        assert mock_download.call_count == 1

    @patch("huggingface_hub.list_repo_files")
    @patch("huggingface_hub.hf_hub_download")
    def test_applies_exclude_patterns(self, mock_download, mock_list_files, tmp_path):
        """Test exclude patterns filter downloaded files."""
        output_folder = tmp_path / "output"

        mock_list_files.return_value = [
            "lan/ddm/model.onnx",
            "lan/ddm/history.csv",
        ]

        temp_file = tmp_path / "temp.onnx"
        temp_file.write_text("content")
        mock_download.return_value = str(temp_file)

        download_model(
            network_type="lan",
            model_name="ddm",
            output_folder=output_folder,
            exclude_patterns=["*.csv"],
        )

        # Only .onnx file should be downloaded
        assert mock_download.call_count == 1

    @patch("huggingface_hub.list_repo_files")
    @patch("huggingface_hub.hf_hub_download")
    def test_force_overwrites_existing(self, mock_download, mock_list_files, tmp_path):
        """Test force=True allows overwriting existing folder."""
        output_folder = tmp_path / "output"
        output_folder.mkdir()

        mock_list_files.return_value = ["lan/ddm/model.onnx"]

        temp_file = tmp_path / "temp.onnx"
        temp_file.write_text("content")
        mock_download.return_value = str(temp_file)

        # Should not raise
        result = download_model(
            network_type="lan",
            model_name="ddm",
            output_folder=output_folder,
            force=True,
        )

        assert result == output_folder


class TestDefaults:
    """Tests for default values."""

    def test_default_repo_id(self):
        """Test default repo ID is franklab/HSSM."""
        assert DEFAULT_REPO_ID == "franklab/HSSM"
