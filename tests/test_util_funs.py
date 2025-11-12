from pathlib import Path
from unittest.mock import mock_open, patch

from lanfactory.utils.util_funs import save_configs


def test_save_configs_with_string_path():
    """Test that save_configs works with string path instead of Path object."""
    model_id = "test_model"
    save_folder = "/some/fake/path/string_folder"  # No tmp_path needed
    network_config = {"test": "config"}
    train_config = {"test": "train"}

    with (
        patch("lanfactory.utils.util_funs.pickle.dump") as mock_dump,
        patch("builtins.open", mock_open()),
        patch("pathlib.Path.mkdir") as mock_mkdir,
    ):
        save_configs(
            model_id=model_id,
            save_folder=save_folder,
            network_config=network_config,
            train_config=train_config,
        )

        # Assert mkdir was called with correct arguments
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Assert pickle.dump was called
        assert mock_dump.call_count == 2


def test_save_configs_with_path_object():
    """Test that save_configs works with Path object."""
    model_id = "test_model"
    save_folder = Path("/some/fake/path/string_folder")  # No tmp_path needed
    network_config = {"test": "config"}
    train_config = {"test": "train"}

    with (
        patch("lanfactory.utils.util_funs.pickle.dump") as mock_dump,
        patch("builtins.open", mock_open()),
        patch("pathlib.Path.mkdir") as mock_mkdir,
    ):
        save_configs(
            model_id=model_id,
            save_folder=save_folder,
            network_config=network_config,
            train_config=train_config,
        )

        # Assert mkdir was called with correct arguments
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Assert pickle.dump was called
        assert mock_dump.call_count == 2


def test_save_configs_existing_directory():
    """Test that save_configs works when directory already exists."""
    model_id = "test_model"
    save_folder = "/fake/existing_folder"  # No tmp_path needed
    network_config = {"test": "config"}
    train_config = {"test": "train"}

    with (
        patch("lanfactory.utils.util_funs.pickle.dump") as mock_dump,
        patch("builtins.open", mock_open()),
        patch("pathlib.Path.mkdir") as mock_mkdir,
    ):
        # Should not raise an error
        save_configs(
            model_id=model_id,
            save_folder=save_folder,
            network_config=network_config,
            train_config=train_config,
        )

        # Assert mkdir was called (exist_ok=True handles existing directories)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Assert pickle.dump was still called
        assert mock_dump.call_count == 2
