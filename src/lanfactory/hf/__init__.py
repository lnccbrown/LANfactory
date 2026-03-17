"""HuggingFace Hub integration for LANfactory.

This module provides utilities for uploading trained models to and
downloading models from HuggingFace Hub.
"""

from lanfactory.hf.model_card import (
    load_model_card_yaml,
    generate_readme,
    ModelCardConfig,
)
from lanfactory.hf.upload import upload_model
from lanfactory.hf.download import download_model

# Default repository for official HSSM models
DEFAULT_REPO_ID = "franklab/HSSM"

__all__ = [
    "DEFAULT_REPO_ID",
    "load_model_card_yaml",
    "generate_readme",
    "ModelCardConfig",
    "upload_model",
    "download_model",
]
