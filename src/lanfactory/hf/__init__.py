"""HuggingFace Hub integration for LANfactory.

This module provides utilities for uploading trained models to and
downloading models from HuggingFace Hub.
"""

from lanfactory.hf.download import download_model
from lanfactory.hf.model_card import (
    ModelCardConfig,
    generate_readme,
    load_model_card_yaml,
)
from lanfactory.hf.upload import upload_model

DEFAULT_REPO_ID = "franklab/HSSM"
VALID_NETWORK_TYPES = ("lan", "cpn", "opn")

__all__ = [
    "DEFAULT_REPO_ID",
    "VALID_NETWORK_TYPES",
    "ModelCardConfig",
    "download_model",
    "generate_readme",
    "load_model_card_yaml",
    "upload_model",
]
