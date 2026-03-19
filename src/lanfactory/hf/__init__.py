"""HuggingFace Hub integration for LANfactory.

This module provides utilities for uploading trained models to and
downloading models from HuggingFace Hub.
"""

DEFAULT_REPO_ID = "franklab/HSSM"
VALID_NETWORK_TYPES = ("lan", "cpn", "opn")

from lanfactory.hf.model_card import (  # noqa: E402
    load_model_card_yaml,
    generate_readme,
    ModelCardConfig,
)
from lanfactory.hf.upload import upload_model  # noqa: E402
from lanfactory.hf.download import download_model  # noqa: E402

__all__ = [
    "DEFAULT_REPO_ID",
    "VALID_NETWORK_TYPES",
    "load_model_card_yaml",
    "generate_readme",
    "ModelCardConfig",
    "upload_model",
    "download_model",
]
