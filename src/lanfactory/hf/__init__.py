"""HuggingFace Hub integration for LANfactory.

This module provides utilities for uploading trained models to and
downloading models from HuggingFace Hub.
"""

DEFAULT_REPO_ID = "franklab/HSSM"
# Matches what franklab/HSSM actually declares. The *code* in this ecosystem is
# MIT, but a model card describes the published artifact, so it follows the
# artifact repo — auto-generated cards claiming MIT would contradict it.
DEFAULT_LICENSE = "bsd-2-clause"
# gonogo included: the trainers already build gonogo networks (cli/utils.py
# train_output_type_dict) and HSSM resolves "{model}_gonogo.onnx", so excluding
# it here made a trainable, loadable network type unpublishable.
VALID_NETWORK_TYPES = ("lan", "cpn", "opn", "gonogo")

from lanfactory.hf.download import download_model
from lanfactory.hf.model_card import (
    ModelCardConfig,
    generate_readme,
    load_model_card_yaml,
)
from lanfactory.hf.upload import upload_model

__all__ = [
    "DEFAULT_LICENSE",
    "DEFAULT_REPO_ID",
    "VALID_NETWORK_TYPES",
    "ModelCardConfig",
    "download_model",
    "generate_readme",
    "load_model_card_yaml",
    "upload_model",
]
