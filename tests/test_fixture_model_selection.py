"""Regression test for the random model fixtures in conftest.py."""

from tests.conftest import UNSAMPLEABLE_MODELS


def test_simple_two_choices_fixture_only_returns_sampleable_two_choice_models(
    dummy_generator_config_simple_two_choices,
):
    """Every draw is a two-choice model that default sampling can simulate."""
    for _ in range(200):
        config = dummy_generator_config_simple_two_choices()
        model_config = config["model_config"]
        assert model_config["nchoices"] == 2
        assert "simulator_param_mappings" not in model_config
        assert config["generator_config"]["model"] not in UNSAMPLEABLE_MODELS
