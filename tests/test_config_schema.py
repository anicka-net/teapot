from pathlib import Path

import pytest

from teapot.compose import parse_config


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/defconfig",
        "configs/test-llama-8b.config",
        "configs/karma-electric.config",
        "configs/apertus-70b-secular.config",
        "configs/cve-backport.config",
    ],
)
def test_shipped_configs_parse(config_path):
    parsed = parse_config(config_path)
    assert "modules" in parsed
    assert "output" in parsed

