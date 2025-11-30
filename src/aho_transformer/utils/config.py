"""Configuration utilities for AhoTransformer.

This module provides utilities for loading and managing configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def save_config(config: dict[str, Any], config_path: str | Path) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save.
        config_path: Path to save the configuration file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: dict[str, Any], override_config: dict[str, Any]) -> dict[str, Any]:
    """Merge two configuration dictionaries.

    The override_config values take precedence over base_config values.
    Nested dictionaries are merged recursively.

    Args:
        base_config: Base configuration dictionary.
        override_config: Configuration dictionary with overrides.

    Returns:
        Merged configuration dictionary.
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result
