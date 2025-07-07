"""
Configuration handler for MatOpt benchmark examples
"""

import json
import os
import sys
from typing import Dict, Any


class ConfigHandler:
    """Handle configuration loading and default values"""

    @staticmethod
    def load_config(
        config_file: str = None, example_name: str = None
    ) -> Dict[str, Any]:
        """
        Load configuration from JSON file or return defaults

        Args:
            config_file: Path to JSON configuration file
            example_name: Name of the example (used for default config)

        Returns:
            Dictionary with configuration parameters
        """
        # First, load default config if example_name is provided
        default_config = {}
        if example_name:
            # Look for default config in multiple locations
            default_paths = [
                f"configs/defaults/{example_name}.json",
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    f"configs/defaults/{example_name}.json",
                ),
            ]

            for default_path in default_paths:
                if os.path.exists(default_path):
                    with open(default_path, "r") as f:
                        default_config = json.load(f)
                    break

        # If config file is provided, load it and merge with defaults
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                user_config = json.load(f)
            return ConfigHandler.merge_configs(default_config, user_config)

        # Return default config (or empty dict if no defaults found)
        return default_config

    @staticmethod
    def merge_configs(
        default_config: Dict[str, Any], user_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge user configuration with defaults

        Args:
            default_config: Default configuration dictionary
            user_config: User-provided configuration

        Returns:
            Merged configuration dictionary
        """
        merged = default_config.copy()

        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if (
                    isinstance(value, dict)
                    and key in base_dict
                    and isinstance(base_dict[key], dict)
                ):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(merged, user_config)
        return merged

    @staticmethod
    def parse_command_line_config():
        """
        Parse configuration file from command line arguments

        Returns:
            Path to config file if provided, None otherwise
        """
        for i, arg in enumerate(sys.argv):
            if arg in ["--config", "-c"] and i + 1 < len(sys.argv):
                return sys.argv[i + 1]
        return None

    @staticmethod
    def get_config_basename(config_file: str) -> str:
        """
        Get the base name of the config file without extension

        Args:
            config_file: Path to config file

        Returns:
            Base name without .json extension
        """
        if config_file:
            return os.path.splitext(os.path.basename(config_file))[0]
        return None

    @staticmethod
    def save_config(config: Dict[str, Any], filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {filepath}")

    @staticmethod
    def validate_config(config: Dict[str, Any], required_fields: Dict[str, Any]):
        """
        Validate that all required fields are present in the configuration

        Args:
            config: Configuration dictionary to validate
            required_fields: Dictionary specifying required fields structure

        Raises:
            ValueError: If required fields are missing
        """

        def check_fields(config_dict, required_dict, path=""):
            for key, value in required_dict.items():
                current_path = f"{path}.{key}" if path else key

                if key not in config_dict:
                    raise ValueError(
                        f"Missing required configuration field: {current_path}"
                    )

                if isinstance(value, dict) and value:
                    if not isinstance(config_dict[key], dict):
                        raise ValueError(
                            f"Configuration field {current_path} must be a dictionary"
                        )
                    check_fields(config_dict[key], value, current_path)
                elif (
                    isinstance(value, list)
                    and len(value) > 0
                    and isinstance(value[0], str)
                ):
                    # List of required field names
                    for field in value:
                        if field not in config_dict[key]:
                            raise ValueError(
                                f"Missing required configuration field: {current_path}.{field}"
                            )

        check_fields(config, required_fields)
