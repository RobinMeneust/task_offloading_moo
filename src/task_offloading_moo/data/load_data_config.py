"""This file contains the function to load data configuration files."""

import json
import importlib.resources


def load_data_config(file_name: str) -> dict:
    """Load a data configuration file.

    Args:
        file_name (str): Name of the file to load.

    Returns:
        dict: Data configuration.
    """
    with importlib.resources.open_text("task_offloading_moo.data.config", file_name + ".json") as f:
        config_data = json.load(f)

    return config_data
