import json
import importlib.resources


def load_data_config(file_name: str) -> dict:
    with importlib.resources.open_text("task_offloading_moo.data.config", file_name + ".json") as f:
        config_data = json.load(f)

    return config_data
