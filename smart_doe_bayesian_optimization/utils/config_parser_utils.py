import json

def config_parser(config_file_path: str) -> dict:
    """
    Parse a configuration file and return a dictionary with the configuration parameters.

    :param config_file_path: The path to the configuration file.
    :return: A dictionary with the configuration parameters.
    """
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config
