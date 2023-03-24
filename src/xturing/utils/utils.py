import yaml


def read_yamls(config_path):
    conf = {}

    with open(config_path) as f:
        conf.update(yaml.safe_load(f))

    return conf
