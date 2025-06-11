import yaml


def yaml_loads(text):
    return yaml.safe_load(text)


def yaml_dumps(obj):
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
