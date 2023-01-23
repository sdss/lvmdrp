# gist taken form https://gist.github.com/jdthorpe/313cafc6bdaedfbc7d8c32fcef799fbf
# author: jdthorpe

import os
import yaml
from types import SimpleNamespace


class DictLoader(yaml.Loader):
    pass

class DictDumper(yaml.Dumper):
    pass


class NamespaceLoader(yaml.Loader):
    pass


class NamespaceDumper(yaml.Dumper):
    pass

def _join(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.join(*seq)

def _format(loader, node):
    to_fmt = {key: f"{{{key}}}" for key in ["path", "label", "kind"]}
    str, keys = loader.construct_sequence(node)
    to_fmt.update(keys.__dict__)
    return str.format(**to_fmt)

def _construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return SimpleNamespace(**dict(loader.construct_pairs(node)))


def _ns_representer(dumper, data):
    return dumper.represent_mapping(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.__dict__.items()
    )

DictLoader.add_constructor(
    "!join", _join
)
DictLoader.add_constructor(
    "!fmt", _format
)

NamespaceDumper.add_representer(SimpleNamespace, _ns_representer)
NamespaceLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)
NamespaceLoader.add_constructor(
    "!join", _join
)
NamespaceLoader.add_constructor(
    "!fmt", _format
)
