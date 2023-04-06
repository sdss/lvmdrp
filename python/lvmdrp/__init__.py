#!/usr/bin/env python
# encoding: utf-8

import pathlib
from sdsstools import get_config, get_logger, get_package_version
from tree import Tree
from sdss_access.path import Path


NAME = 'lvmdrp'


# init the logger
log = get_logger(NAME)


# get the DRP package and reduction config
cpath = pathlib.Path(__file__).parent / 'etc/lvmdrp.yaml'
try:
    config = get_config('lvmdrp', config_file=cpath, allow_user=True)
except FileNotFoundError:
    config = {}


# setup the sdss tree environment and paths
def setup_paths(release: str = 'sdss5', replant: bool = False):
    tree = Tree(release)
    if replant:
        tree.replant_tree(release)
    return Path(release='sdss5')


path = setup_paths()


__version__ = get_package_version(path=__file__, package_name=NAME)
