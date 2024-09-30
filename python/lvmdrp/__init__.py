#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
from sdsstools import get_config, get_logger, get_package_version
from tree import Tree
from sdss_access.path import Path
import subprocess


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
def setup_paths(release: str = 'sdsswork', replant: bool = False):
    tree = Tree(release)
    if replant:
        tree.replant_tree(release)
    return Path(release='sdsswork')


path = setup_paths()


__version__ = os.getenv("LVMDRP_VERSION") or get_package_version(path=__file__, package_name=NAME)


# NOTE: taken from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


DRP_COMMIT = get_git_revision_short_hash()