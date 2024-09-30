#!/usr/bin/env python
# encoding: utf-8

import os
import pathlib
from sdsstools import get_config, get_logger, get_package_version
from tree import Tree
from sdss_access.path import Path
import subprocess
import inspect


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



# Function to find the root directory of the Git repository
def get_git_root(path):
    try:
        # Run the git rev-parse command to get the top-level directory
        git_root = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=path
        ).strip().decode('utf-8')
        return git_root
    except subprocess.CalledProcessError:
        return None

# Function to get the current commit hash
def get_git_commit_hash(repo_path):
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path
        ).strip().decode('ascii')
        return commit_hash
    except subprocess.CalledProcessError as e:
        print(f"Error getting Git commit hash: {e}")
        return None

# Get the current file location (__init__.py) and traverse to the git root
current_file = os.path.abspath(inspect.getfile(inspect.currentframe()))  # Absolute path to __init__.py
current_dir = os.path.dirname(current_file)  # Directory of __init__.py

# Find the git root directory
git_root = get_git_root(current_dir)
# if git_root:
#     commit_hash = get_git_commit_hash(git_root)
#     print(f"Current commit hash: {commit_hash}")
# else:
#     print("Not inside a Git repository")


DRP_COMMIT = git_root 
