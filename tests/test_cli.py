# encoding: utf-8
#
# test_cli.py
#
# The command line tools are console_scripts entry points of the lvmdrp package

import configparser
import importlib
import pathlib

import pytest
from click.testing import CliRunner

SETUP_CFG = pathlib.Path(__file__).parents[1] / 'setup.cfg'


def _entry_points():
    cfg = configparser.ConfigParser()
    cfg.read(SETUP_CFG)
    lines = cfg['options.entry_points']['console_scripts'].strip().splitlines()
    return dict(tuple(part.strip() for part in line.split('=')) for line in lines)


def test_no_copied_scripts():
    """ scripts are copied at install time and go stale; the tools must be entry points """
    cfg = configparser.ConfigParser()
    cfg.read(SETUP_CFG)
    assert 'scripts' not in cfg['options']
    assert set(_entry_points()) == {'drp', 'pix2wave', 'envcheck'}


@pytest.mark.parametrize('name', ['drp', 'pix2wave', 'envcheck'])
def test_entry_point_resolves(name):
    module, func = _entry_points()[name].split(':')
    assert callable(getattr(importlib.import_module(module), func))


@pytest.mark.parametrize('args', [['--help'], ['run', '--help']])
def test_drp_help(args):
    from lvmdrp.cli.drp import cli
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert 'Usage' in result.output
