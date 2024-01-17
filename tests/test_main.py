# encoding: utf-8
#

import os


def test_mock_sas():
    assert 'pytest' in os.getenv("SAS_BASE_DIR")
    assert os.getenv("LVMDRP_VERSION") == 'test'


def test_mock_drpver():
    from lvmdrp import __version__
    assert __version__ == 'test'