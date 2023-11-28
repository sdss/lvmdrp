# encoding: utf-8
#

import importlib
import pytest

import lvmdrp.utils.metadata
from lvmdrp.utils.metadata import get_frames_metadata


@pytest.fixture(autouse=True)
def mock_meta():
    """ fixture to reload the metadata module

    This is so the global METADATA_PATH variable
    is using the correct test paths
    """
    importlib.reload(lvmdrp.utils.metadata)


def test_get_frames_metadata(make_fits):
    """ test we can extract metadata from a fits file """
    meta = get_frames_metadata(61234)
    meta = meta.sort_values('camera')
    assert len(meta) == 3
    assert 61234 in meta['mjd'].unique()
    assert 1111 in meta['tileid'].unique()
    assert 6817 in meta['expnum'].unique()
    assert meta.iloc[0]['name'] == 'sdR-s-b1-00006817.fits'
    assert set(meta['camera']) == {'b1', 'b2', 'b3'}
