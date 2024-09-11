

import pathlib
import pytest

from lvmdrp.utils.hdrfix import read_hdrfix_file, apply_hdrfix, write_hdrfix_file


# make fake hdr
header = {
    'MJD': 60255,
    'CCD': 'b1',
    'EXPOSURE': 7230,
    'TILE_ID': 12345,
    'OBSERVAT': 'LCO',
    'SKYWNAME': 'old_west_name',
    'SKYENAME': 'old_east_name',
}


@pytest.fixture()
def mock_path(mocker):
    """ fixture to mock the return path of the hdrfix file """
    def _mock(mjd):
        path = pathlib.Path(__file__).parent.parent / 'data' / f'lvmHdrFix-{mjd}.yaml'
        mocker.patch('lvmdrp.utils.hdrfix.get_hdrfix_path', return_value=path)
    return _mock


def test_read_hdrfix_file(mock_path):
    """ test we can read the hdrfix file """
    mock_path(60255)
    df = read_hdrfix_file(60255)
    assert len(df) == 6
    assert df['fileroot'].iloc[0] == 'sdR-*-*-00007334'
    assert df['keyword'].iloc[0] == 'SKYENAME'
    assert df['value'].iloc[0] == 'grid125'


def test_read_hdrfix_file_no_file(mock_path):
    """ test when the hdrfix file does not exist """
    mock_path(60256)
    df = read_hdrfix_file(60256)
    assert df is None


@pytest.mark.parametrize('input, exp',
                         [(('b1', 7334), {'SKYWNAME': 'WHAM_south_08', 'SKYENAME': 'grid125'}),
                          (('r1', 7334), {'SKYWNAME': 'old_west_name', 'SKYENAME': 'grid125'}),
                          (('z1', 7330), {'SKYWNAME': 'old_west_name', 'SKYENAME': 'old_east_name'}),
                          (('z2', 7324), {'SKYWNAME': 'WHAM_south_02', 'SKYENAME': 'grid103'}),
                          (('b2', 7326), {'SKYWNAME': 'WHAM_south_02', 'SKYENAME': 'grid103'}),
                          (('r3', 7326), {'SKYWNAME': 'old_west_name', 'SKYENAME': 'old_east_name'}),
                          (('r1', 7329), {'SKYWNAME': 'WHAM_south_02', 'SKYENAME': 'grid125'}),
                          (('z2', 7329), {'SKYWNAME': 'old_west_name', 'SKYENAME': 'old_east_name'}),
                          ]
                         )
def test_apply_hdrfix(mock_path, input, exp):
    """ test we can apply the hdrfix """
    mock_path(60255)
    hdr = header.copy()
    hdr.update(dict(zip(['CCD', 'EXPOSURE'], input)))
    new_hdr = apply_hdrfix(60255, hdr=hdr.copy())

    assert new_hdr['SKYWNAME'] == exp['SKYWNAME']
    assert new_hdr['SKYENAME'] == exp['SKYENAME']


def test_write_hdrfix(tmp_path, mocker):
    """ test we can write a new hdrfix file """
    path = tmp_path / 'data' / 'lvmHdrFix-60257.yaml'
    mocker.patch('lvmdrp.utils.hdrfix.get_hdrfix_path', return_value=path)

    write_hdrfix_file(60257, 'sdR-*-*-00007334', 'SKYENAME', 'grid125')

    read_hdrfix_file.cache_clear()
    df = read_hdrfix_file(60257)
    assert path.exists()
    assert len(df) == 1
    assert df['fileroot'].iloc[0] == 'sdR-*-*-00007334'
    assert df['keyword'].iloc[0] == 'SKYENAME'
    assert df['value'].iloc[0] == 'grid125'