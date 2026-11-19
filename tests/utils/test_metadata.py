# encoding: utf-8
#

import importlib
import pandas as pd
import pytest
from astropy.io import fits

from lvmdrp import path
import lvmdrp.utils.metadata
from lvmdrp.utils.metadata import (get_frames_metadata, extract_from_filename, _collect_header_data,
                                   update_summary_file)


@pytest.fixture(autouse=True)
def mock_meta():
    """ fixture to reload the metadata module

    This is so the global METADATA_PATH variable
    is using the correct test paths
    """
    importlib.reload(lvmdrp.utils.metadata)


def test_get_frames_metadata(make_fits):
    """ test we can extract metadata from a fits file """
    make_fits(mjd=61234, cameras=['b1', 'b2', 'b3'], expnum=6817)
    meta = get_frames_metadata(61234)
    meta = meta.sort_values('camera')
    assert len(meta) == 3
    assert 61234 in meta['mjd'].unique()
    assert 11111 in meta['tileid'].unique()
    assert '0011XX' in meta['tilegrp'].unique()
    assert 6817 in meta['expnum'].unique()
    assert meta.iloc[0]['name'] == 'sdR-s-b1-00006817.fits'
    assert set(meta['camera']) == {'b1', 'b2', 'b3'}


def test_extract_from_filename():
    """ test we can extract correct info from a sframe file """
    path = 'sas/sdsswork/lvm/spectro/redux/1.0.2/1054XX/1054755/60297/lvmSFrame-00009532.fits'
    meta = extract_from_filename(path)
    assert meta == ('1054755', '60297', '9532')


def test_collect_header_data(make_framefits):
    """ test we can extract the correct header data from the frame file """
    sframe = make_framefits(tileid=1054755, mjd=60297, expnum=9532)
    hdr_data = _collect_header_data(sframe)


    expdata = {'drpver': '0.1.1', 'drpqual': 0, 'dpos': 0, 'object': 'tile_id=1054755',
               'obstime': '2023-12-19T00:47:39.095', 'sci_ra': 65.949555, 'sci_dec': 15.348684,
               'sci_pa': 66.7, 'sci_amass': 1.857, 'sci_astsrc': 'GDR coadd', 'sci_kmpos': -87.5,
               'sci_focpos': 36.58, 'sci_alt': 32.5605,
               'sci_sh_hght': 165.52885,'sci_moon_sep': 29.8097, 'skye_ra': 21.008216, 'skye_dec': -22.933382,
               'skye_pa': 0.0, 'skye_amass': 1.013, 'skye_astsrc': 'GDR coadd', 'skye_kmpos': -37.5,
               'skye_focpos': 36.19, 'skye_name': 'WHAM_south_02', 'skye_alt': 80.9755, 'sci_skye_sep': 58.2987,
               'skye_sh_hght': 164.79533, 'skye_moon_sep': 18.4705,
               'skyw_ra': 58.011871, 'skyw_dec': 11.817184, 'skyw_pa': 0.0, 'skyw_amass': 1.555,
               'skyw_astsrc': 'GDR coadd', 'skyw_kmpos': -54.51, 'skyw_focpos': 37.11, 'skyw_name': 'grid087',
               'skyw_alt': 40.0142, 'sci_skyw_sep': 8.4839, 'skyw_sh_hght': 165.41738,
               'skyw_moon_sep': 27.7977, 'moon_ra': 348.42157, 'moon_dec': -7.55955,
               'moon_phase': 79.91, 'moon_fli': 0.4136, 'sun_alt': -13.3779, 'moon_alt': 46.8744,
               'std_mean_senb': 2.0e-14, 'std_mean_senr': 2.0e-14, 'std_mean_senz': 2.0e-14,
               'sci_mean_senb': 2.0e-14, 'sci_mean_senr': 2.0e-14, 'sci_mean_senz': 2.0e-14,
               'mod_mean_senb': 2.0e-14, 'mod_mean_senr': 2.0e-14, 'mod_mean_senz': 2.0e-14,
               'std_rms_senb': 2.0e-14, 'std_rms_senr': 2.0e-14, 'std_rms_senz': 2.0e-14,
               'sci_rms_senb': 2.0e-14, 'sci_rms_senr': 2.0e-14, 'sci_rms_senz': 2.0e-14,
               'mod_rms_senb': 2.0e-14, 'mod_rms_senr': 2.0e-14, 'mod_rms_senz': 2.0e-14,
               'fluxcal': "MOD"}

    assert {k: hdr_data[k] for k in expdata} == expdata
    # the sky-line plane columns are there, empty for a frame without the keywords
    skyplane = {k: v for k, v in hdr_data.items() if k.startswith('skyplane_')}
    assert len(skyplane) == 59 and all(v is None for v in skyplane.values())
    # and the flat-field correction columns, empty too
    corr = {k: v for k, v in hdr_data.items() if k.startswith(('ffcorr_', 'flatgrad_'))}
    assert len(corr) == 48 and all(v is None for v in corr.values())
    assert set(hdr_data) == set(expdata) | set(skyplane) | set(corr)


def test_collect_header_data_skyplane(make_framefits):
    """ test the sky-line plane keywords are collected when present """
    sframe = make_framefits(tileid=1054755, mjd=60297, expnum=9533)
    with fits.open(sframe, mode='update') as hdulist:
        hdulist[0].header['HIERARCH SKYPLANE Z GX'] = 6.25
        hdulist[0].header['HIERARCH SKYPLANE SKYE B F1'] = 1.02
        hdulist[0].header['HIERARCH SKYPLANE ZOH N'] = 1400
    hdr_data = _collect_header_data(sframe)
    assert hdr_data['skyplane_z_gx'] == 6.25
    assert hdr_data['skyplane_skye_b_f1'] == 1.02
    assert hdr_data['skyplane_zoh_n'] == 1400
    assert hdr_data['skyplane_z_gy'] is None


def test_collect_header_data_corrections(make_framefits):
    """ test the flat-field correction keywords are collected when present """
    sframe = make_framefits(tileid=1054755, mjd=60297, expnum=9533)
    with fits.open(sframe, mode='update') as hdulist:
        hdulist[0].header['HIERARCH FFCORR APPLIED'] = True
        hdulist[0].header['HIERARCH FFCORR SOURCE'] = 'direct'
        hdulist[0].header['HIERARCH FFCORR Z GX'] = 6.13
        hdulist[0].header['HIERARCH FLATGRAD DB GY'] = 0.25
    hdr_data = _collect_header_data(sframe)
    assert hdr_data['ffcorr_applied'] is True
    assert hdr_data['ffcorr_source'] == 'direct'
    assert hdr_data['ffcorr_z_gx'] == 6.13
    assert hdr_data['flatgrad_db_gy'] == 0.25
    assert hdr_data['ffcorr_b_gx'] is None



def test_update_summary_file(make_fits, make_framefits):
    """ test we can update the summary file """

    data = [dict(tileid=1054755, mjd=60297, expnum=9532),
            dict(tileid=1022456, mjd=60300, expnum=9540)]
    for item in data:
        # setup
        make_fits(tileid=item['tileid'], mjd=item['mjd'], cameras=['b1', 'b2', 'b3'], expnum=item['expnum'])
        get_frames_metadata(item['mjd'])

        sframe = make_framefits(**item)
        update_summary_file(sframe, tileid=item['tileid'], mjd=item['mjd'],
                            expnum=item['expnum'], master_mjd=60255)

    drpall = path.full('lvm_drpall', drpver='test')
    drpall = drpall.replace('.fits', '.h5')

    assert path.exists('', full=drpall)
    # the sky-line plane columns are written as floats, with -999 where missing
    summary = pd.read_hdf(drpall, key='summary')
    assert summary['skyplane_z_gx'].dtype == 'float64'
    assert (summary['skyplane_z_gx'] == -999).all()
    # and the flat-field correction columns: floats with -999, strings with 'None'
    assert summary['ffcorr_z_gx'].dtype == 'float64' and (summary['ffcorr_z_gx'] == -999).all()
    assert (summary['ffcorr_source'] == 'None').all()
