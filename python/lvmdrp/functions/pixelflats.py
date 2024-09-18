import os
import numpy as np
import pandas as pd


from lvmdrp import path, __version__ as drpver
from lvmdrp.functions import imageMethod as image_tasks
from cextern.fast_median.fast_median import fast_median_filter_2d
from lvmdrp.utils import metadata as md
from lvmdrp import main as drp


def detrend_pixelflats(mjd, camera, flat_expnums, bias_expnums=None, dark_expnums=None, use_pixmask=False, skip_done=True):

    frames = md.get_frames_metadata(mjd=mjd, overwrite=False, suffix="fits.gz").query("camera == @camera").sort_values("expnum")

    flats = frames.query("expnum in @flat_expnums")
    if bias_expnums is not None:
        biases = frames.query("expnum in @bias_expnums")
    else:
        biases = pd.DataFrame(data={"expnum": [None]*len(flats)})
    if dark_expnums is not None:
        darks = frames.query("expnum in @dark_expnums")
    else:
        darks = pd.DataFrame(data={"expnum": [None]*len(flats)})

    if biases is None and darks is None:
        # get latest bias and dark fiducial frames
        pass
    elif biases is None:
        # get latest fiducial bias
        pass
    elif darks is None:
        # get latest fiducial dark
        pass
    else:
        pass

    if use_pixmask:
        mpixmask_path = path.full("lvm_calib", mjd="pixelmasks", kind="pixmask", camera=camera)
    else:
        mpixmask_path = None

    # NOTE: repeating bias and darks if necessary
    if len(darks) < len(flats):
        n = len(flats) / len(darks)
        darks = darks.loc[darks.index.repeat(n)].reset_index(drop=True)
    if len(biases) < len(flats):
        n = len(flats) / len(biases)
        biases = biases.loc[biases.index.repeat(n)].reset_index(drop=True)


    dflat_paths = []
    for (_, bias), (_, dark), (_, flat) in zip(biases.iterrows(), darks.iterrows(), flats.iterrows()):

        # BIAS -----------------------
        if bias.expnum is not None:
            rbias_path = path.full("lvm_raw", hemi="s", mjd=mjd, camspec=camera, expnum=bias.expnum)
            pbias_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="p", imagetype="bias", expnum=bias.expnum, camera=camera)
            if skip_done and os.path.isfile(pbias_path):
                pass
            else:
                image_tasks.preproc_raw_frame(in_image=rbias_path, out_image=pbias_path, in_mask=mpixmask_path, assume_imagetyp="bias")
            pbias_path = pbias_path if os.path.isfile(pbias_path) else None
        else:
            pbias_path = None

        # DARKS ----------------------
        if dark.expnum is not None:
            rdark_path = path.full("lvm_raw", hemi="s", mjd=mjd, camspec=camera, expnum=dark.expnum)
            pdark_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="p", imagetype="dark", expnum=dark.expnum, camera=camera)
            ddark_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="dark", expnum=dark.expnum, camera=camera)
            if skip_done and os.path.isfile(pdark_path):
                pass
            else:
                image_tasks.preproc_raw_frame(in_image=rdark_path, out_image=pdark_path, in_mask=mpixmask_path, assume_imagetyp="dark")
                image_tasks.detrend_frame(in_image=pdark_path, out_image=ddark_path, in_bias=pbias_path, reject_cr=False)
            ddark_path = ddark_path if os.path.isfile(ddark_path) else None
        else:
            ddark_path = None

        # FLATS ----------------------
        rflat_path = path.full("lvm_raw", hemi="s", mjd=mjd, camspec=camera, expnum=flat.expnum)
        pflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="p", imagetype="pixflat", expnum=flat.expnum, camera=camera)
        dflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="pixflat", expnum=flat.expnum, camera=camera)

        if skip_done and os.path.isfile(dflat_path):
            pass
        else:
            image_tasks.preproc_raw_frame(in_image=rflat_path, out_image=pflat_path, in_mask=mpixmask_path, assume_imagetyp="pixflat")
            image_tasks.detrend_frame(in_image=pflat_path, out_image=dflat_path, in_bias=pbias_path, reject_cr=False, normalize_pixelflat=False)
        if os.path.isfile(dflat_path):
            dflat_paths.append(dflat_path)

    return dflat_paths


def combine_pixelflats(mjd, camera, flat_expnums, comb_stat=np.median, median_box=(31,31), skip_done=True):
    frames = md.get_frames_metadata(mjd=mjd, overwrite=False, suffix="fits.gz").query("camera == @camera").sort_values("expnum")

    flats = frames.query("expnum in @flat_expnums")
    dflat_paths = [path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="pixflat", expnum=flat.expnum, camera=camera) for _, flat in flats.iterrows()]
    cflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="c", imagetype="pixflat", expnum=f"{flats.expnum.min()}_{flats.expnum.max()}", camera=camera)
    mflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="m", imagetype="pixflat", expnum=f"{flats.expnum.min()}_{flats.expnum.max()}", camera=camera)
    fflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="f", imagetype="pixflat", expnum=f"{flats.expnum.min()}_{flats.expnum.max()}", camera=camera)
    paths = (cflat_path, mflat_path, fflat_path)

    if skip_done and all([os.path.isfile(xpath) for xpath in paths]):
        return None, None, None, paths
    else:
        flat_imgs = [image_tasks.loadImage(dflat_path) for dflat_path in dflat_paths]
        flat_data = [flat_img._data for flat_img in flat_imgs]
        flat_error = [flat_img._error for flat_img in flat_imgs]

        cflat_data = comb_stat(flat_data, axis=0)
        cflat_error = comb_stat(flat_error, axis=0) / np.sqrt(len(flat_error))
        cflat = image_tasks.Image(data=cflat_data, error=cflat_error)
        cflat.writeFitsData(cflat_path)

        # rolling mean spectrum of 100 rows and normalize by that to get rid of the lamp spectrum

        cflat_median = fast_median_filter_2d(cflat._data, median_box)
        mflat = (cflat / cflat_median)
        mflat.writeFitsData(mflat_path)

        fflat = cflat / mflat
        fflat.writeFitsData(fflat_path)

    return cflat, mflat, fflat, paths


def create_pixflats_60171(median_box=(31,31), skip_done=True):
    mjd = 60171
    flat_expnums = np.arange(3098, 3117+1)

    flats = md.get_frames_metadata(mjd=mjd, suffix="fits.gz", overwrite=False).query("expnum in @flat_expnums")

    calibs = drp.get_calib_paths(mjd=60171, from_sanbox=True)

    cameras = flats.camera.unique()
    flat_paths = dict.fromkeys(cameras)
    for camera in cameras:
        for expnum in flat_expnums:
            flat = flats.query("expnum == @expnum and camera == @camera").squeeze()

            rflat_path = path.full("lvm_raw", hemi="s", mjd=mjd, camspec=flat.camera, expnum=flat.expnum)
            pflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="p", imagetype="pixflat", expnum=flat.expnum, camera=flat.camera)
            dflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="pixflat", expnum=flat.expnum, camera=flat.camera)

            if skip_done and os.path.isfile(dflat_path):
                pass
            else:
                image_tasks.preproc_raw_frame(in_image=rflat_path, out_image=pflat_path, assume_imagetyp="pixflat")
                image_tasks.detrend_frame(in_image=pflat_path, out_image=dflat_path, in_bias=calibs["bias"][flat.camera], reject_cr=False, normalize_pixelflat=False)

        cflat_b, mflat_b, fflat_b, paths = combine_pixelflats(mjd=mjd, camera=camera, flat_expnums=flat_expnums, median_box=median_box, skip_done=skip_done)
        flat_paths[camera] = paths

    return flat_paths


def test_pixflats(mjd, camera, flat_expnums, target_expnum):
    target_mjd = drp.mjd_from_expnum(target_expnum)
    rframe_path = path.full("lvm_raw", hemi="s", mjd=target_mjd, camspec=camera, expnum=target_expnum)
    pframe_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=target_mjd, kind="p", imagetype="pixflat", expnum=target_expnum, camera=camera)
    dframe_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=target_mjd, kind="d", imagetype="pixflat", expnum=target_expnum, camera=camera)

    calibs = drp.get_calib_paths(mjd=mjd, from_sanbox=True)
    mflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="m", imagetype="pixflat", expnum=f"{flat_expnums.min()}_{flat_expnums.max()}", camera=camera)

    image_tasks.preproc_raw_frame(in_image=rframe_path, out_image=pframe_path)
    image_tasks.detrend_frame(in_image=pframe_path, out_image=dframe_path, in_bias=calibs["bias"][camera], in_pixelflat=mflat_path, reject_cr=False)

    return dframe_path


def compare_pixflats(mjd, camera, flat_expnums_a, flat_expnums_b):
    dframe_a_path = test_pixflats(mjd=mjd, camera=camera, flat_expnums=flat_expnums_a)
    dframe_b_path = test_pixflats(mjd=mjd, camera=camera, flat_expnums=flat_expnums_b)

    # TODO: do some plots



def create_pixflats(mjd, camera, flat_expnums, dark_expnums=None, bias_expnums=None, median_box=(31,31), skip_done=True):

    detrend_pixelflats(mjd=mjd, camera=camera, flat_expnums=flat_expnums, dark_expnums=dark_expnums, bias_expnums=bias_expnums, skip_done=skip_done)
    cflat_b, mflat_b, fflat_b, paths = combine_pixelflats(mjd=mjd, camera=camera, flat_expnums=flat_expnums, median_box=median_box, skip_done=skip_done)

    return paths
