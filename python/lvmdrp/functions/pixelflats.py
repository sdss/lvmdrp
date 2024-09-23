import os
import numpy as np
import pandas as pd
from copy import deepcopy as copy
from astropy.io import fits


from lvmdrp import log, path, __version__ as drpver
from lvmdrp.functions import imageMethod as image_tasks
from cextern.fast_median.fast_median import fast_median_filter_2d
from lvmdrp.utils import metadata as md
from lvmdrp import main as drp
from scipy import ndimage as ndi


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
                image_tasks.preproc_raw_frame(in_image=rbias_path, out_image=pbias_path, in_mask=mpixmask_path, assume_imagetyp="bias", replace_with_nan=False)
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
                image_tasks.preproc_raw_frame(in_image=rdark_path, out_image=pdark_path, in_mask=mpixmask_path, assume_imagetyp="dark", replace_with_nan=False)
                image_tasks.detrend_frame(in_image=pdark_path, out_image=ddark_path, in_bias=pbias_path, reject_cr=False, replace_with_nan=False)
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
            image_tasks.preproc_raw_frame(in_image=rflat_path, out_image=pflat_path, in_mask=mpixmask_path, assume_imagetyp="pixflat", replace_with_nan=False)
            image_tasks.detrend_frame(in_image=pflat_path, out_image=dflat_path, in_bias=pbias_path, reject_cr=False, normalize_pixelflat=False, replace_with_nan=False)
        if os.path.isfile(dflat_path):
            dflat_paths.append(dflat_path)

    return dflat_paths


def combine_pixelflats(mjd, camera, flat_expnums, comb_stat=np.median, median_box=(31,31), skip_done=True):
    frames = md.get_frames_metadata(mjd=mjd, overwrite=False, suffix="fits.gz").query("camera == @camera").sort_values("expnum")

    flats = frames.query("expnum in @flat_expnums")
    dflat_paths = [path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="d", imagetype="pixflat", expnum=flat.expnum, camera=camera) for _, flat in flats.iterrows()]
    cflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="c", imagetype="pixflat", expnum=f"{flats.expnum.min()}_{flats.expnum.max()}", camera=camera)

    if skip_done and os.path.isfile(cflat_path):
        cflat = image_tasks.loadImage(cflat_path)
        return cflat, cflat_path
    else:
        cflat = image_tasks.combineImages([image_tasks.loadImage(dflat_path) for dflat_path in dflat_paths], method="median", replace_with_nan=False)
        cflat.writeFitsData(cflat_path)

    return cflat, cflat_path


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

        cflat, cflat_path = combine_pixelflats(mjd=mjd, camera=camera, flat_expnums=flat_expnums, median_box=median_box, skip_done=skip_done)

        mflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="m", imagetype="pixflat", expnum=f"{flats.expnum.min()}_{flats.expnum.max()}", camera=camera)
        fflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="f", imagetype="pixflat", expnum=f"{flats.expnum.min()}_{flats.expnum.max()}", camera=camera)

        cflat_median = fast_median_filter_2d(cflat._data, median_box)
        mflat = (cflat / cflat_median)
        mflat.writeFitsData(mflat_path)

        fflat = cflat / mflat
        fflat.writeFitsData(fflat_path)

        flat_paths[camera] = (cflat_path, mflat_path, fflat_path)

    return flat_paths


def test_pixflats(mjd, camera, flat_expnums, target_expnum):
    target_mjd = drp.mjd_from_expnum(target_expnum)[0]
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


def do_for_quadrants(image_path, func, *args, **kwargs):
    image = image_tasks.loadImage(image_path)
    sections = image.getHdrValue("AMP? TRIMSEC").values()

    f_image = copy(image)
    for isec, sec in enumerate(sections):
        quad = image.getSection(sec)
        data, error, mask = quad._data, quad._error, quad._mask
        mask = ~mask*~np.isfinite(data)*error<=0
        ivar = np.where(mask, 1.0/(error**2), 0.0)
        _, f_data = func(data, ivar, *args, **kwargs)

        quad.setData(data=f_data)

        f_image.setSection(sec, quad)

    return image, f_image

    # with fits.open(image_path) as hdul:
    #     image = hdul['PRIMARY'].data
    #     error = hdul['ERROR'].data
    #     mask = hdul['BADPIX'].data
    #     mask = ~mask*~np.isfinite(image)*error<=0
    #     ivar = np.where(mask, 1.0/(error**2), 0.0)
    #     header = hdul[0].header
    #     q1x, q1y = _parse_ccd_section(header['HIERARCH AMP1 TRIMSEC'])
    #     q2x, q2y = _parse_ccd_section(header['HIERARCH AMP2 TRIMSEC'])
    #     q3x, q3y = _parse_ccd_section(header['HIERARCH AMP3 TRIMSEC'])
    #     q4x, q4y = _parse_ccd_section(header['HIERARCH AMP4 TRIMSEC'])
    #     _, f1 = func(image[q1y[0]:q1y[1],q1x[0]:q1x[1]], ivar[q1y[0]:q1y[1],q1x[0]:q1x[1]], *args, **kwargs)
    #     _, f2 = func(image[q2y[0]:q2y[1],q2x[0]:q2x[1]], ivar[q2y[0]:q2y[1],q2x[0]:q2x[1]], *args, **kwargs)
    #     _, f3 = func(image[q3y[0]:q3y[1],q3x[0]:q3x[1]], ivar[q3y[0]:q3y[1],q3x[0]:q3x[1]], *args, **kwargs)
    #     _, f4 = func(image[q4y[0]:q4y[1],q4x[0]:q4x[1]], ivar[q4y[0]:q4y[1],q4x[0]:q4x[1]], *args, **kwargs)

    #     filtered = image.copy()*0
    #     filtered[q1y[0]:q1y[1],q1x[0]:q1x[1]] = f1
    #     filtered[q2y[0]:q2y[1],q2x[0]:q2x[1]] = f2
    #     filtered[q3y[0]:q3y[1],q3x[0]:q3x[1]] = f3
    #     filtered[q4y[0]:q4y[1],q4x[0]:q4x[1]] = f4

    #     return image, filtered, header


def median_nan(image, ivar, size=51):
    image_tmp = np.where(ivar>0, image, np.NaN)
    return fast_median_filter_2d(image_tmp, size=(size, size))


def filtering(image, ivar, size=51, replace_with_nan=True, debug=False):

    minflat = 0.001
    min_flat_for_fit_mask = 0.99
    max_flat_for_fit_mask = 1.02

    # if replace_with_nan:
    #     image.apply_pixelmask()

    # initial model
    smooth = median_nan(image, ivar, size=size)
    # smooth = image.medianImg()
    if debug:
    #     smooth.writeFitsData('testmodel0.fits')
        fits.writeto('testmodel0.fits', smooth, overwrite=True)

    # initial flat by dividing by smoothed image, masking only where we have no data
    flat  =  (ivar>0)*(smooth>minflat)*image/(smooth*(smooth>minflat)+(smooth<=minflat))
    flat  += (smooth<=minflat)|(ivar<=0)  # set flat to 1 where masked
    if debug:
        fits.writeto('testflat0.fits', flat, overwrite=True)

    # dilate the mask, increasing sigma until not too large
    err = np.sqrt(1./(ivar+(ivar==0)))/(smooth*(image>0)+(image<=0))  # error image
    for nsig in [3.,3.5,4.,5.,10.,20.]:
        mask = (flat<(min_flat_for_fit_mask-nsig*err))|(flat>(max_flat_for_fit_mask+nsig*err))
        mask = ndi.binary_dilation(mask)
        frac = np.sum(mask>0)/float(np.sum(ivar>0))
        if frac<0.05 :
            break
    print("Used nsig = {}, frac = {:4.3f}".format(nsig,frac))

    # https://github.com/desihub/desispec/blob/main/bin/desi_compute_pixel_flatfield#L619

    # now start iterating smoothing and filtering the flat, ignoring newly mased pixels in the smoothing
    mask = mask | (ivar==0)
    smooth = median_nan(image, ~mask, size=size)
    flat  =  (ivar>0)*(smooth>minflat)*image/(smooth*(smooth>minflat)+(smooth<=minflat))   # divide by model
    flat  += (smooth<=minflat)|(ivar<=0)  # set flat to 1 where no data

    if debug:
        fits.writeto('testmask.fits', mask.astype(int), overwrite=True)
        fits.writeto('testivar.fits', ivar, overwrite=True)
        fits.writeto('testmodel.fits', smooth, overwrite=True)
        fits.writeto('testflat.fits', flat, overwrite=True)

    return image, smooth

def filter_image(image_path):
    return do_for_quadrants(image_path, filtering)
    #return do_for_quadrants(image_path, median_nan, size=51)


def job(mflat_path, pixflat_path, fflat_path):
    log.info(f"Reading : {mflat_path}")
    image, filtered = filter_image(mflat_path)
    flat = image/filtered
    flat._data = np.where((flat._data>0.01)*(np.isfinite(flat._data)), flat._data, 1.0)
    # outf = fits.HDUList(fits.PrimaryHDU(filtered))
    # log.info("Writing :", filt_path)
    # outf.writeto(filt_path, overwrite=True)
    # outf.close()
    log.info(f"Writing : {pixflat_path}")
    flat.writeFitsData(pixflat_path)

    log.info(f"Writing : {fflat_path}")
    fflat = image / flat
    fflat.writeFitsData(fflat_path)

    # out = fits.HDUList(fits.PrimaryHDU(flat))
    # fflat = fits.HDUList(fits.PrimaryHDU(image / flat, header=header))
    # fflat.append(fits.ImageHDU(np.sqrt(image), name="ERROR"))
    # fflat.append(fits.ImageHDU(np.zeros_like(image, dtype="uint8"), name="BADPIX"))
    # out.writeto(pixflat_path, overwrite=True)
    # fflat.writeto(fflat_path, overwrite=True)
    # out.close()
    # fflat.close()


def create_pixflats(mjd, camera, flat_expnums, dark_expnums=None, bias_expnums=None, median_box=(31,31), skip_done=True):

    detrend_pixelflats(mjd=mjd, camera=camera, flat_expnums=flat_expnums, dark_expnums=dark_expnums, bias_expnums=bias_expnums, skip_done=skip_done, use_pixmask=True)
    cflat, cflat_path = combine_pixelflats(mjd=mjd, camera=camera, flat_expnums=flat_expnums, median_box=median_box, skip_done=skip_done)

    mflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="m", imagetype="pixflat", expnum=f"{flat_expnums.min()}_{flat_expnums.max()}", camera=camera)
    fflat_path = path.full("lvm_anc", drpver=drpver, tileid=11111, mjd=mjd, kind="f", imagetype="pixflat", expnum=f"{flat_expnums.min()}_{flat_expnums.max()}", camera=camera)

    cflat_median = fast_median_filter_2d(cflat._data, median_box)
    mflat = (cflat / cflat_median)
    mflat.writeFitsData(mflat_path)

    fflat = cflat / mflat
    fflat.writeFitsData(fflat_path)

    paths = (cflat_path, mflat_path, fflat_path)

    return paths
