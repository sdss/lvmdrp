import os
from multiprocessing import Pool, cpu_count

from astropy.io import fits as pyfits

from lvmdrp.core.header import Header
from lvmdrp.core.image import Image
from lvmdrp.functions.headerMethod import *
from lvmdrp.functions.imageMethod import *
from lvmdrp.functions.rssMethod import *


description = "Provides Methods to reduce GMOS data"
gmos_calib_n = os.path.dirname(__file__) + "/../../config/GMOS-N/"
gmos_calib_s = os.path.dirname(__file__) + "/../../config/GMOS-S/"


def createCCDfromArchive_drp(
    infile,
    prefix,
    master_bias=None,
    splits="0,0,0,0",
    single=False,
    mask_saturated=True,
    saturate_value=65535,
):
    single = bool(single)
    splits = numpy.array(splits.split(",")).astype(numpy.int16)
    hdu = pyfits.open(infile, do_not_scale_image_data=False, memmap=False)
    hdr = hdu[0].header
    bins = numpy.array(hdu[1].header["CCDSUM"].split()).astype("int")
    if "Hamamatsu" in hdr["DETECTOR"]:
        sections = 12
        CCD = numpy.zeros((4224 / bins[1], 6144 / bins[0]), dtype=numpy.float32)
        CCD_err = numpy.zeros((4224 / bins[1], 6144 / bins[0]), dtype=numpy.float32)
        CCD_mask = numpy.zeros((4224 / bins[1], 6144 / bins[0]), dtype="bool")
    else:
        sections = 6
        CCD = numpy.zeros((4608, 6144), dtype=numpy.float32)
        CCD_err = numpy.zeros((4608, 6144), dtype=numpy.float32)
        CCD_mask = numpy.zeros((4608, 6144), dtype="bool")
    if single == True:
        sections = sections / 3
    if master_bias is not None:
        hdu_bias = pyfits.open(master_bias, do_not_scale_image_data=True, memmap=False)

    for i in range(1, sections + 1):
        temp_sec = hdu[i].header["DETSEC"].replace("[", "").replace("]", "").split(",")
        CCDsec_x = [
            (int(temp_sec[0].split(":")[0]) - 1) / bins[0],
            int(temp_sec[0].split(":")[1]) / bins[0],
        ]
        CCDsec_y = [
            (int(temp_sec[1].split(":")[0]) - 1) / bins[1],
            int(temp_sec[1].split(":")[1]) / bins[1],
        ]
        if CCDsec_x[0] < 0:
            CCDsec_x[0] = 0
        if CCDsec_y[0] < 0:
            CCDsec_y[0] = 0

        temp_dat = hdu[i].header["DATASEC"].replace("[", "").replace("]", "").split(",")
        DATAsec_x = [int(temp_dat[0].split(":")[0]) - 1, int(temp_dat[0].split(":")[1])]
        DATAsec_y = [int(temp_dat[1].split(":")[0]) - 1, int(temp_dat[1].split(":")[1])]
        img = hdu[i].data[DATAsec_y[0] : DATAsec_y[1], DATAsec_x[0] : DATAsec_x[1]]

        temp_bias = (
            hdu[i].header["BIASSEC"].replace("[", "").replace("]", "").split(",")
        )
        BIASsec_x = (
            int(temp_bias[0].split(":")[0]) - 1,
            int(temp_bias[0].split(":")[1]),
        )
        BIASsec_y = (
            int(temp_bias[1].split(":")[0]) - 1,
            int(temp_bias[1].split(":")[1]),
        )
        bias = numpy.median(
            hdu[i].data[BIASsec_y[0] : BIASsec_y[1], BIASsec_x[0] : BIASsec_x[1]]
        )
        gain = float(hdu[i].header["gain"])
        rdnoise = float(hdu[i].header["rdnoise"])
        # print(infile,gain,rdnoise)
        if master_bias is not None:
            img_bias = hdu_bias[i].data
            # gain = hdu_bias[i].header['GAINORIG']
            # rdnoise = hdu_bias[i].header['RONORIG']
            bias_bias = numpy.median(
                img_bias[BIASsec_y[0] : BIASsec_y[1], BIASsec_x[0] : BIASsec_x[1]]
            )
            bias = img_bias[
                DATAsec_y[0] : DATAsec_y[1], DATAsec_x[0] : DATAsec_x[1]
            ] - (bias_bias - bias)
        if master_bias is not None:
            img[(img - bias) <= 0] = bias[(img - bias) <= 0]
        else:
            img[(img - bias) <= 0] = bias
        CCD[CCDsec_y[0] : CCDsec_y[1], CCDsec_x[0] : CCDsec_x[1]] = (img - bias) * gain
        CCD_err[CCDsec_y[0] : CCDsec_y[1], CCDsec_x[0] : CCDsec_x[1]] = numpy.sqrt(
            (img - bias) * gain + rdnoise**2
        )
        select_nan = numpy.isnan(CCD_err)
        CCD_err[select_nan] = rdnoise
        if mask_saturated:
            select = img == saturate_value
            CCD_mask[CCDsec_y[0] : CCDsec_y[1], CCDsec_x[0] : CCDsec_x[1]] = select

    if splits[0] == 0:
        if not mask_saturated:
            CCD1_out = Image(
                data=CCD[:, 0 : 2046 / bins[0] + 1],
                error=CCD_err[:, 0 : 0 : 2046 / bins[0] + 1],
                header=hdr,
            )
        else:
            CCD1_out = Image(
                data=CCD[:, 0 : 2046 / bins[0] + 1],
                error=CCD_err[:, 0 : 2046 / bins[0] + 1],
                mask=CCD_mask[:, 0 : 2046 / bins[0] + 1],
                header=hdr,
            )
    else:
        if not mask_saturated:
            CCD1_out = Image(
                data=CCD[:, splits[0] / bins[0] : 2046 / bins[0] + 1],
                error=CCD_err[:, splits[0] / bins[0] : 2046 / bins[0] + 1],
                header=hdr,
            )
        else:
            CCD1_out = Image(
                data=CCD[:, splits[0] / bins[0] : 2046 / bins[0] + 1],
                error=CCD_err[:, splits[0] / bins[0] : 2046 / bins[0] + 1],
                mask=CCD_mask[:, splits[0] / bins[0] : 2046 / bins[0] + 1],
                header=hdr,
            )
    CCD1_out.writeFitsData(prefix + ".CCD1.fits")
    if splits[1] == 0 and splits[2] == 0:
        if not mask_saturated:
            CCD2_out = Image(
                data=CCD[:, 2048 / bins[0] : 4094 / bins[0] + 1],
                error=CCD_err[:, 2048 / bins[0] : 4094 / bins[0] + 1],
                header=hdr,
            )
        else:
            CCD2_out = Image(
                data=CCD[:, 2048 / bins[0] : 4094 / bins[0] + 1],
                error=CCD_err[:, 2048 / bins[0] : 4094 / bins[0] + 1],
                mask=CCD_mask[:, 2048 / bins[0] : 4094 / bins[0] + 1],
                header=hdr,
            )
        CCD2_out.writeFitsData(prefix + ".CCD2.fits")
    else:
        if not mask_saturated:
            CCD2_out = Image(
                data=CCD[:, 2048 : 2048 + splits[1]],
                error=CCD_err[:, 2048 : 2048 + splits[1]],
                header=hdr,
            )
        else:
            CCD2_out = Image(
                data=CCD[:, 2048 : 2048 + splits[1]],
                error=CCD_err[:, 2048 : 2048 + splits[1]],
                mask=CCD_mask[:, 2048 : 2048 + splits[1]],
                header=hdr,
            )
        CCD2_out.writeFitsData(prefix + ".CCD2L.fits")
        if not mask_saturated:
            CCD2_out = Image(
                data=CCD[:, 2048 + splits[2] : 4095],
                error=CCD_err[:, 2048 + splits[2] : 4095],
                header=hdr,
            )
        else:
            CCD2_out = Image(
                data=CCD[:, 2048 + splits[2] : 4095],
                error=CCD_err[:, 2048 + splits[2] : 4095],
                mask=CCD_mask[:, 2048 + splits[2] : 4095],
                header=hdr,
            )
        CCD2_out.writeFitsData(prefix + ".CCD2R.fits")
    if splits[3] == 0:
        if not mask_saturated:
            CCD3_out = Image(
                data=CCD[:, 4096 / bins[0] : 6142 / bins[0] + 1],
                error=CCD_err[:, 4096 / bins[0] : 6142 / bins[0] + 1],
                header=hdr,
            )
        else:
            CCD3_out = Image(
                data=CCD[:, 4096 / bins[0] : 6142 / bins[0] + 1],
                error=CCD_err[:, 4096 / bins[0] : 6142 / bins[0] + 1],
                mask=CCD_mask[:, 4096 / bins[0] : 6142 / bins[0] + 1],
                header=hdr,
            )

    else:
        if not mask_saturated:
            CCD3_out = Image(
                data=CCD[:, 4096 / bins[0] : 4096 / bins[0] + splits[3]],
                error=CCD_err[:, 4096 / bins[0] : 4096 / bins[0] + splits[3]],
                header=hdr,
            )
        else:
            CCD3_out = Image(
                data=CCD[:, 4096 / bins[0] : 4096 / bins[0] + splits[3]],
                error=CCD_err[:, 4096 / bins[0] : 4096 / bins[0] + splits[3]],
                mask=CCD_mask[:, 4096 / bins[0] : 4096 / bins[0] + splits[3]],
                header=hdr,
            )
    CCD3_out.writeFitsData(prefix + ".CCD3.fits")
    return bins


def combineBias_drp(file_list, file_out):
    files = open(file_list, "r")
    lines = files.readlines()
    hdu = pyfits.open(lines[0][:-1], do_not_scale_image_data=True, memmap=False)
    hdulist = [pyfits.PrimaryHDU()]
    for j in range(1, len(hdu)):
        for i in range(len(lines)):
            hdu = pyfits.open(lines[i][:-1], do_not_scale_image_data=True, memmap=False)
            if i == 0:
                dim = hdu[j].data.shape
                frames = numpy.zeros((len(lines), dim[0], dim[1]), dtype=numpy.float32)
            frames[i, :, :] = hdu[j].data

        mean = numpy.median(frames, 0)
        hdu_out = pyfits.ImageHDU(mean)
        hdu_out.header = hdu[j].header
        hdulist.append(hdu_out)
    hdu = pyfits.HDUList(hdulist)
    hdu.writeto(file_out, overwrite=True)


def reduceCalib_drp(trace, master_bias, arc="", fiberflat="1", reduce_ccd="ALL"):
    fiberflat = int(fiberflat)
    hdr_trace = Header()
    hdr_trace.loadFitsHeader(trace)
    IFU_mask = hdr_trace.getHdrValue("MASKNAME")
    grating = hdr_trace.getHdrValue("GRATING")
    centwave = float(hdr_trace.getHdrValue("CENTWAVE"))
    instrument = hdr_trace.getHdrValue("INSTRUME")
    # print(grating, centwave,IFU_mask)
    if instrument == "GMOS-N":
        gmos_calib = gmos_calib_n
    elif instrument == "GMOS-S":
        gmos_calib = gmos_calib_s
    if IFU_mask == "IFU-2":
        if grating == "R400+_G5305" and centwave == 800.0:
            splits = "500,1000,1760,0"
            slice1 = 500
            slice2L = 400
            slice2R = 250
            slice3 = 200
            setup = "2R400_800"
            poly_disp = [-5, -5, -2, -5]
            poly_fwhm = ["-3,-3", "-3,-3", "-3,-2", "-3,-3"]
            start_wave = 7000
            end_wave = 8500
            disp_pix = 0.7
        elif grating == "R400+_G5305" and centwave == 760.0:
            splits = "70,400,1200,1730"
            slice1 = 500
            slice2L = 200
            slice2R = 250
            slice3 = 200
            setup = "2R400_760"
            poly_disp = [-5, -2, -5, -5]
            poly_fwhm = ["-3,-3", "-3,-2", "-3,-3", "-3,-3"]
            start_wave = 7000
            end_wave = 8500
            disp_pix = 0.7
        elif grating == "R150+_G5308" and centwave == 730.0:
            splits = "0,700,1200,1900"
            slice1 = 200
            slice2L = 100
            slice2R = 200
            slice3 = 200
            setup = "2R150_730"
            poly_disp = [-5, -2, -2, -5]
            poly_fwhm = ["-3,-3", "-2,-2", "-2,-2", "-3,-3"]
            start_wave = 4800
            end_wave = 9900
            disp_pix = 2.0
        elif grating == "R150+_G5308" and centwave == 760.0:
            splits = "0,800,1250,1900"
            slice1 = 200
            slice2L = 100
            slice2R = 200
            slice3 = 200
            setup = "2R150_760"
            poly_disp = [-5, -2, -2, -4]
            poly_fwhm = ["-3,-3", "-2,-2", "-2,-2", "-3,-3"]
            start_wave = 5100
            end_wave = 9900
            disp_pix = 2.0
        ccds = ["CCD1", "CCD2L", "CCD2R", "CCD3"]
        steps = [40, 20, 20, 40]

        bins = createCCDfromArchive_drp(
            trace, "FLAT", master_bias=master_bias, splits=splits
        )
        if arc != "":
            createCCDfromArchive_drp(arc, "ARC", master_bias=master_bias, splits=splits)
        findPeaksAuto_drp(
            "FLAT.CCD1.fits",
            "peaks.CCD1",
            nfibers=750,
            slice=slice1,
            median_cross=1,
            median_box=20,
            verbose=0,
        )
        findPeaksAuto_drp(
            "FLAT.CCD2L.fits",
            "peaks.CCD2L",
            nfibers=750,
            slice=slice2L,
            median_cross=1,
            median_box=20,
            verbose=0,
        )
        findPeaksMaster2_drp(
            "FLAT.CCD2R.fits",
            "%s/master_peaks.BLUE_slit_2019" % (gmos_calib),
            "peaks.CCD2R",
            threshold=10000,
            slice=slice2R,
            median_cross=1,
            median_box=20,
            verbose=0,
        )
        findPeaksMaster2_drp(
            "FLAT.CCD3.fits",
            "%s/master_peaks.BLUE_slit_2019" % (gmos_calib),
            "peaks.CCD3",
            threshold=10000,
            slice=slice3,
            median_cross=1,
            median_box=20,
            verbose=0,
        )
        tracePeaks_drp(
            "FLAT.CCD1.fits",
            "peaks.CCD1",
            "tjunk.CCD1.trc.fits",
            poly_disp="-5",
            steps=50,
            coadd=20,
            max_diff=1,
            median_cross=1,
            threshold_peak=400,
        )
        tracePeaks_drp(
            "FLAT.CCD2L.fits",
            "peaks.CCD2L",
            "tjunk.CCD2L.trc.fits",
            poly_disp="-2",
            steps=20,
            coadd=30,
            max_diff=1,
            threshold_peak=400,
            median_cross=1,
        )
        tracePeaks_drp(
            "FLAT.CCD2R.fits",
            "peaks.CCD2R",
            "tjunk.CCD2R.trc.fits",
            poly_disp="-2",
            steps=20,
            coadd=30,
            max_diff=1,
            median_cross=1,
            threshold_peak=400,
        )
        tracePeaks_drp(
            "FLAT.CCD3.fits",
            "peaks.CCD3",
            "tjunk.CCD3.trc.fits",
            poly_disp="-5",
            steps=50,
            coadd=20,
            max_diff=1,
            median_cross=1,
            threshold_peak=400,
        )
        for i in range(len(ccds)):
            subtractStraylight_drp(
                "FLAT.%s.fits" % (ccds[i]),
                "tjunk.%s.trc.fits" % (ccds[i]),
                "FLAT.%s.back.fits" % (ccds[i]),
                "FLAT.%s.stray.fits" % (ccds[i]),
                aperture=14,
                poly_cross=6,
                smooth_disp=80,
                minfit=20,
                maxfit=-10,
            )
            traceFWHM_drp(
                "FLAT.%s.stray.fits" % (ccds[i]),
                "tjunk.%s.trc.fits" % (ccds[i]),
                "tjunk.%s.fwhm.fits" % (ccds[i]),
                blocks=16,
                steps=steps[i],
                poly_disp=-5,
                init_fwhm=3,
                clip="1.0,4.0",
                threshold_flux=2000,
            )
            if arc != "":
                LACosmic_drp(
                    "ARC.%s.fits" % (ccds[i]),
                    "ARC.%s.cosmic.fits" % (ccds[i]),
                    sigma_det=5.0,
                    flim=2.0,
                    iter=4,
                    error_box="20,3",
                    replace_box="20,3",
                    rdnoise=3.5,
                    sig_gauss="1.4,1.4",
                    increase_radius=1,
                    parallel=2,
                )
                subtractStraylight_drp(
                    "ARC.%s.fits" % (ccds[i]),
                    "tjunk.%s.trc.fits" % (ccds[i]),
                    "ARC.%s.back.fits" % (ccds[i]),
                    "ARC.%s.stray.fits" % (ccds[i]),
                    aperture=14,
                    poly_cross=6,
                    smooth_disp=70,
                    minfit=20,
                    maxfit=-10,
                )
                extractSpec_drp(
                    "ARC.%s.cosmic.fits" % (ccds[i]),
                    "tjunk.%s.trc.fits" % (ccds[i]),
                    "ARC.%s.ms.fits" % (ccds[i]),
                    method="aperture",
                    aperture=5,
                )
                detWaveSolution_drp(
                    "ARC.%s.ms.fits" % (ccds[i]),
                    "ARC.%s" % (ccds[i]),
                    "%s/arc.%s.%s.txt" % (gmos_calib, ccds[i], setup),
                    poly_dispersion=poly_disp[i],
                    poly_fwhm=poly_fwhm[i],
                    flux_min=100.0,
                    fwhm_max=8.0,
                    rel_flux_limits="0.1,6.0",
                    aperture=20,
                )
            if fiberflat == 1:
                extractSpec_drp(
                    "FLAT.%s.stray.fits" % (ccds[i]),
                    "tjunk.%s.trc.fits" % (ccds[i]),
                    "FLAT.%s.ms.fits" % (ccds[i]),
                    method="optimal",
                    fwhm="tjunk.%s.fwhm.fits" % (ccds[i]),
                    parallel=1,
                )
                createPixTable_drp(
                    "FLAT.%s.ms.fits" % (ccds[i]),
                    "FLAT.%s.rss.fits" % (ccds[i]),
                    "ARC.%s.disp.fits" % (ccds[i]),
                    "ARC.%s.res.fits" % (ccds[i]),
                )
        if fiberflat == 1:
            glueRSS_drp("FLAT.CCD1.rss.fits,FLAT.CCD2L.rss.fits", "FLAT_red.rss.fits")
            glueRSS_drp("FLAT.CCD2R.rss.fits,FLAT.CCD3.rss.fits", "FLAT_blue.rss.fits")
            resampleWave_drp(
                "FLAT_blue.rss.fits",
                "FLAT_blue.disp_cor.fits",
                start_wave=start_wave,
                end_wave=end_wave,
                disp_pix=disp_pix,
                err_sim=0,
                method="linear",
                parallel=1,
            )
            resampleWave_drp(
                "FLAT_red.rss.fits",
                "FLAT_red.disp_cor.fits",
                start_wave=start_wave,
                end_wave=end_wave,
                disp_pix=disp_pix,
                err_sim=0,
                method="linear",
            )
            mergeRSS_drp(
                "FLAT_red.disp_cor.fits,FLAT_blue.disp_cor.fits", "FLAT.disp_cor.fits"
            )
            createFiberFlat_drp("FLAT.disp_cor.fits", "FIBERFLAT.fits", clip="0.25,2.0")

    elif IFU_mask == "IFU-R" or IFU_mask == "IFU-B":
        ccds = numpy.array(["CCD1", "CCD2", "CCD3"])
        splits = "0,0,0,0"

        if grating == "R400+_G5305" and centwave == 700.0 and IFU_mask == "IFU-R":
            setup = "1RR400_700"
            slice_CCD = [1000, 1000, 100]
            poly_disp = [-6, -6, -6]
            poly_fwhm = ["-2,-3", "-2,-3", "-2,-3"]
            start_wave = 5800
            end_wave = 9000
            disp_pix = 0.7

        elif grating == "R400+_G5325" and centwave == 690.0 and IFU_mask == "IFU-R":
            setup = "1RR400_690"
            slice_CCD = [700, 350, 200]
            poly_disp = [-4, -4, -4]
            poly_fwhm = ["-2,-3", "-2,-3", "-2,-3"]
            smooth_median_flat = 100
            if reduce_ccd == "ALL":
                start_wave = 6300
                end_wave = 9000
            elif reduce_ccd == "CCD2":
                start_wave = 6150
                end_wave = 7600
            disp_pix = 0.7

        elif grating == "R400+_G5305" and centwave == 710.0 and IFU_mask == "IFU-R":
            setup = "1RR400_710"
            slice_CCD = [700, 350, 200]
            poly_disp = [-4, -4, -4]
            poly_fwhm = ["-2,-3", "-2,-3", "-2,-3"]
            start_wave = 5800
            end_wave = 9000
            disp_pix = 0.7
        elif grating == "B600+_G5307" and centwave == 625.0 and IFU_mask == "IFU-R":
            setup = "1RB600_625"
            slice_CCD = [2000, 2000, 2000]

            poly_disp = [-3, -3, -3]
            poly_fwhm = ["-2,-3", "-2,-3", "-2,-3"]
            start_wave = 4670
            end_wave = 7850
            disp_pix = 0.7

        bins = createCCDfromArchive_drp(
            trace, "FLAT", master_bias=master_bias, splits=splits
        )
        if arc != "":
            createCCDfromArchive_drp(arc, "ARC", master_bias=master_bias, splits=splits)
        if reduce_ccd == "ALL":
            indices = [0, 1, 2]
        elif reduce_ccd == "CCD1":
            indices = [0]
        elif reduce_ccd == "CCD2":
            indices = [1]
        elif reduce_ccd == "CCD3":
            indices = [2]
        for i in indices:
            addCCDMask_drp(
                "FLAT.%s.fits" % (ccds[i]),
                "%s/MASK.%s.Hamamatsu.fits" % (gmos_calib, ccds[i]),
            )
            if instrument == "GMOS-N":
                findPeaksAuto_drp(
                    "FLAT.%s.fits" % (ccds[i]),
                    "peaks.%s" % (ccds[i]),
                    nfibers=750,
                    slice=slice_CCD[i] / bins[0],
                    median_cross=1,
                    median_box=20,
                    verbose=0,
                )
            else:
                # findPeaksOffset_drp('FLAT.%s.fits'%(ccds[i]), '%s/master_peaks.RED_slit'%(gmos_calib), 'peaks.%s'%(ccds[i]), threshold=7000, slice=slice_CCD[i],median_cross=1, median_box=20)
                findPeaksMaster_drp(
                    "FLAT.%s.fits" % (ccds[i]),
                    "%s/master_peaks.RED_slit" % (gmos_calib),
                    "peaks.%s" % (ccds[i]),
                    slice=slice_CCD[i] / bins[0],
                    median_cross=1,
                    median_box=10,
                )
            # tracePeaks_drp('FLAT.%s.fits'%(ccds[i]), 'peaks.%s'%(ccds[i]), 'tjunk.%s.trc.fits'%(ccds[i]), poly_disp='-5', steps=20/bins[1], max_diff=1,threshold_peak=50, median_box=50, verbose=1)
            # subtractStraylight_drp('FLAT.%s.fits' % (ccds[i]), 'tjunk.%s.trc.fits' % (ccds[i]), 'FLAT.%s.back.fits' % (ccds[i]), 'FLAT.%s.stray.fits' % (ccds[i]), aperture=10, poly_cross=6, smooth_disp=70/bins[0],smooth_gauss=10)
            # traceFWHM_drp('FLAT.%s.stray.fits'%(ccds[i]), 'tjunk.%s.trc.fits'%(ccds[i]), 'tjunk.%s.fwhm.fits'%(ccds[i]), blocks=16, steps=50/bins[0], poly_disp=-5, init_fwhm=3, clip='1.0,8.0', threshold_flux=500)
            if arc != "":
                #   LACosmic_drp('ARC.%s.fits'%(ccds[i]), 'ARC.%s.cosmic.fits'%(ccds[i]), sigma_det=5.0, flim=2.0, iter=4, error_box='20,3', replace_box='20,3', rdnoise=3.5,sig_gauss='1.4,1.4', increase_radius=1, parallel=2)
                #  addCCDMask_drp('ARC.%s.cosmic.fits'%(ccds[i]),'%s/MASK.%s.Hamamatsu.fits'%(gmos_calib,ccds[i]))
                extractSpec_drp(
                    "ARC.%s.cosmic.fits" % (ccds[i]),
                    "tjunk.%s.trc.fits" % (ccds[i]),
                    "ARC.%s.ms.fits" % (ccds[i]),
                    method="aperture",
                    aperture=5,
                )
                detWaveSolution_drp(
                    "ARC.%s.ms.fits" % (ccds[i]),
                    "ARC.%s" % (ccds[i]),
                    "%s/arc.%s.%s.txt" % (gmos_calib, ccds[i], setup),
                    poly_dispersion=poly_disp[i],
                    poly_fwhm=poly_fwhm[i],
                    flux_min=40.0,
                    fwhm_max=8.0,
                    aperture=20 / bins[0],
                    rel_flux_limits="0.1,6.0",
                )
            if fiberflat == 1:
                extractSpec_drp(
                    "FLAT.%s.stray.fits" % (ccds[i]),
                    "tjunk.%s.trc.fits" % (ccds[i]),
                    "FLAT.%s.ms.fits" % (ccds[i]),
                    method="optimal",
                    fwhm="tjunk.%s.fwhm.fits" % (ccds[i]),
                    parallel=1,
                )
                createPixTable_drp(
                    "FLAT.%s.ms.fits" % (ccds[i]),
                    "FLAT.%s.rss.fits" % (ccds[i]),
                    "ARC.%s.disp.fits" % (ccds[i]),
                    "ARC.%s.res.fits" % (ccds[i]),
                )

        if fiberflat == 1:
            if reduce_ccd == "ALL":
                glueRSS_drp("FLAT.CCD1.rss.fits,FLAT.CCD2.rss.fits", "FLAT.rss.fits")

                resampleWave_drp(
                    "FLAT.rss.fits",
                    "FLAT.disp_cor.fits",
                    start_wave=start_wave,
                    end_wave=end_wave,
                    disp_pix=disp_pix,
                    compute_densities=1,
                    err_sim=0,
                    method="linear",
                    parallel=1,
                )
            else:
                resampleWave_drp(
                    "FLAT.%s.rss.fits" % (reduce_ccd),
                    "FLAT.disp_cor.fits",
                    start_wave=start_wave,
                    end_wave=end_wave,
                    disp_pix=disp_pix,
                    compute_densities=1,
                    err_sim=0,
                    method="linear",
                    parallel=1,
                )
            createFiberFlat_drp(
                "FLAT.disp_cor.fits",
                "FIBERFLAT.fits",
                smooth_median=smooth_median_flat,
                clip="0.25,2.0",
            )


def reduceSTD_drp(
    std,
    master_bias,
    fiberflat="1",
    ref_star="",
    straylight="0",
    mask_telluric="",
    reduce_ccd="ALL",
):
    hdr_std = Header()
    hdr_std.loadFitsHeader(std)
    IFU_mask = hdr_std.getHdrValue("MASKNAME")
    grating = hdr_std.getHdrValue("GRATING")
    centwave = float(hdr_std.getHdrValue("CENTWAVE"))
    instrument = hdr_std.getHdrValue("INSTRUME")
    if instrument == "GMOS-N":
        gmos_calib = gmos_calib_n
    elif instrument == "GMOS-S":
        gmos_calib = gmos_calib_s
    if IFU_mask == "IFU-2":
        if grating == "R400+_G5305" and centwave == 800.0:
            splits = "500,1000,1760,0"
            start_wave = 7000
            end_wave = 8500
            disp_pix = 0.7
            mask_wave = "7150,7330,7530,7700,8120,8270,8350,8450"
            smooth_poly = -15
        elif grating == "R400+_G5305" and centwave == 760.0:
            splits = "70,400,1200,1730"
            start_wave = 7000
            end_wave = 8500
            disp_pix = 0.7
            mask_wave = "7130,7260,7550,7700,7930,8050,8120,8270"
            smooth_poly = -40
        elif grating == "R150+_G5308" and centwave == 730.0:
            splits = "0,700,1200,1900"
            setup = "2R150_730"
            start_wave = 4800
            end_wave = 9900
            disp_pix = 2.0
        ccds = ["CCD1", "CCD2L", "CCD2R", "CCD3"]
        steps = [40, 20, 20, 40]

        createCCDfromArchive_drp(std, "STD", master_bias=master_bias, splits=splits)
        for i in range(len(ccds)):
            LACosmic_drp(
                "STD.%s.fits" % (ccds[i]),
                "STD.%s.cosmic.fits" % (ccds[i]),
                sigma_det=5.0,
                flim=2.0,
                iter=4,
                error_box="20,3",
                replace_box="20,3",
                rdnoise=3.5,
                sig_gauss="1.4,1.4",
                increase_radius=1,
                parallel=2,
            )
            addCCDMask_drp(
                "STD.%s.cosmic.fits" % (ccds[i]),
                "%s/MASK.%s.Hamamatsu.fits" % (gmos_calib, ccds[i]),
            )
            if int(straylight) == 1:
                subtractStraylight_drp(
                    "STD.%s.cosmic.fits" % (ccds[i]),
                    "tjunk.%s.trc.fits" % (ccds[i]),
                    "STD.%s.back.fits" % (ccds[i]),
                    "STD.%s.stray.fits" % (ccds[i]),
                    aperture=14,
                    poly_cross=6,
                    smooth_disp=70,
                    minfit=20,
                    maxfit=10,
                )
                extractSpec_drp(
                    "STD.%s.cosmic.fits" % (ccds[i]),
                    "tjunk.%s.trc.fits" % (ccds[i]),
                    "STD.%s.ms.fits" % (ccds[i]),
                    method="optimal",
                    fwhm="tjunk.%s.fwhm.fits" % (ccds[i]),
                    parallel=1,
                )
            else:
                extractSpec_drp(
                    "STD.%s.stray.fits" % (ccds[i]),
                    "tjunk.%s.trc.fits" % (ccds[i]),
                    "STD.%s.ms.fits" % (ccds[i]),
                    method="optimal",
                    fwhm="tjunk.%s.fwhm.fits" % (ccds[i]),
                    parallel=1,
                )
            createPixTable_drp(
                "STD.%s.ms.fits" % (ccds[i]),
                "STD.%s.rss.fits" % (ccds[i]),
                "ARC.%s.disp.fits" % (ccds[i]),
                "ARC.%s.res.fits" % (ccds[i]),
            )
        glueRSS_drp("STD.CCD1.rss.fits,STD.CCD2L.rss.fits", "STD_red.rss.fits")
        glueRSS_drp("STD.CCD2R.rss.fits,STD.CCD3.rss.fits", "STD_blue.rss.fits")
        resampleWave_drp(
            "STD_blue.rss.fits",
            "STD_blue.disp_cor.fits",
            start_wave=start_wave,
            end_wave=end_wave,
            disp_pix=disp_pix,
            err_sim=0,
            method="linear",
            parallel=1,
        )
        resampleWave_drp(
            "STD_red.rss.fits",
            "STD_red.disp_cor.fits",
            start_wave=start_wave,
            end_wave=end_wave,
            disp_pix=disp_pix,
            err_sim=0,
            method="linear",
        )
        mergeRSS_drp(
            "STD_red.disp_cor.fits,STD_blue.disp_cor.fits", "STD.disp_cor.fits"
        )
        correctFiberFlat_drp("STD.disp_cor.fits", "STD.flat.fits", "FIBERFLAT.fits")
        includePosTab_drp("STD.flat.fits", "%s/GMOS_2slit_pt.txt" % (gmos_calib))
        splitFibers_drp(
            "STD.flat.fits",
            "STD.obj_red.fits,STD.sky_red.fits,STD.obj_blue.fits,STD.sky_blue.fits",
            "O_R,S_R,O_B,S_B",
        )
        constructSkySpec_drp("STD.sky_red.fits", "STD.skyspec_red.fits", nsky=200)
        constructSkySpec_drp("STD.sky_blue.fits", "STD.skyspec_blue.fits", nsky=200)
        subtractSkySpec_drp(
            "STD.obj_red.fits", "STD.sobj_red.fits", "STD.skyspec_red.fits"
        )
        subtractSkySpec_drp(
            "STD.obj_blue.fits", "STD.sobj_blue.fits", "STD.skyspec_blue.fits"
        )
        mergeRSS_drp("STD.sobj_red.fits,STD.sobj_blue.fits", "STD.sobj.fits")
        createCube_drp(
            "STD.sobj.fits", "STD.cube.fits", mode="drizzle", resolution=0.2, parallel=1
        )
    elif IFU_mask == "IFU-R" or IFU_mask == "IFU-B":
        ccds = ["CCD1", "CCD2", "CCD3"]
        splits = "0,0,0,0"
        if grating == "R400+_G5305" and centwave == 700.0 and IFU_mask == "IFU-R":
            start_wave = 5800
            end_wave = 9000
            disp_pix = 0.7
            mask_wave = (
                "6200,6350,6500,6600,6860,7050,7150,7330,7570,7760,8120,8350,8650,8730"
            )
            smooth_poly = -15
        elif grating == "R400+_G5325" and centwave == 690.0 and IFU_mask == "IFU-R":
            setup = "1RR400_690"
            if reduce_ccd == "ALL":
                start_wave = 4600
                end_wave = 9000
            elif reduce_ccd == "CCD2":
                start_wave = 6140
                end_wave = 7600
            disp_pix = 0.7
            mask_wave = (
                "6200,6350,6500,6600,6860,7050,7150,7330,7570,7760,8120,8350,8650,8730"
            )
            smooth_poly = -15

        createCCDfromArchive_drp(std, "STD", master_bias=master_bias, splits=splits)
        if reduce_ccd == "ALL":
            indices = [0, 1, 2]
        elif reduce_ccd == "CCD1":
            indices = [0]
        elif reduce_ccd == "CCD2":
            indices = [1]
        elif reduce_ccd == "CCD3":
            indices = [2]
        for i in indices:
            # print(i)
            LACosmic_drp(
                "STD.%s.fits" % (ccds[i]),
                "STD.%s.cosmic.fits" % (ccds[i]),
                sigma_det=5.0,
                flim=1.3,
                iter=4,
                error_box="20,3",
                replace_box="20,3",
                rdnoise=3.5,
                sig_gauss="3,3",
                increase_radius=1,
                parallel=2,
            )
            addCCDMask_drp(
                "STD.%s.cosmic.fits" % (ccds[i]),
                "%s/MASK.%s.Hamamatsu.fits" % (gmos_calib, ccds[i]),
            )
            if int(straylight) == 1:
                subtractStraylight_drp(
                    "STD.%s.cosmic.fits" % (ccds[i]),
                    "tjunk.%s.trc.fits" % (ccds[i]),
                    "STD.%s.back.fits" % (ccds[i]),
                    "STD.%s.stray.fits" % (ccds[i]),
                    aperture=10,
                    poly_cross=6,
                    smooth_disp=70,
                    smooth_gauss=10,
                )
                extractSpec_drp(
                    "STD.%s.stray.fits" % (ccds[i]),
                    "tjunk.%s.trc.fits" % (ccds[i]),
                    "STD.%s.ms.fits" % (ccds[i]),
                    method="optimal",
                    fwhm="tjunk.%s.fwhm.fits" % (ccds[i]),
                    parallel=1,
                )
            else:
                extractSpec_drp(
                    "STD.%s.cosmic.fits" % (ccds[i]),
                    "tjunk.%s.trc.fits" % (ccds[i]),
                    "STD.%s.ms.fits" % (ccds[i]),
                    method="optimal",
                    fwhm="tjunk.%s.fwhm.fits" % (ccds[i]),
                    parallel=1,
                )
            createPixTable_drp(
                "STD.%s.ms.fits" % (ccds[i]),
                "STD.%s.rss.fits" % (ccds[i]),
                "ARC.%s.disp.fits" % (ccds[i]),
                "ARC.%s.res.fits" % (ccds[i]),
            )
        if reduce_ccd == "ALL":
            glueRSS_drp(
                "STD.CCD1.rss.fits,STD.CCD2.rss.fits,STD.CCD3.rss.fits", "STD.rss.fits"
            )
            resampleWave_drp(
                "STD.rss.fits",
                "STD.disp_cor.fits",
                start_wave=start_wave,
                end_wave=end_wave,
                disp_pix=disp_pix,
                err_sim=0,
                method="linear",
            )
        else:
            resampleWave_drp(
                "STD.%s.rss.fits" % (ccds[i]),
                "STD.disp_cor.fits",
                start_wave=start_wave,
                end_wave=end_wave,
                disp_pix=disp_pix,
                err_sim=0,
                method="linear",
            )
        correctFiberFlat_drp("STD.disp_cor.fits", "STD.flat.fits", "FIBERFLAT.fits")
        if IFU_mask == "IFU-R":
            includePosTab_drp("STD.flat.fits", "%s/GMOS_1slitR_pt.txt" % (gmos_calib))
            splitFibers_drp("STD.flat.fits", "STD.obj.fits,STD.sky.fits", "O_R,S_R")
        elif IFU_mask == "IFU-B":
            includePosTab_drp("STD.flat.fits", "%s/GMOS_1slitB_pt.txt" % (gmos_calib))
            splitFibers_drp("STD.flat.fits", "STD.obj.fits,STD.sky.fits", "O_B,S_B")
    constructSkySpec_drp("STD.sky.fits", "STD.skyspec_red.fits", nsky=200)
    subtractSkySpec_drp("STD.obj.fits", "STD.sobj.fits", "STD.skyspec_red.fits")
    createCube_drp(
        "STD.obj.fits", "STD.cube.fits", mode="drizzle", resolution=0.2, parallel=1
    )
    if ref_star != "":
        createSensFunction_drp(
            "STD.sobj.fits",
            "ratio.txt",
            "%s/%s" % (gmos_calib, ref_star),
            airmass="AIRMASS",
            exptime="EXPTIME",
            coadd=200,
            extinct_curve="Paranal",
            out_star="star.txt",
            mask_wave=mask_wave,
            mask_telluric=mask_telluric,
            smooth_poly=smooth_poly,
            verbose=1,
        )


def reduceObject_drp(
    obj,
    master_bias,
    res_fwhm="0.0",
    fiberflat="1",
    straylight="1",
    flux_calib="1",
    telluric_cor="1",
    flexure_correct="1",
    reduce_ccd="ALL",
):
    straylight = int(straylight)
    fiberflat = int(fiberflat)
    flux_calib = int(fiberflat)
    telluric_cor = int(telluric_cor)
    flexure_correct = int(flexure_correct)
    res_fwhm = float(res_fwhm)

    hdr_obj = Header()
    hdr_obj.loadFitsHeader(obj)
    IFU_mask = hdr_obj.getHdrValue("MASKNAME")
    grating = hdr_obj.getHdrValue("GRATING")
    centwave = float(hdr_obj.getHdrValue("CENTWAVE"))
    instrument = hdr_obj.getHdrValue("INSTRUME")
    # print(grating, centwave,IFU_mask)
    if instrument == "GMOS-N":
        gmos_calib = gmos_calib_n
    elif instrument == "GMOS-S":
        gmos_calib = gmos_calib_s
    if IFU_mask == "IFU-2":
        if grating == "R400+_G5305" and centwave == 800.0:
            splits = "500,1000,1760,0"
            start_wave = 7000
            end_wave = 8500
            disp_pix = 0.7
            sky_line_list = ""
        elif grating == "R400+_G5305" and centwave == 760.0:
            splits = "70,400,1200,1730"
            start_wave = 7000
            end_wave = 8500
            disp_pix = 0.7
            sky_line_list = ""
        elif grating == "R150+_G5308" and centwave == 730.0:
            splits = "0,700,1200,1900"
            start_wave = 4800
            end_wave = 9900
            disp_pix = 2.0
            sky_line_list = ""
        elif grating == "R150+_G5308" and centwave == 760.0:
            splits = "0,800,1250,1900"
            start_wave = 5100
            end_wave = 9900
            disp_pix = 2.0
            sky_line_list = ""
        ccds = ["CCD1", "CCD2L", "CCD2R", "CCD3"]
    elif IFU_mask == "IFU-R" or IFU_mask == "IFU-B":
        ccds = ["CCD1", "CCD2", "CCD3"]
        splits = "0,0,0,0"
        if grating == "R400+_G5305" and centwave == 700.0 and IFU_mask == "IFU-R":
            start_wave = 5800
            end_wave = 9000
            disp_pix = 0.7
            sky_line_list = ""
        elif grating == "R400+_G5325" and centwave == 690.0 and IFU_mask == "IFU-R":
            disp_pix = 0.7
            splits = "0,0,0,900"
            sky_line_list = ""
            if reduce_ccd == "ALL":
                start_wave = 6300
                end_wave = 9000
            elif reduce_ccd == "CCD2":
                start_wave = 6150
                end_wave = 7600
        elif grating == "R400+_G5305" and centwave == 710.0 and IFU_mask == "IFU-R":
            start_wave = 5800
            end_wave = 9000
            disp_pix = 0.7
            sky_line_list = ""

    createCCDfromArchive_drp(obj, "OBJ", master_bias=master_bias, splits=splits)
    if reduce_ccd == "ALL":
        indices = numpy.arange(len(ccds))
    elif reduce_ccd == "CCD1":
        indices = [0]
    elif reduce_ccd == "CCD2":
        indices = [1]
    elif reduce_ccd == "CCD3":
        indices = [2]
    for i in indices:
        LACosmic_drp(
            "OBJ.%s.fits" % (ccds[i]),
            "OBJ.%s.cosmic.fits" % (ccds[i]),
            sigma_det=5.0,
            flim=2.0,
            iter=4,
            error_box="20,3",
            replace_box="20,3",
            rdnoise=3.5,
            sig_gauss="1.4,1.4",
            increase_radius=1,
            parallel=2,
        )
        if instrument == "GMOS-S":
            addCCDMask_drp(
                "OBJ.%s.cosmic.fits" % (ccds[i]),
                "%s/MASK.%s.Hamamatsu.fits" % (gmos_calib, ccds[i]),
            )
        if sky_line_list != "":
            offsetTrace_drp(
                "OBJ.%s.cosmic.fits" % (ccds[i]),
                "tjunk.%s.trc.fits" % (ccds[i]),
                "ARC.%s.disp.fits" % (ccds[i]),
                sky_line_list,
                "offsetTrace_%s.log" % (ccds[i]),
                blocks="10",
                size="30",
            )
            correctTraceMask_drp(
                "tjunk.%s.trc.fits" % (ccds[i]),
                "tjunk.%s.trc_temp.fits" % (ccds[i]),
                "offsetTrace_%s.log" % (ccds[i]),
                "OBJ.%s.cosmic.fits" % (ccds[i]),
                poly_smooth=flexure_order,
            )
        else:
            os.system(
                "cp tjunk.%s.trc.fits tjunk.%s.trc_temp.fits" % (ccds[i], ccds[i])
            )
        if straylight == 1:
            subtractStraylight_drp(
                "OBJ.%s.cosmic.fits" % (ccds[i]),
                "tjunk.%s.trc_temp.fits" % (ccds[i]),
                "OBJ.%s.back.fits" % (ccds[i]),
                "OBJ.%s.stray.fits" % (ccds[i]),
                aperture=14,
                poly_cross=6,
                smooth_disp=70,
                smooth_gauss=15,
            )
            extractSpec_drp(
                "OBJ.%s.stray.fits" % (ccds[i]),
                "tjunk.%s.trc_temp.fits" % (ccds[i]),
                "OBJ.%s.ms.fits" % (ccds[i]),
                method="optimal",
                fwhm="tjunk.%s.fwhm.fits" % (ccds[i]),
                parallel=3,
            )
        else:
            extractSpec_drp(
                "OBJ.%s.cosmic.fits" % (ccds[i]),
                "tjunk.%s.trc_temp.fits" % (ccds[i]),
                "OBJ.%s.ms.fits" % (ccds[i]),
                method="optimal",
                fwhm="tjunk.%s.fwhm.fits" % (ccds[i]),
                parallel=2,
            )
        createPixTable_drp(
            "OBJ.%s.ms.fits" % (ccds[i]),
            "OBJ.%s.rss.fits" % (ccds[i]),
            "ARC.%s.disp.fits" % (ccds[i]),
            "ARC.%s.res.fits" % (ccds[i]),
        )
        if sky_line_list != "":
            checkPixTable_drp(
                "OBJ.%s.rss.fits" % (ccds[i]),
                sky_line_list,
                "offsetWave_%s.log" % (ccds[i]),
                aperture="12",
            )
        if float(res_fwhm) != 0.0:
            matchResolution_drp(
                "OBJ.%s.rss.fits" % (ccds[i]),
                "OBJ.%s.rss.fits" % (ccds[i]),
                res_fwhm,
                parallel=4,
            )

    if IFU_mask == "IFU-2":
        glueRSS_drp("OBJ.CCD1.rss.fits,OBJ.CCD2L.rss.fits", "OBJ_red.rss.fits")
        glueRSS_drp("OBJ.CCD2R.rss.fits,OBJ.CCD3.rss.fits", "OBJ_blue.rss.fits")
        resampleWave_drp(
            "OBJ_blue.rss.fits",
            "OBJ_blue.disp_cor.fits",
            start_wave=start_wave,
            end_wave=end_wave,
            disp_pix=disp_pix,
            err_sim=200,
            method="linear",
            parallel=1,
        )
        resampleWave_drp(
            "OBJ_red.rss.fits",
            "OBJ_red.disp_cor.fits",
            start_wave=start_wave,
            end_wave=end_wave,
            disp_pix=disp_pix,
            err_sim=200,
            method="linear",
        )
        mergeRSS_drp(
            "OBJ_red.disp_cor.fits,OBJ_blue.disp_cor.fits", "OBJ.disp_cor.fits"
        )
        includePosTab_drp("OBJ.disp_cor.fits", "%s/GMOS_2slit_pt.txt" % (gmos_calib))
    elif IFU_mask == "IFU-R":
        if reduce_ccd == "ALL":
            glueRSS_drp(
                "OBJ.CCD1.rss.fits,OBJ.CCD2.rss.fits,OBJ.CCD3.rss.fits", "OBJ.rss.fits"
            )
            resampleWave_drp(
                "OBJ.rss.fits",
                "OBJ.disp_cor.fits",
                start_wave=start_wave,
                end_wave=end_wave,
                disp_pix=disp_pix,
                err_sim=200,
                method="linear",
            )
        else:
            resampleWave_drp(
                "OBJ.%s.rss.fits" % (reduce_ccd),
                "OBJ.disp_cor.fits",
                start_wave=start_wave,
                end_wave=end_wave,
                disp_pix=disp_pix,
                err_sim=200,
                method="linear",
            )
    includePosTab_drp("OBJ.disp_cor.fits", "%s/GMOS_1slitR_pt.txt" % (gmos_calib))

    if fiberflat == 1:
        correctFiberFlat_drp("OBJ.disp_cor.fits", "OBJ.flat.fits", "FIBERFLAT.fits")

    if flux_calib == 1:
        if fiberflat == 1:
            fluxCalibration_drp(
                "OBJ.flat.fits",
                "OBJ.fobj.fits",
                "ratio.txt",
                "AIRMASS",
                "EXPTIME",
                extinct_curve="Paranal",
                ref_units="1e-16",
                target_units="1e-16",
                norm_sb_fib="",
            )
        else:
            fluxCalibration_drp(
                "OBJ.disp_cor.fits",
                "OBJ.fobj.fits",
                "ratio.txt",
                "AIRMASS",
                "EXPTIME",
                extinct_curve="Paranal",
                ref_units="1e-16",
                target_units="1e-16",
                norm_sb_fib="",
            )
        if telluric_cor == 1:
            correctTelluric_drp(
                "OBJ.fobj.fits",
                "OBJ.fobj.fits",
                "telluric_template.fits",
                airmass="AIRMASS",
            )
