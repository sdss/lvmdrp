#from skyMethod import *

# from python.lvmdrp.functions import skyMethod
# from skyMethod import (
#     configureSkyModel_drp,
#     createMasterSky_drp,
#     sepContinuumLine_drp,
#     evalESOSky_drp,
#     subtractGeocoronal_drp,
#     corrSkyLine_drp,
#     corrSkyContinuum_drp,
#     coaddContinuumLine_drp,
#     subtractSky_drp,
#     refineContinuum_drp,
#     subtractPCAResiduals_drp
# )

description = "the main function that runs the sky subtraction module"

def run_sky(lvm_object_sky1, lvm_object_sky2, lvm_object_science, lvm_object_star):
    pass

    # input: RSS with camera combined, extracted, wave calibrated spectra for each fiber, grouped by type
    # lvm-object file for each sky1, sky2, science, and star1-12


    # combine fibers from each group into super spectra (note star1-12 are single fiber, so no super_star)
    # input: RSSs grouped by target type
    # output: super_sky1, super_sky2, super_science
    # example: skyMethod.createMasterSky_drp(in_rss=out_path.format("sci"), out_sky=out_path.format("msci"))
    skyMethod.createMasterSky_drp(in_rss=lvm_object_sky1, out_sky='super_sky1')
    skyMethod.createMasterSky_drp(in_rss=lvm_object_sky2, out_sky='super_sky2')
    skyMethod.createMasterSky_drp(in_rss=lvm_object_science, out_sky='super_science')

    # separate lines and continuum for each super_*
    # input: super_*
    # output: continuum of each super_*, emission lines of each super_*
    # example: skyMethod.sepContinuumLine_drp(sky_ref=out_path.format("msci"), out_cont_line=out_path.format("sci_cl"),
    #                                         method="skycorr", sky_sci=out_path.format("msci"))
    skyMethod.sepContinuumLine_drp(sky_ref='super_sky1', out_cont_line='super_sky1_cl',
                                             method="skycorr", sky_sci='super_sky1') # dont understand the inputs for skycorr here, def want to change!!!
    skyMethod.sepContinuumLine_drp(sky_ref='super_sky2', out_cont_line='super_sky2_cl',
                                             method="skycorr", sky_sci='super_sky2')
    skyMethod.sepContinuumLine_drp(sky_ref='super_science', out_cont_line='super_science_cl',
                                             method="skycorr", sky_sci='super_scinece')
    # do I need to do this for the standard stars too? Going to ignore for now...
    # output file *_cl is a fits with 2 ext (cont and line)

    # extrapolate continuum at sky to science and star and remove from each fiber
    # input: sky parameters of super_sky1, sky parameters of super_sky2, sky parameters of super_science, sky paramters of super_stan*,
    #  continuum of super_sky1, continuum of super_sky2 
    # output: science and stan* continuum subtracted, with subtracted continuum saved as fits extension, ESO model sky saved as fits extenstion, 
    #  add Halpha tranmission/extinction to header for later geocoronal
    # example: skyMethod.evalESOSky_drp(sky_ref=out_path.format("msci"), out_rss=out_path.format("model_sci"),
    #  resample_method="linear", err_sim=50, parallel=0)
    # think evalESOSky just does the model, but no extrapolation ??
    # the extrapol is in skyCorrCont

    # run skycorr with super_sky_line and super_science/star_line to remove lines from each science/star fibers
    # input: emission lines of each super_*
    # output: science and stan* with sky lines removed

    # remove geocoronal line from each science/star fiber
    # input: geocoronal paramaters, halpha extinction, shadow height for each super_*, super_* (cont and lines removed)
    # output: science and stan* with geocoronal removed, geo halpha assumed in header
     

    # output: lvm-sobject with all fibers sky subtracted with extensions of sky model, continuum subtracted,
    #  lines subtracted, super_*
    # header: geocoronal halpha removed, extinction parameter, sky parameters

def run_pca():
    pass

    #rountine written by Liz to run on flux calibrated data
    #output should have extra extenstion for pre_pca

def update_pca_library():
    pass

    #see Liz's notes and workflow on PCA

def update_geocoronal_parameters():
    pass

    #updates the dependence of geocoronal halpha on shadow height
    #input: a set of data from source with known MW halpha of shadow heights, halpha emissions,
    #  halpha atm extinction, date/time, solar flux
    #output: updated function/parameters 
