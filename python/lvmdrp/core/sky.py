

def select_sky_fibers():
    """select sky fibers to be used to build master skies 1 and 2"""
    pass


def build_master_sky():
    """run continuum/line separation algorithm on master sky 1 and 2 and master science and produce line-only (sky1_line, sky2_line, sci_line) and continuum-only (sky1_cont, sky2_cont, sci_cont) spectra for each"""
    pass


def cont_line_separation():
    """average sky1_line and sky2_line into "sky_line", and run skycorr on "sky_line" and "sci_line" to produce "sky_line_corr"
    """
    pass


def eval_eso_sky():
    """run ESO sky model for observation parameters (ephemeris, atmospheric conditions, site, etc) to evaluate sky spectrum at each telescope pointing (model_sky1, model_sky2, model_skysci)"""
    pass


def sky_cont_correct():
    """correct and combine continuum only spectra by doing:   sky_cont_corr=0.5*( sky1_cont*(model_skysci/model_sky1) + sky2_cont*(model_skysci/model_sky2)) """
    pass


def coadd_sky():
    """coadd corrected line and continuum combined sky frames:    sky_corr=sky_cont_corr+sky_line_corr"""
    pass


def sky_lsf_matching():
    """do LSF matching of sky_corr to each science fiber and subtract the LSF-matched corrected sky spectra"""
    pass


def refine_cont_subtraction():
    """optionally apply an extra residual continuum subtraction using the faintest (i.e. with no stellar light detection) spaxels in the science IFU"""
    pass
