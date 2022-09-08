

def select_sky_fibers(rss, fiber_map_path):
    """select sky fibers to be used to build master skies 1 and 2"""
    fiber_map = load_fiber_map(fiber_map_path)
    sky1_fibers, sky2_fibers, sci_fibers, std_fibers = rss[fiber_map["SKY_1"]], rss[fiber_map["SKY_2"]], rss[fiber_map["SCI"]], rss[fiber_map["STD"]]
    return sky1_fibers, sky2_fibers, sci_fibers, std_fibers


def build_master_sky(sky1_fibers, sky2_fibers, method="naive"):
    """run continuum/line separation algorithm on master sky 1 and 2 and master science and produce line-only (sky1_line, sky2_line, sci_line) and continuum-only (sky1_cont, sky2_cont, sci_cont) spectra for each"""
    if method == "naive":
        # deal with bad pixels (pixmask, pixel outliers)
        # deal with outlying fibers (remove fibers with stellar contribution)
        # match wavelength sampling between fibers
        # average the cleaned fibers
        pass
    elif method == "smart":
        # ESO way
        pass
    else:
        raise ValueError
    

def cont_line_separation(wl_master_sky, master_sky1, master_sky2, sky_lines_list, window_width):
    """average sky1_line and sky2_line into "sky_line", and run skycorr on "sky_line" and "sci_line" to produce "sky_line_corr"
    """
    wl_sky_lines, sky_lines = load_sky_lines(sky_lines_list)

    # the window size may be a function of the wavelength (there may be sky line groupings)
    # hw = window_width / 2
    # lines_mask = np.ones_like(wl_sky, dtype=bool)
    # for wl_sky in wl_sky_lines:
    #     lines_mask &= (wl_sky-hw<=wl_master_sky)&(wl_master_sky<=wl_sky+hw)
    return master_sky2_cont, master_sky2_cont, master_sky1_line, master_sky2_line


def eval_eso_sky():
    """run ESO sky model for observation parameters (ephemeris, atmospheric conditions, site, etc) to evaluate sky spectrum at each telescope pointing (model_sky1, model_sky2, model_skysci)
    

    Returns
    -------
    sky_line_corr, sky_cont_corr, model_skysci

    """
    pass


def sky_cont_correct(master_sky1_cont, master_sky2_cont, model_skysci):
    """correct and combine continuum only spectra by doing:   sky_cont_corr=0.5*( sky1_cont*(model_skysci/model_sky1) + sky2_cont*(model_skysci/model_sky2)) """
    return sky_cont_corr


def coadd_sky(sky_cont_corr, sky_line_corr):
    """coadd corrected line and continuum combined sky frames:    sky_corr=sky_cont_corr+sky_line_corr"""
    sky_corr = sky_cont_corr+sky_line_corr
    return sky_corr


def sky_lsf_matching():
    """do LSF matching of sky_corr to each science fiber and subtract the LSF-matched corrected sky spectra"""
    pass


def refine_cont_subtraction():
    """optionally apply an extra residual continuum subtraction using the faintest (i.e. with no stellar light detection) spaxels in the science IFU"""
    pass


def pca_residual_subtraction():
    """PCA residual subtraction"""
    pass


def geocoronal_subtraction():
    pass
