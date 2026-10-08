# encoding: utf-8
#
"""Adapter between lvmdrp and the vendored lvmsky sky-model code.

The lvmsky code (sky decomposition and machine-learning sky prediction) is
kept as an unmodified copy in ``lvmdrp/external/lvmsky`` (see
``bin/sync_lvmsky``).  This module is the only part of lvmdrp that imports it.
It builds lvmsky's inputs from an lvmCFrame and returns what the sky methods in
``functions/skyMethod.py`` need.

The lvmsky data bundle (PALACE tables, solar spectrum, moon/zodiacal-light
assets, sensitivity curves) and any trained model live in a sky-data
directory, by default ``$LVM_MASTER_DIR/sky_models/<version>/``, with the data
bundle in its ``data/`` subdirectory.  lvmsky finds the bundle through the
``LVMSKY_DATA_ROOT`` environment variable, which this module sets before the
first import of the lvmsky code.
"""

import os
import pathlib
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from lvmdrp import log
from lvmdrp.core.constants import MASTERS_DIR


LVMSKY_DIR = pathlib.Path(__file__).resolve().parent.parent / "external" / "lvmsky"
DATA_ROOT_ENV = "LVMSKY_DATA_ROOT"
# decompose_parallel's default --factor: CFrame flux (erg/s/cm^2/A) times this
# is the O(1) scale the decomposition fits in; its components come back scaled.
FACTOR = 1e14
# line-like decomposition components; the rest of the model is the continuum
LINE_KEYS = ("oh", "atom", "orc", "o2")
TELESCOPES = ("Sci", "SkyE", "SkyW")

_LOADED = {}


class XSkyError(Exception):
    """A sky-model step cannot run for this exposure (the caller falls back)."""


def lvmsky_source():
    """Return the vendored lvmsky provenance from its SOURCE file as a dict."""
    info = {}
    source = LVMSKY_DIR / "SOURCE"
    if source.exists():
        for line in source.read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() in ("repository", "ref", "commit", "commit date"):
                info[key.strip()] = value.strip()
    return info


def sky_data_dir(version=None, path=None):
    """Sky-data directory: the lvmsky data bundle (in data/) and any trained model.

    ``path`` wins if given; otherwise ``$LVM_MASTER_DIR/sky_models/<version>``.
    """
    if path:
        return pathlib.Path(path).expanduser().resolve()
    if not version:
        raise XSkyError("no sky_model_version or sky_data_dir configured")
    if not MASTERS_DIR:
        raise XSkyError("LVM_MASTER_DIR is not set")
    return pathlib.Path(MASTERS_DIR) / "sky_models" / version


def load_decomposition(model_dir):
    """Import the vendored lvmsky decomposition driver with its data bundle.

    ``LVMSKY_DATA_ROOT`` is read by lvmsky when it is first imported, so the
    data root cannot change within one Python process; asking for a different
    one raises XSkyError.
    """
    data_root = pathlib.Path(model_dir) / "data"
    if not (data_root / "bundle_manifest.json").exists():
        raise XSkyError(f"no lvmsky data bundle at {data_root}")
    if "decompose_parallel" in _LOADED:
        if _LOADED["data_root"] != data_root:
            raise XSkyError(f"lvmsky already loaded with data root {_LOADED['data_root']}, "
                            f"cannot switch to {data_root} in the same process")
        return _LOADED["decompose_parallel"]

    current = os.environ.get(DATA_ROOT_ENV)
    if current and pathlib.Path(current).resolve() != data_root.resolve():
        log.warning(f"{DATA_ROOT_ENV}={current} is overridden by the sky-data directory's {data_root}")
    os.environ[DATA_ROOT_ENV] = str(data_root)
    for p in (LVMSKY_DIR / "skysub", LVMSKY_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import decompose_parallel  # noqa: E402 -- vendored lvmsky, found through sys.path

    _LOADED.update(decompose_parallel=decompose_parallel, data_root=data_root)
    log.info(f"loaded lvmsky {lvmsky_source().get('commit', '?')[:8]} with data root {data_root}")
    return decompose_parallel


def _fill_lsf(lsf):
    """Replace non-finite or non-positive LSF pixels by linear interpolation."""
    lsf = np.array(lsf, dtype=np.float64)
    good = np.isfinite(lsf) & (lsf > 0)
    if not good.any():
        raise XSkyError("LSF has no valid pixel")
    if not good.all():
        pix = np.arange(lsf.size)
        lsf[~good] = np.interp(pix[~good], pix[good], lsf[good])
    return lsf


def telescope_lsfs(cframe_path):
    """Median LSF FWHM (A) of the good fibers of each telescope, and fiber counts."""
    with fits.open(cframe_path) as hdul:
        slitmap = Table(hdul["SLITMAP"].data)
        lsf = np.asarray(hdul["LSF"].data, dtype=np.float64)
    good = np.asarray(slitmap["fibstatus"]) == 0
    tel = np.char.strip(np.asarray(slitmap["telescope"]).astype(str))
    lsfs, counts = {}, {}
    for name in TELESCOPES:
        sel = good & (tel == name)
        if not sel.any():
            raise XSkyError(f"no good {name} fibers")
        with np.errstate(all="ignore"):
            lsfs[name] = _fill_lsf(np.nanmedian(lsf[sel], axis=0))
        counts[name] = int(sel.sum())
    return lsfs, counts


def near_far(header):
    """Labels of the sky telescopes nearest to and farthest from the science pointing."""
    sci = SkyCoord(header["SCIRA"], header["SCIDEC"], unit="deg")
    sep = {name: sci.separation(SkyCoord(header[f"{key}RA"], header[f"{key}DEC"], unit="deg")).deg
           for name, key in (("SkyE", "SKYE"), ("SkyW", "SKYW"))}
    near = min(sep, key=sep.get)
    return near, ("SkyW" if near == "SkyE" else "SkyE")


def build_stack(header, wave, spectra, lsfs, counts):
    """One exposure as the one-row stack that lvmsky's decompose_parallel reads.

    ``spectra`` and ``lsfs`` are dicts keyed by "Sci", "SkyE", "SkyW" holding
    flux in erg/s/cm^2/A and LSF FWHM in A on ``wave``; ``counts`` the number
    of fibers combined.  META carries the columns the telluric fit models use.
    """
    near, far = near_far(header)
    pos = {"Sci": ("SCIRA", "SCIDEC"), "SkyE": ("SKYERA", "SKYEDEC"), "SkyW": ("SKYWRA", "SKYWDEC")}
    ra = {name: float(header[k[0]]) for name, k in pos.items()}
    dec = {name: float(header[k[1]]) for name, k in pos.items()}
    pwv = header.get("PWV_MED", np.nan)
    meta = Table(rows=[dict(
        expnum=int(header["EXPOSURE"]), mjd=int(header.get("MJD", -1)),
        date_obs=str(header["OBSTIME"]).strip(), obstime=str(header["OBSTIME"]).strip(),
        exptime=float(header.get("EXPTIME", 900.0)), fluxcal=str(header.get("FLUXCAL", "")),
        pwv_med=float(pwv) if pwv is not None else np.nan,
        sci_ra=ra["Sci"], sci_dec=dec["Sci"], skye_ra=ra["SkyE"], skye_dec=dec["SkyE"],
        skyw_ra=ra["SkyW"], skyw_dec=dec["SkyW"],
        sky_near_label=near, sky_far_label=far,
        sky_near_ra=ra[near], sky_near_dec=dec[near], sky_far_ra=ra[far], sky_far_dec=dec[far],
        sci_airmass=float(header["SCIAM"]), skye_airmass=float(header["SKYEAM"]),
        skyw_airmass=float(header["SKYWAM"]),
        fibers_sci_used=counts["Sci"], fibers_sky_near_used=counts[near],
        fibers_sky_far_used=counts[far],
    )])

    def image(arr, name):
        return fits.ImageHDU(np.asarray(arr, dtype=np.float32)[None, :], name=name)

    return fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.asarray(wave, dtype=np.float64), name="WAVE"),
        image(spectra["Sci"], "FLUX_SCI"),
        image(spectra[near], "FLUX_SKY_NEAR"),
        image(spectra[far], "FLUX_SKY_FAR"),
        image(lsfs["Sci"], "LSF_SCI"),
        image(lsfs[near], "LSF_SKY_NEAR"),
        image(lsfs[far], "LSF_SKY_FAR"),
        fits.BinTableHDU(meta, name="META"),
    ])


def continuum_from_fit(fit):
    """The model continuum of a decomposition, in erg/s/cm^2/A: the best fit
    minus its line components (moonlight, zodiacal light, airglow continua)."""
    lines = np.zeros_like(np.asarray(fit.bestfit_lsf, dtype=np.float64))
    for key in LINE_KEYS:
        comp = fit.components.get(key)
        if comp is not None:
            lines += np.asarray(comp, dtype=np.float64)
    return (np.asarray(fit.bestfit_lsf, dtype=np.float64) - lines) / FACTOR


def separate(cframe_path, header, wave, spectra, model_dir, decompose_science=True):
    """Continuum/line separation of the Sci, SkyE and SkyW spectra by lvmsky's decomposition.

    Parameters
    ----------
    cframe_path : str
        The lvmCFrame (for the per-telescope LSFs and fiber counts).
    header : fits.Header
        Its primary header (pointings, airmasses, PWV, time).
    wave : array
        Wavelength grid of ``spectra``.
    spectra : dict
        "Sci", "SkyE", "SkyW" spectra in erg/s/cm^2/A, e.g. the biweight means
        quick_sky_subtraction already computes.
    model_dir : path
        Sky-data directory (see sky_data_dir); its data/ is the lvmsky bundle.
    decompose_science : bool
        Decompose the Sci spectrum too.  If False, its entry is None and the
        caller separates it another way.

    Returns
    -------
    dict
        Keyed "sci", "skye", "skyw": dict(continuum, lines, status), and
        "info": dict(near, far, statuses).
    """
    if str(header.get("FLUXCAL", "")).strip() != "MOD":
        raise XSkyError(f"FLUXCAL={header.get('FLUXCAL')!r}: the lvmsky decomposition assumes "
                        f"the DRP's MOD telluric correction")
    dp = load_decomposition(model_dir)
    lsfs, counts = telescope_lsfs(cframe_path)
    stack = build_stack(header, wave, spectra, lsfs, counts)
    near, far = near_far(header)
    kinds = ("sky1", "sky2", "sci") if decompose_science else ("sky1", "sky2")
    fits_ = dp.decompose_in_process(stack, rows=(0,), kinds=kinds)

    tel_of_kind = {"sky1": near, "sky2": far, "sci": "Sci"}
    out = {"sci": None, "info": {"near": near, "far": far, "statuses": {}}}
    for kind in kinds:
        fit, _flags = fits_[(kind, 0)]
        tel = tel_of_kind[kind]
        status = str(fit.fit_status)
        out["info"]["statuses"][tel] = status
        if status not in ("Solved", "AlmostSolved"):
            raise XSkyError(f"decomposition of {tel} did not solve: {status}")
        cont = continuum_from_fit(fit)
        out[tel.lower()] = dict(continuum=cont, lines=np.asarray(spectra[tel], dtype=np.float64) - cont,
                                status=status)
    return out
