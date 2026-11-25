# encoding: utf-8
#
# @Author: Alfredo Mejía-Narváez
# @Date: May 16, 2022
# @Filename: constants.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import functools
import os
import subprocess

from lvmdrp.utils import get_env_lib_directory


# LVMi constants
LVM_ELEVATION = 2380.0 # in m
LVM_LAT = '29.0146S'
LVM_LON = '70.6926W'

# sources server URL
LVM_UNAM_URL = "http://ifs.astroscu.unam.mx/LVM"
LVM_SRC_URL = f"{LVM_UNAM_URL}/lvmdrp_src.zip"

# installation path
INS_PATH = os.getenv("LVM_ESOSKY_DIR", get_env_lib_directory())

LIB_PATH = os.path.join(INS_PATH, "lib")
BIN_PATH = os.path.join(INS_PATH, "bin")
INC_PATH = os.path.join(INS_PATH, "include")

# sources path
SRC_PATH = os.path.abspath("src")
SKYCORR_SRC_PATH = os.path.join(SRC_PATH, "skycorr.tar.gz")
SKYMODEL_SRC_PATH = os.path.join(SRC_PATH, "SM-01.tar.gz")

# installation path for ESO routines
SKYCORR_INST_PATH = os.path.join(LIB_PATH, "skycorr")
SKYMODEL_INST_PATH = os.path.join(LIB_PATH, "skymodel")

# root package path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# configuration paths
CONFIG_PATH = os.path.join(ROOT_PATH, "etc")

# dataproduct blueprints path
DATAPRODUCT_BP_PATH = os.path.join(CONFIG_PATH, "dataproducts")

# main DRP configuration path
MASTER_CONFIG_PATH = os.path.join(CONFIG_PATH, "drp_master_config.yaml")

# ESO sky model (web version) configuration files
SKYCALC_CONFIG_PATH = os.path.join(CONFIG_PATH, "third_configs", "skycalc.json")
ALMANAC_CONFIG_PATH = os.path.join(CONFIG_PATH, "third_configs", "almanac.json")

# ESO sky model master configuration file
SKYMODEL_CONFIG_PATH = os.path.join(CONFIG_PATH, "third_configs", "skymodel_config.yml")
# high-level ESO sky model configuration files
SKYMODEL_INST_CONFIG_PATH = os.path.join(
    SKYMODEL_INST_PATH, "sm-01_mod2", "config", "instrument_etc_ref.par"
)
SKYMODEL_MODEL_CONFIG_PATH = os.path.join(
    SKYMODEL_INST_PATH, "sm-01_mod2", "config", "skymodel_etc_ref.par"
)

# ESO skycorr configuration file
SKYCORR_CONFIG_PATH = os.path.join(CONFIG_PATH, "third_configs", "skycorr_config.yml")

# ephemeris path
EPHEMERIS_DIR = os.path.join(os.getenv("LVMCORE_DIR"), "etc")

# fiducial calibrations directory: read from $LVM_MASTER_DIR when it is needed (not at import),
# so that it can be changed for a run (drp run --calib-dir, see sdss/lvmdrp#272)
def get_masters_dir():
    """Returns the calibration directory in use ($LVM_MASTER_DIR)."""
    return os.getenv("LVM_MASTER_DIR")


def set_masters_dir(calib_dir):
    """Sets the calibration directory for the rest of the run.

    Sets $LVM_MASTER_DIR, which is what the calibration paths are built from,
    so that the choice holds even when something else (e.g. an sdss tree setup
    in a calling tool) has changed the environment before.

    Args:
        calib_dir: the calibration directory, laid out as $LVM_MASTER_DIR
            (<mjd>/lvm-m<kind>-<camera>.fits, pixelmasks/, stellar_models/, ...).

    Returns:
        The absolute path set.

    Raises:
        FileNotFoundError: if `calib_dir` is not a directory.
    """
    calib_dir = os.path.abspath(os.path.expanduser(calib_dir))
    if not os.path.isdir(calib_dir):
        raise FileNotFoundError(f"calibration directory {calib_dir} not found")
    os.environ["LVM_MASTER_DIR"] = calib_dir
    return calib_dir


def configured_calib_dir(config=None):
    """Returns the calibration directory named by ``calib_dir`` in lvmdrp.yaml, or None.

    A branch can name its own calibration set this way, so that checking out the
    branch also selects its calibrations. A relative path is taken from
    $SAS_BASE_DIR/sdsswork/lvm/ (e.g. ``sandbox/calib_2.0.0dev``), so that the same
    entry works on every machine.

    Args:
        config: the lvmdrp configuration (a mapping); default: ``lvmdrp.config``.

    Returns:
        The absolute path, or None if lvmdrp.yaml names none.
    """
    if config is None:
        from lvmdrp import config
    calib_dir = (config or {}).get("calib_dir")
    if not calib_dir:
        return None
    calib_dir = os.path.expanduser(str(calib_dir))
    if not os.path.isabs(calib_dir):
        calib_dir = os.path.join(os.getenv("SAS_BASE_DIR", ""), "sdsswork", "lvm", calib_dir)
    return os.path.abspath(calib_dir)


def calib_dir_in_args(args):
    """Whether a command line (a list of arguments) gives --calib-dir."""
    return any(a == "--calib-dir" or a.startswith("--calib-dir=") for a in args)


def help_requested(group, ctx, args):
    """Whether a ``drp`` command line only asks for help.

    Help needs no calibrations, so the ``drp`` group does not check the calibration
    directory for it: ``drp run --help`` works on a computer without the directory
    named in lvmdrp.yaml (sdss/lvmdrp#272).

    Args:
        group: the ``drp`` click group.
        ctx: its click context.
        args: the arguments after ``drp``.

    Returns:
        True if the arguments include a help option, or end with a group (such as
        ``drp calibrations``), which then prints its help.
    """
    import click
    if any(a in ctx.help_option_names for a in args):
        return True
    cmd = group
    while isinstance(cmd, click.Group):
        if not args:
            return True
        cmd = cmd.get_command(ctx, args[0])
        args = args[1:]
    return False


def apply_configured_calib_dir(config=None, explicit=False):
    """Uses the calibration directory of lvmdrp.yaml for the run, if it names one.

    Called when a ``drp`` command starts. The order of precedence is: --calib-dir on
    the command line (`explicit`), then ``calib_dir`` in lvmdrp.yaml, then
    $LVM_MASTER_DIR. A directory named in lvmdrp.yaml that does not exist is an
    error, not a reason to fall back to $LVM_MASTER_DIR, which would quietly reduce
    with other calibrations (sdss/lvmdrp#272).

    Args:
        config: the lvmdrp configuration; default: ``lvmdrp.config``.
        explicit: True if the command line gives --calib-dir, which then takes precedence.

    Returns:
        The directory set, or None if lvmdrp.yaml names none (or --calib-dir is given).

    Raises:
        FileNotFoundError: if the directory named in lvmdrp.yaml does not exist.
    """
    if explicit:
        return None
    calib_dir = configured_calib_dir(config)
    if calib_dir is None:
        return None
    if not os.path.isdir(calib_dir):
        raise FileNotFoundError(f"calibration directory {calib_dir}, named by calib_dir in lvmdrp.yaml, "
                                "not found: create it, or give --calib-dir")
    return set_masters_dir(calib_dir)


@functools.lru_cache(maxsize=None)
def _git_version(repo_dir):
    """Returns the commit of a git checkout (12 characters, -dirty if tracked files differ), or 'unknown'."""
    if not repo_dir or not os.path.isdir(repo_dir):
        return "unknown"
    try:
        run = functools.partial(subprocess.run, cwd=repo_dir, capture_output=True, text=True, timeout=30)
        head = run(["git", "rev-parse", "--short=12", "HEAD"])
        if head.returncode != 0:
            return "unknown"
        dirty = run(["git", "diff", "--quiet", "HEAD", "--"]).returncode != 0
        return head.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def get_lvmcore_version():
    """Returns the lvmcore commit in use ($LVMCORE_DIR), to record with the products.

    lvmcore holds the calibration and ffactor epochs, the exposure excludes, the header
    fixes and other inputs that change independently of lvmdrp, so the commit is needed
    to reproduce a reduction (sdss/lvmdrp#272).

    Returns:
        The commit hash (12 characters), with '-dirty' if tracked files differ from it,
        or 'unknown' if $LVMCORE_DIR is not a git checkout.
    """
    return _git_version(os.getenv("LVMCORE_DIR"))


def __getattr__(name):
    # MASTERS_DIR is kept for code outside lvmdrp; inside it, use get_masters_dir()
    if name == "MASTERS_DIR":
        return get_masters_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# path for pixel shifts table
PIXELSHIFTS_PATH = os.path.join(os.getenv("LVMCORE_DIR"), "etc", "pixel_shifts.parquet")

# path to validated electronic pixel shifts
PIXELSHIFTS_DIR = os.path.join(os.getenv("LVMCORE_DIR"), "pixelshifts")

SKYCORR_PAR_MAP = {
    "INPUT_OBJECT_SPECTRUM": "objfile",
    "INPUT_SKY_SPECTRUM": "skyfile",
    "OUTPUT_NAME": "outfile",
    "OUTPUT_DIR": "outdir",
    "COL_NAMES": "colnames",
    "INST_DIR": "install",
    "DEFAULT_ERROR": "defaultError",
    "WLG_TO_MICRON": "wave2Micron",
    "VAC_AIR": "vacOrAir",
    "DATE_KEY": "dateKey",
    "TIME_KEY": "timeKey",
    "TELALT_KEY": "telAltKey",
    "LINETABNAME": "linetab",
    "VARDATNAME": "vardat",
    "SOLDATURL": "soldaturl",
    "SOLFLUX": "solflux",
    "FWHM": "fwhm",
    "VARFWHM": "varfwhm",
    "LTOL": "ltol",
    "MIN_LINE_DIST": "minLineDist",
    "FLUXLIM": "fluxLim",
    "FTOL": "ftol",
    "XTOL": "xtol",
    "WTOL": "wtol",
    "CHEBY_MAX": "chebyMax",
    "CHEBY_MIN": "chebyMin",
    "CHEBY_CONST": "chebyConst",
    "REBINTYPE": "rebinType",
    "WEIGHTLIM": "weightLim",
    "SIGLIM": "sigLim",
    "FITLIM": "fitLim",
    "PLOT_TYPE": "plotType",
}

# high-level calibration types
CALIBRATION_TYPES = {"bias", "trace", "wave", "dome", "twilight"}
# calibration products
CALIBRATION_PRODUCTS = {"pixmask", "pixflat", "bias", "trace_guess", "centroids", "sigmas", "counts", "model", "wave", "lsf", "fiberflat_dome", "fiberflat_twilight"}
# a map for high-level calibration types and their products
CALIBRATION_MAPPINGS = {
    "trace": ["centroids", "sigmas", "model"],
    "wave": ["wave", "lsf"],
    "dome": ["fiberflat_dome"],
    "twilight": ["fiberflat_twilight"]}
# calibration products needed depending on the type of frame
CALIBRATION_NEEDS = {
    "bias": ["pixmask"],
    "pixflat": ["pixmask", "bias", "dark"],
    "trace": ["pixmask", "pixflat", "bias"],
    "wave": ["pixmask", "pixflat", "bias", "centroids", "sigmas", "model"],
    "dome": ["pixmask", "pixflat", "bias", "centroids", "sigmas", "model", "wave", "lsf"],
    "twilight": ["pixmask", "pixflat", "bias", "centroids", "sigmas", "model", "wave", "lsf"],
    "object": ["pixmask", "pixflat", "bias", "centroids", "sigmas", "model", "wave", "lsf", "fiberflat_twilight"],
}

CAMERAS = ["b1", "b2", "b3", "r1", "r2", "r3", "z1", "z2", "z3"]

# spectrograph channels as spec
# SPEC_CHANNELS = {"b": (3600, 5930), "r": (5660, 7720), "z": (7470, 9800)}
# spectrograph channels as built, removing regions past the dichroics passbands
SPEC_CHANNELS = {"b": (3600, 5800), "r": (5775, 7570), "z": (7520, 9800)}

ARC_LAMPS = ["NEON", "HGNE", "ARGON", "XENON"]
CON_LAMPS = ["LDLS", "QUARTZ"]

# 2D image constants
LVM_NOVER = 17
LVM_NPRE = 3
LVM_NROWS = 4080
LVM_NCOLS = 4086

LVM_NBLOCKS = 18
LVM_BLOCKSIZE = 36
LVM_NFIBERS = LVM_NBLOCKS * LVM_BLOCKSIZE
LVM_REFERENCE_COLUMN = 2000
FIDUCIAL_PLATESCALE = 112.36748321030637 # Focal plane platescale in "/mm

# standard fiber labels
STD_FIBER_LABELS = [
    'P1-1',
    'P2-2',
    'P1-4',
    'P2-5',
    'P2-8',
    'P1-7',
    'P2-11',
    'P1-10',
    'P1-3',
    'P2-1',
    'P1-6',
    'P2-4',
    'P2-7',
    'P1-9',
    'P2-10',
    'P1-12',
    'P1-2',
    'P2-3',
    'P1-5',
    'P2-6',
    'P2-9',
    'P1-8',
    'P2-12',
    'P1-11']

# GB hand picked isolated bright lines across each channel which are not doublest in UVES atlas
# true wavelengths taken from UVES sky line atlas
REF_SKYLINES = {
    "b": [5577.346680],
    "r": [6363.782715, 7358.680176, 7392.209961],
    "z": [8399.175781, 8988.383789, 9552.546875, 9719.838867]
}

# sky lines to fit spec-to-spec flat field factors
SKYLINES_FIBERFLAT = {
    "b": 5577.346680,
    "r": 6363.782715,
    "z": 8399.175781
}

# continuum wavelengths
CONTINUUM_FIBERFLAT = {
    "b": 4600,
    "r": 6760,
    "z": 8170
}

# stellar templates
STELLAR_TEMP_PATH = os.path.join(os.getenv("LVM_SANDBOX"), "stellar_models")
