"""Cross-match LVM fiber positions with Gaia DR3 via TAP."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from os import environ
from pathlib import Path

import click
import pyvo
from astropy.table import Table
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIBER_DIAMETER_ARCSEC = 35.3
FIBER_RADIUS_DEG = (FIBER_DIAMETER_ARCSEC / 2) / 3600.0

TAP_URL = "https://gaia.ari.uni-heidelberg.de/tap"
TAP_MAXREC = 10000000
DEFAULT_WORKERS = 5

ADQL_TEMPLATE = f"""\
SELECT
    f.fiberid, f.ra as fiber_ra, f.dec as fiber_dec,
    g.source_id, g.ra, g.dec,
    g.parallax, g.parallax_error,
    g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
    g.teff_gspphot, g.logg_gspphot, g.mh_gspphot
FROM TAP_UPLOAD.fiber_upload AS f
JOIN gaiadr3.gaia_source_lite AS g
    ON 1=CONTAINS(POINT('ICRS', g.ra, g.dec),
                  CIRCLE('ICRS', f.ra, f.dec, {FIBER_RADIUS_DEG}))
"""

SUPPORTED_FORMATS = {
    "parquet": (".parquet", "parquet"),
    "fits":    (".fits",    "fits"),
    "vot":     (".vot",     "votable"),
}

REQUIRED_COLUMNS = {"fiberid", "ra", "dec"}

# ---------------------------------------------------------------------------
# Colored output helpers
# ---------------------------------------------------------------------------

def _info(msg):
    click.echo(msg)

def _ok(msg):
    click.echo(click.style(msg, fg="green"))

def _warn(msg):
    click.echo(click.style("WARN: " + msg, fg="yellow"), err=True)

def _err(msg):
    click.echo(click.style("ERROR: " + msg, fg="red", bold=True), err=True)

# ---------------------------------------------------------------------------
# Path defaults from environment
# ---------------------------------------------------------------------------

def _env_path(*keys_and_suffixes):
    """Try env vars in order, append suffix to the first hit."""
    for var, suffix in keys_and_suffixes:
        base = environ.get(var)
        if base:
            return Path(base) / suffix
    return None


def _default_input():
    """Default input from $SAS_BASE_DIR."""
    return _env_path(("SAS_BASE_DIR", "sdsswork/lvm/spectro/redux/1.2.0"))


def _default_output():
    """Default output: $LVM_SANDBOX or $SAS_BASE_DIR/sdsswork/lvm/sandbox."""
    return _env_path(
        ("LVM_SANDBOX", "calib/gaia_cache/sources_by_expnum"),
        ("SAS_BASE_DIR", "sdsswork/lvm/sandbox/calib/gaia_cache/sources_by_expnum"),
    )

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_sframes(input_dir):
    """Find lvmSFrame-????????.fits files recursively."""
    _info(f"Scanning {input_dir} ...")
    t0 = time.time()
    files = sorted(Path(input_dir).rglob("lvmSFrame-????????.fits"))
    _info(f"Found {len(files)} SFrame files in {time.time()-t0:.1f}s")
    return files


def load_file_list(path):
    """Read file paths from a text file (one per line)."""
    lines = Path(path).read_text().strip().splitlines()
    files = [Path(l.strip()) for l in lines if l.strip() and not l.startswith("#")]
    _info(f"Loaded {len(files)} paths from {path}")
    return files


def query_gaia(slitmap, tap):
    """Upload fiber table and cross-match against Gaia."""
    buf = BytesIO()
    slitmap["fiberid", "ra", "dec"].write(buf, format="votable")
    buf.seek(0)
    t0 = time.time()
    result = tap.run_async(
        ADQL_TEMPLATE, uploads={"fiber_upload": buf}, maxrec=TAP_MAXREC
    ).to_table()
    return result, time.time() - t0


def save_table(table, output_base, formats, verbose=False):
    """Write table to disk in each requested format."""
    for fmt in formats:
        ext, astropy_fmt = SUPPORTED_FORMATS[fmt]
        path = output_base + ext
        t0 = time.time()
        table.write(path, format=astropy_fmt, overwrite=True)
        if verbose:
            mb = Path(path).stat().st_size / 1048576
            _ok(f"    {ext} {mb:.2f} MB ({time.time()-t0:.3f}s)")


def _process_one(filepath, output_dir, formats, tap, verbose=False):
    """Read one SFrame, query Gaia, and save results."""
    expnum = filepath.stem.split("-")[-1]
    fail = (expnum, 0, False)

    try:
        slitmap = Table.read(filepath, "SLITMAP", format="fits")
    except Exception as e:
        _err(f"[{expnum}] failed to read SLITMAP: {e}")
        return fail

    if missing := REQUIRED_COLUMNS - set(slitmap.colnames):
        _err(f"[{expnum}] SLITMAP missing columns: {missing}")
        return fail

    try:
        result, dt = query_gaia(slitmap, tap)
    except Exception as e:
        _err(f"[{expnum}] query failed: {e}")
        return fail

    if not result:
        _warn(f"[{expnum}] no Gaia sources found, skipping.")
        return fail

    if verbose:
        _info(f"  [{expnum}] {len(result)} sources in {dt:.1f}s")
    save_table(result, str(output_dir / f"lvmFiberGaia-{expnum}"), formats, verbose)
    return expnum, len(result), True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class ColorHelpCommand(click.Command):
    """Click command with colored help output."""

    def _h(self, text):
        """Bold cyan heading (command/header)."""
        return click.style(text, fg="bright_cyan", bold=True)

    def _section(self, title, lines):
        """Section: bold cyan heading + body lines."""
        return [self._h(title)] + lines + [""]

    def format_help(self, ctx, formatter):
        r = FIBER_DIAMETER_ARCSEC / 2
        adql_lines = ["  " + click.style(l, dim=True) for l in ADQL_TEMPLATE.strip().splitlines()]

        sections = [
            click.style("\nGaia DR3 cross-match for LVM SFrame files", fg="bright_blue", bold=True),
            "",
            *self._section("WORKFLOW", [
                "  For each lvmSFrame-????????.fits found in INPUT_DIR (or --file-list):",
                "    1) Read the SLITMAP extension (fiberid, ra, dec)",
                "    2) Upload fiber positions to Gaia TAP (ARI Heidelberg)",
                "    3) Cross-match against gaiadr3.gaia_source_lite",
                f"       within r = {FIBER_DIAMETER_ARCSEC}/2 = {r} arcsec per fiber",
                "    4) Save result as " + click.style("lvmFiberGaia-{expnum}.{format}", fg="magenta"),
            ]),
            *self._section("ADQL QUERY", adql_lines),
            *self._section("OUTPUT COLUMNS", [
                "  fiberid, fiber_ra, fiber_dec, source_id, ra, dec,",
                "  parallax, parallax_error,",
                "  phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,",
                "  teff_gspphot, logg_gspphot, mh_gspphot",
            ]),
            *self._section("OUTPUT FORMATS", [
                "  " + click.style("parquet", fg="green") + " (default) -- smallest, columnar, fast I/O",
                "  " + click.style("fits", fg="green") + "              -- standard FITS binary table",
                "  " + click.style("vot", fg="green") + "               -- VOTable XML",
            ]),
            *self._section("DEFAULT PATHS", [
                "  input:  " + click.style("$SAS_BASE_DIR", fg="yellow") + "/sdsswork/lvm/spectro/redux/1.2.0",
                "          (only needed if " + click.style("-i", fg="yellow") + " is not provided)",
                "  output: " + click.style("$LVM_SANDBOX", fg="yellow") + "/calib/gaia_cache/sources_by_expnum/",
                "          if $LVM_SANDBOX is not set, constructed as",
                "          " + click.style("$SAS_BASE_DIR", fg="yellow") + "/sdsswork/lvm/sandbox/calib/gaia_cache/...",
                click.style("  input without -i requires $SAS_BASE_DIR; output without -o", fg="yellow"),
                click.style("  requires $LVM_SANDBOX or $SAS_BASE_DIR. Otherwise exits.", fg="yellow"),
            ]),
        ]
        click.echo("\n".join(sections))

        click.echo(self._h("OPTIONS"))
        opts = [p.get_help_record(ctx) for p in self.get_params(ctx)]
        opts = [o for o in opts if o]
        if opts:
            f = click.HelpFormatter()
            with f.indentation():
                f.write_dl(opts)
            click.echo(f.getvalue())


@click.command(cls=ColorHelpCommand)
@click.option(
    "--input-dir", "-i", default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing lvmSFrame-????????.fits files (searched recursively). "
         "If not given, resolved from $SAS_BASE_DIR. Ignored if --file-list is used.",
)
@click.option(
    "--file-list", "-l", default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Text file with SFrame paths (one per line). Skips directory scanning.",
)
@click.option(
    "--output-dir", "-o", default=None,
    type=click.Path(file_okay=False),
    help="Directory where lvmFiberGaia-{expnum}.{fmt} files are saved. "
         "Created if missing. Resolved from $LVM_SANDBOX or $SAS_BASE_DIR.",
)
@click.option(
    "--format", "-f", "formats",
    multiple=True, default=["parquet"],
    type=click.Choice(list(SUPPORTED_FORMATS.keys()), case_sensitive=False),
    help="Output format. Can be repeated (e.g. -f parquet -f fits). Default: parquet.",
)
@click.option(
    "--test-limit", default=None, type=int,
    help="Process only the first N frames. Useful for quick testing.",
)
@click.option(
    "--workers", "-w", default=DEFAULT_WORKERS, type=int,
    show_default=True,
    help="Number of parallel threads for TAP queries.",
)
@click.option(
    "-v", "--verbose", is_flag=True,
    help="Print per-file query time, source count, and saved file sizes.",
)
def main(input_dir, file_list, output_dir, formats, test_limit, workers, verbose):
    """Batch-fetch Gaia sources for LVM SFrame files."""
    def _resolve(val, default_fn, err_msg):
        if val:
            return Path(val)
        resolved = default_fn()
        if resolved is None:
            _err(err_msg)
            sys.exit(1)
        return resolved

    output_dir = _resolve(
        output_dir, _default_output,
        "Neither $LVM_SANDBOX nor $SAS_BASE_DIR is set. Provide --output-dir.",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_list:
        files = load_file_list(file_list)
    else:
        input_dir = _resolve(
            input_dir, _default_input,
            "$SAS_BASE_DIR is not set. Provide --input-dir, --file-list, or set $SAS_BASE_DIR.",
        )
        files = find_sframes(input_dir)

    if test_limit is not None:
        files = files[:test_limit]

    src = file_list if file_list else input_dir
    _info(f"source:  {src}")
    _info(f"output:  {output_dir}")
    _info(f"formats: {','.join(formats)}  workers: {workers}  files: {len(files)}")

    if not files:
        _warn("No lvmSFrame files found. Nothing to do.")
        return

    tap = pyvo.dal.TAPService(TAP_URL)
    ok_count = 0
    t_total = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_one, fp, output_dir, formats, tap, verbose
            ): fp
            for fp in files
        }
        pbar = tqdm(
            as_completed(futures), total=len(futures),
            desc="Fetching", unit="frame", ascii=True,
        )
        for future in pbar:
            fp = futures[future]
            try:
                expnum, nsrc, success = future.result()
                if success:
                    ok_count += 1
                    pbar.set_postfix_str(f"{expnum}: {nsrc} src")
            except Exception as e:
                _err(f"unexpected error {fp}: {e}")

    dt = time.time() - t_total
    _ok(f"Done: {ok_count}/{len(files)} frames in {dt:.0f}s")


if __name__ == "__main__":
    main()
