
import os
import yaml
import numpy as np
import pandas as pd

from lvmdrp import log
from lvmdrp.utils import parse_list


EXCLUDES_PATH = os.path.join(os.path.expandvars("$LVMCORE_DIR"), "etc", "exclude-exposures.yaml")


def load_excludes(excludes_path=EXCLUDES_PATH):
    """Load and validate exclusion rules from a YAML file.

    Reads a YAML file containing exclusion criteria and returns a pandas DataFrame
    with normalized exclusion rules. Validates that all keywords are valid types.

    Parameters
    ----------
    excludes_path : str, optional
        Path to the YAML file containing exclusion rules. Defaults to the standard
        location in $LVMCORE_DIR/etc/exclude-exposures.yaml.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - keyword : str
            Type of exclusion criterion ('mjd', 'tileid', or 'expnum')
        - value : str
            The exclusion value(s) to match

    Raises
    ------
    ValueError
        If the 'keyword' column contains invalid values other than 'mjd', 'tileid', or 'expnum'.

    """
    with open(excludes_path, 'r') as f:
        excludes = pd.json_normalize(yaml.safe_load(f)["excludes"])

    # validate 'keyword' column
    valid_keywords = excludes["keyword"].isin({"mjd", "tileid", "expnum"})
    if not valid_keywords.all():
        invalids = excludes.keyword[~valid_keywords].unique().tolist()
        raise ValueError(f"Invalid value in `keyword` column: {invalids}. Expected 'mjd', 'tileid' or 'expnum'")

    excludes["value"] = excludes["value"].astype(str)

    return excludes


def match_values(values, spec):
    """Return a boolean mask of `values` matching an exclusion value specification.

    Supports the conventions x; xi-xf; xi-; -xf; x1,...,xn. Ranges are inclusive.

    Parameters
    ----------
    values : array-like
        Values to test (e.g., MJDs, tile IDs or exposure numbers).
    spec : str
        Value specification.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where `values` match `spec`.
    """
    values = np.asarray(values)
    spec = str(spec).strip()
    if "," not in spec and "-" in spec:
        item_i, item_f = spec.split("-")
        mask = np.ones(values.shape, dtype="bool")
        if item_i:
            mask &= values >= int(item_i)
        if item_f:
            mask &= values <= int(item_f)
        return mask
    return np.isin(values, parse_list(spec))


def remove_excluded_frames(frames, excludes):
    """Remove frames that match exclusion criteria from a DataFrame.

    Filters out frames based on exclusion rules defined in the excludes DataFrame.
    Supports exclusion by MJD (Modified Julian Date), tile ID, or exposure number.

    Parameters
    ----------
    frames : pandas.DataFrame
        DataFrame containing frame information with columns 'mjd', 'tileid', and 'expnum'.
    excludes : pandas.DataFrame
        DataFrame with exclusion rules, typically loaded from load_excludes().
        Must have columns 'keyword' and 'value'.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with excluded frames removed. Index is reset.

    Notes
    -----
    The function logs debug information for each exclusion rule applied and
    logs the total number of frames excluded.
    """
    to_exclude = np.zeros(frames.shape[0], dtype="bool")
    for idx, exclude in excludes.iterrows():
        to_exclude |= match_values(frames[exclude.keyword].values, exclude.value)
        log.debug(f"{idx}: excluding exposures with {exclude.keyword} in {exclude.value}.")

    if to_exclude.any():
        expnums = frames[to_exclude].expnum.unique()
        log.info(f"excluding {expnums.size} exposure(s):")
        log.info(f"  {expnums}")

    return frames[~to_exclude].reset_index(drop=True)
