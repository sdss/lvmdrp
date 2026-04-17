
import os
import yaml
import numpy as np
import pandas as pd

from lvmdrp import log


EXCLUDES_PATH = os.path.join(os.path.expandvars("$LVMCORE_DIR"), "etc", "exclude-mjds.yaml")


def _parse_list(items_str):
    """Parse a string representation of a list or range into a list of integers.

    Handles three formats:
    - Comma-separated values: "1,2,3" -> [1, 2, 3]
    - Range: "1-5" -> [1, 2, 3, 4, 5]
    - Single value: "42" -> [42]

    Parameters
    ----------
    items_str : str or int
        String representation of items to parse, or already an integer.

    Returns
    -------
    list of int
        List of parsed integer values.

    Examples
    --------
    >>> _parse_list("1,3,5")
    [1, 3, 5]
    >>> _parse_list("10-15")
    [10, 11, 12, 13, 14, 15]
    >>> _parse_list("42")
    [42]
    """
    # handle list
    if isinstance(items_str, str) and "," in items_str:
        items = [int(i) for i in items_str.split(",") if i]
    # handle range
    elif isinstance(items_str, str) and "-" in items_str:
        item_i, item_f = items_str.split("-")
        if not item_i or not item_f:
            items = []
        else:
            items = list(range(int(item_i), int(item_f)+1))
    # handle rest
    else:
        items = [int(items_str)]

    return items

def load_excludes(excludes_path=EXCLUDES_PATH):
    """Load and validate exclusion rules from a YAML file.

    Reads a YAML file containing exclusion criteria and returns a pandas DataFrame
    with normalized exclusion rules. Validates that all keywords are valid types.

    Parameters
    ----------
    excludes_path : str, optional
        Path to the YAML file containing exclusion rules. Defaults to the standard
        location in $LVMCORE_DIR/etc/exclude-mjds.yaml.

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

        # map each keyword to a list of exposure numbers
        items = _parse_list(exclude.value)
        if exclude.keyword == "mjd":
            to_exclude |= np.isin(frames.mjd.values, items)
        elif exclude.keyword == "tileid":
            to_exclude |= np.isin(frames.tileid.values, items)
        elif exclude.keyword == "expnum":
            to_exclude |= np.isin(frames.expnum.values, items)

        log.debug(f"{idx}: Excluding frames with {exclude.keyword} in {items}.")

    log.info(f"Excluding {to_exclude.sum()} frames based on excludes table.")

    return frames[~to_exclude].reset_index(drop=True)