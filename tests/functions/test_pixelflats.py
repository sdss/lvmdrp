# encoding: utf-8
#

import os

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from lvmdrp import path
from lvmdrp.core.image import Image
from lvmdrp.functions import pixelflats as pf
from lvmdrp.qa import pixelflats as qa


def test_parse_expnums_ranges_are_half_open():
    expnums = pf._parse_expnums(["10, 13", 20, "15, 16"])
    assert expnums.tolist() == [10, 11, 12, 15, 20]


def test_parse_expnums_rejects_bad_items():
    with pytest.raises(TypeError):
        pf._parse_expnums(["10-13"])


def test_compress_expnums_roundtrip():
    expnums = [1, 2, 5, 6, 7, 10]
    compressed = pf._compress_expnums(expnums)
    assert compressed == ["1, 3", "5, 8", 10]
    assert pf._parse_expnums(compressed).tolist() == expnums


@pytest.mark.parametrize("kind, pairs", [("2f1d", [(2, "f"), (1, "d")]), ("10f2b", [(10, "f"), (2, "b")])])
def test_parse_kind(kind, pairs):
    assert pf._parse_kind(kind) == pairs


@pytest.mark.parametrize("kind", ["", "2f1x", "f1d", "0f1d"])
def test_parse_kind_invalid(kind):
    with pytest.raises(ValueError):
        pf._parse_kind(kind)


def test_expand_sequence_with_rejects():
    sequence = pf._parse_sequence({"kind": "2f1d", "expnums": ["100, 109"], "rejects": [103, 104, 105]})
    assert sequence["flat"].tolist() == [100, 101, 106, 107]
    assert sequence["dark"].tolist() == [102, 108]


def test_expand_sequence_left_over_exposures():
    with pytest.raises(ValueError, match="can't be split into groups of 3"):
        pf._parse_sequence({"kind": "2f1d", "expnums": ["100, 104"]})


def test_expand_sequence_repeated_type():
    with pytest.raises(ValueError, match="repeats an exposure type"):
        pf._parse_sequence({"kind": "1f1d1f", "expnums": ["100, 106"]})


def test_auto_sequence_requires_metadata_context():
    with pytest.raises(ValueError, match="requires `mjds` and `camera`"):
        pf._parse_sequence({"kind": pf.AUTO_KIND, "expnums": ["100, 106"]})


def test_artifact_centroids_largest_first():
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:4, 2:4] = True        # 4 pixels
    mask[10:16, 10:16] = True    # 36 pixels
    mask[30:33, 30:33] = True    # 9 pixels
    labels, n = ndi.label(mask)
    artifacts = pf._calculate_artifact_centroids([(mask, labels, n)], max_nregions=2)
    assert artifacts == [[(12, 12), (31, 31)]]


def test_artifact_centroids_empty_bin():
    mask = np.zeros((5, 5), dtype=bool)
    assert pf._calculate_artifact_centroids([(mask, np.zeros_like(mask, dtype=int), 0)]) == [[]]


@pytest.mark.parametrize("reverse", [False, True])
def test_measure_contrast_runs_from_center_outwards(reverse):
    pixels = np.arange(1, 1001)
    cut = np.linspace(2.0, 1.0, pixels.size)  # brightest at pixel 1, near the center
    if reverse:
        pixels, cut = pixels[::-1], cut[::-1]
    positions, levels, contrasts = pf._measure_contrast(pixels, cut, center=0, n=3, window=11)
    assert positions[0] < positions[-1]
    assert contrasts[0] == 1
    assert np.all(np.diff(contrasts) < 0)


def test_format_expnums_inclusive():
    assert qa._format_expnums(["35921, 35951", 36284]) == "35921–35950, 36284"
    assert qa._format_expnums(None) == "–"


def test_describe_sequence_roles(monkeypatch):
    monkeypatch.setattr(qa, "_find_raw", lambda sources, camera, expnum: None if expnum == 104 else f"{expnum}.fits")
    epoch = {"sources": [60000], "sequences": {"b1": {"kind": "2f1d", "expnums": ["100, 108"], "rejects": [106]}}}
    description = qa.describe_sequence(epoch, "b1")
    exposures = description["exposures"].set_index("expnum")
    assert description["kind_text"] == "2 flats + 1 dark"
    assert (description["ngroups"], description["nleftover"]) == (2, 1)
    assert exposures.role.to_dict() == {100: "flat", 101: "flat", 102: "dark", 103: "flat", 104: "flat",
                                        105: "dark", 106: "rejected", 107: "unassigned"}
    assert pd.isna(exposures.path[104])
    assert exposures.path[100] == "100.fits"


def test_get_ivar_ignores_invalid_errors():
    image = Image(data=np.ones(4), error=np.array([0.5, 0.0, np.inf, np.nan]))
    assert image.get_ivar().tolist() == [4.0, 0.0, 0.0, 0.0]


def test_pixflat_qa_dir_matches_ancillary_tree():
    anc_path = path.full("lvm_anc", drpver=qa.drpver, tileid=11111, mjd=60741, kind="p",
                         imagetype="pixflat_qa", expnum=0, camera="b1")
    assert qa._qa_dir(60741, "60741_raw") == os.path.join(os.path.dirname(anc_path), "pixflat_qa", "60741_raw")
