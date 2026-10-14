# encoding: utf-8
#

import pandas as pd
import pytest

from lvmdrp.functions import run_calseq as rc


def make_dome_frames(mjd=60255):
    """Camera frames of a dome sequence: one LDLS and one quartz exposure per standard fiber of the primary ring"""
    standards = rc._get_standards_ring("primary")
    rows = []
    for i, calibfib in enumerate(standards):
        for lamp, offset in (("ldls", 0), ("quartz", 100)):
            expnum = 1000 + offset + i
            for camera in rc.CAMERAS:
                rows.append({"mjd": mjd, "tileid": 11111, "expnum": expnum, "camera": camera, "imagetyp": "flat",
                             "qaqual": "GOOD", "calibfib": calibfib, "ldls": lamp == "ldls", "quartz": lamp == "quartz",
                             "neon": False, "hgne": False, "argon": False, "xenon": False})
    return pd.DataFrame(rows)


def test_choose_dome_sequence():
    frames = make_dome_frames()
    nstandards = len(rc._get_standards_ring("primary"))

    chosen = rc.choose_dome_sequence(frames, channels="brz", kind="longterm")
    # each channel only keeps its cameras in the exposures of its lamp
    for channel, lamp in rc.MASTER_CON_LAMPS.items():
        sel = chosen.loc[chosen.camera.str.startswith(channel)]
        assert sel[lamp].all() and sel.expnum.nunique() == nstandards
        assert sorted(sel.camera.unique()) == [f"{channel}{i}" for i in "123"]
        assert sorted(sel.calibfib.unique()) == sorted(rc._get_standards_ring("primary"))

    # nightly sequences take the first two exposures of each lamp
    nightly = rc.choose_dome_sequence(frames, channels="bz", kind="nightly")
    assert sorted(nightly.loc[nightly.camera.str[0] == "b"].expnum.unique()) == [1000, 1001]
    assert sorted(nightly.loc[nightly.camera.str[0] == "z"].expnum.unique()) == [1100, 1101]
    assert set(nightly.camera.str[0]) == {"b", "z"}

    # without selection all the exposures of the lamp are kept
    everything = rc.choose_dome_sequence(frames, channels="r", kind=None)
    assert len(everything) == 3 * nstandards

    # a missing lamp leaves its channel out
    no_quartz = rc.choose_dome_sequence(frames.loc[~frames.quartz], channels="brz", kind="longterm")
    assert set(no_quartz.camera.str[0]) == {"b", "r"}
    assert rc.choose_dome_sequence(frames.loc[~frames.quartz], channels="z").empty


def test_create_dome_fiberflats(monkeypatch, tmp_path):
    frames = make_dome_frames()
    calls = {"reduce_2d": [], "fit": [], "combine": []}

    monkeypatch.setattr(rc.md, "get_calibrations_metadata", lambda mjds, calibration: frames.copy())
    monkeypatch.setattr(rc, "get_calib_paths", lambda **kwargs: {})
    monkeypatch.setattr(rc, "reduce_2d", lambda **kwargs: calls["reduce_2d"].append(kwargs))

    def fake_fit(sequence, channel, mjd, calibs, product, **kwargs):
        calls["fit"].append((channel, product, sorted(sequence.expnum.unique()), kwargs["cnorm"]))
        expnums = sorted(sequence.expnum.unique())
        return ([f"x{e}" for e in expnums], [f"f{e}" for e in expnums], [f"{product}-{channel}-{e}" for e in expnums])
    monkeypatch.setattr(rc, "_fit_exposure_fiberflats", fake_fit)
    monkeypatch.setattr(rc, "_combine_fiberflat_sequence", lambda *args: calls["combine"].append(args))

    epochs = {60255: {"flavors": {"dome": "60255, 60256"}}}
    rc.create_dome_fiberflats(mjd=60255, epochs=epochs, channels="bz", kind="longterm")

    # 2D reduction of the cameras of each channel, with stray light and without cosmic rays rejection
    assert [sorted({c[0] for c in call["cameras"]}) for call in calls["reduce_2d"]] == [["b"], ["z"]]
    assert all(call["sub_straylight"] and not call["reject_cr"] for call in calls["reduce_2d"])
    # per-exposure flats of each channel from the exposures of its lamp, at the same wavelengths as the twilights
    (b_channel, b_product, b_expnums, b_cnorm), (z_channel, _, z_expnums, z_cnorm) = calls["fit"]
    assert (b_channel, b_product, z_channel) == ("b", "DFlat", "z")
    assert all(e < 1100 for e in b_expnums) and all(e >= 1100 for e in z_expnums)
    assert (b_cnorm, z_cnorm) == (rc.SKYLINES_FIBERFLAT["b"], rc.SKYLINES_FIBERFLAT["z"])
    # combined into the long-term master dome fiber flats
    assert [args[0] for args in calls["combine"]] == ["b", "z"]
    assert all("mfiberflat_dome" in args[-1] for args in calls["combine"])

    # without combination, only the per-exposure flats are made
    calls["combine"].clear()
    rc.create_dome_fiberflats(mjd=60255, epochs=epochs, channels="r", skip_combination=True)
    assert calls["combine"] == []

    # nightly sequences make the nightly master
    rc.create_dome_fiberflats(mjd=60255, epochs=epochs, channels="r", kind="nightly")
    assert "nfiberflat_dome" in calls["combine"][-1][-1]

    with pytest.raises(ValueError):
        rc.create_dome_fiberflats(mjd=60255, channels="x")


def test_create_dome_fiberflats_without_frames(monkeypatch):
    monkeypatch.setattr(rc.md, "get_calibrations_metadata", lambda mjds, calibration: make_dome_frames().iloc[:0])
    monkeypatch.setattr(rc, "reduce_2d", lambda **kwargs: pytest.fail("nothing to reduce"))
    rc.create_dome_fiberflats(mjd=60255)
