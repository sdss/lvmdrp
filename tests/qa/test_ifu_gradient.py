# encoding: utf-8
#

import numpy as np
import pandas as pd
import pytest

import lvmdrp.qa.ifu_gradient as ig


def _vector(strength, angle):
    return strength * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])


def _gradients(source="twilight", instrument=((3.0, 45.0), (6.0, 60.0)), break_mjd=60465, sky=1.5, slope=0.1, n_nights=20,
               per_night=8, phi=None, rotation=0, noise=0.05, seed=3, channel="b"):
    """Gradient rows of one channel: an instrument gradient that jumps at `break_mjd` plus a sky gradient

    For twilights the sky gradient points towards the Sun with an amplitude linear in the depression of the
    Sun; for science frames it points towards the horizon with a random amplitude per exposure. With
    `rotation`, the instrument gradient rotates with the K-mirror angle `phi` (random per exposure if None).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for night in range(n_nights):
        mjd = 60300 + 20 * night
        base = rng.uniform(-180, 180)
        for k in range(per_night):
            sun_alt = -3 - 1.0 * k
            direction = base + rng.normal(0, 20)
            angle = rng.uniform(-180, 180) if phi is None else phi
            strength, inst_dir = instrument[0] if mjd < break_mjd else instrument[-1]
            g = _vector(strength, inst_dir + rotation * angle)
            if source == "twilight":
                g = g + _vector(sky + slope * (-3 - sun_alt), direction)
            else:
                g = g + _vector(rng.uniform(0, 3), direction)
            g = g + rng.normal(0, noise, 2)
            rows.append({"channel": channel, "mjd": mjd, "expnum": 1000 * night + k, "grad_x": g[0] / 100, "grad_y": g[1] / 100,
                         "dir_sun": direction, "dir_horizon": direction, "dir_moon": rng.uniform(-180, 180), "sun_alt": sun_alt,
                         "pa": angle, "field": 0.0})
    return ig.prepare_gradients(pd.DataFrame(rows), source)


def test_prepare_gradients():
    data = _gradients()
    assert {"gx", "gy", "depth", "phi", "source"} <= set(data.columns)
    assert np.allclose(data["depth"], -3 - data["sun_alt"])
    # PA - field_sign x field, with the LCO sign -1 and a zero field angle
    assert np.allclose(data["phi"], ig._wrap_angle(data["pa"]))


def test_intervention_breaks():
    epochs = {60255: {"trigger": "Survey start"}, 60339: {"trigger": "Instrument intervention"},
              60446: {"trigger": "Warm-up"}, 60511: {"trigger": "Bad weather"}}
    assert ig.intervention_breaks(epochs) == [60339, 60446]


def test_fit_recovers_instrument_and_sky():
    data = _gradients()
    fit = ig.fit_gradient_model(data, frame="ifu", breaks=[60465])
    inst = fit["instrument"]
    assert len(inst) == 2
    assert inst["strength"].to_numpy() == pytest.approx([3.0, 6.0], abs=0.05)
    assert inst["direction"].to_numpy() == pytest.approx([45.0, 60.0], abs=1.0)
    sky = fit["sky"].set_index("amplitude")["value"]
    assert sky["depth"] == pytest.approx(1.5, abs=0.05) and sky["slope"] == pytest.approx(0.1, abs=0.02)
    assert fit["rms"] < 0.1 and fit["cv_rms"] < 0.12
    assert (inst[["gx_err", "gy_err"]].to_numpy() < 0.1).all()
    # the residual gradients point to the Sun once the instrument gradient is removed
    alignment = ig.residual_alignment(fit).set_index(["gradient", "frame"])
    assert alignment.loc[("g - I", "Sun"), "R"] > 0.95
    assert alignment.loc[("g", "IFU"), "R"] > alignment.loc[("g - I", "IFU"), "R"]


def test_compare_models_prefers_the_true_one():
    data = _gradients()
    table, fits = ig.compare_gradient_models(data, breaks={"none": [], "interventions": [60465]})
    best = table.iloc[0]
    assert best["frame"] in ("ifu", "kmirror", "kmirror-") and best["breaks"] == "interventions"
    assert not table.loc[table["frame"] == "ifu", "degenerate"].any()
    # random K-mirror angles: the K-mirror frames are identifiable and worse than the IFU
    assert best["frame"] == "ifu"
    assert table.loc[table["frame"] == "none", "dbic"].min() > 100


def test_kmirror_frame():
    # instrument gradient rotating with the K-mirror: the K-mirror frame wins
    data = _gradients(rotation=1, instrument=((4.0, 30.0),))
    table, _ = ig.compare_gradient_models(data)
    assert table.iloc[0]["frame"] == "kmirror"
    # a parked K-mirror: the K-mirror frames are degenerate with the IFU
    data = _gradients(phi=88.0)
    table, _ = ig.compare_gradient_models(data, frames=("ifu", "kmirror"))
    assert table.set_index("frame").loc["kmirror", "degenerate"]
    assert not table.set_index("frame").loc["ifu", "degenerate"]


def test_nightly_instrument():
    data = _gradients()
    fit = ig.fit_gradient_model(data, frame="ifu", breaks="nights")
    inst = fit["instrument"]
    assert len(inst) == data["mjd"].nunique()
    expected = np.where(inst["mjd_first"] < 60465, 3.0, 6.0)
    assert inst["strength"].to_numpy() == pytest.approx(expected, abs=0.1)


def test_sky_direction_scan():
    data = _gradients(n_nights=10)
    scan = ig.sky_direction_scan(data, breaks=[60465], offsets=range(-90, 91, 30))
    best = scan.loc[scan["rms_inst"].idxmin()]
    assert best["offset"] == 0 and not best["mirrored"]
    assert scan["rms_sky"].min() > 5 * best["rms_inst"]


def test_cross_check_with_science():
    twilight = _gradients()
    science = _gradients(source="science", seed=5)
    fit = ig.fit_gradient_model(twilight, frame="ifu", breaks=[60465])
    check = ig.cross_check(science, fit, {"dir_horizon": "exposure"})
    assert check["n"] == len(science)
    assert check["sky_corrected"] < 0.1 < check["sky_raw"]
    alignment = check["alignment"].set_index(["gradient", "frame"])
    assert alignment.loc[("g", "Horizon"), "R"] > 0.95


def test_qa_instrument_gradient(tmp_path):
    twilight = _gradients()
    science = _gradients(source="science", seed=5, rotation=0)
    twilight.to_csv(tmp_path / "twilight.csv", index=False)
    result = ig.qa_instrument_gradient(twilight=str(tmp_path / "twilight.csv"), science=science, breaks=[60465], channels="b",
                                       output_dir=str(tmp_path / "out"))
    assert set(result["comparison"]["group"]) == {"twilight", "science", "joint"}
    joint = result["instrument"].query("group == 'joint'")
    assert joint["strength"].to_numpy() == pytest.approx([3.0, 6.0], abs=0.1)
    assert ("science", "b") in result["cross_checks"]
    html = open(result["report"]).read()
    assert "LVM Instrument Gradient" in html and "vectors|twilight|b" in html
    assert (tmp_path / "out" / "instrument_gradient_models.csv").exists()


def test_choose_examples_spans_the_sky_direction():
    data = _gradients()
    data["exptime"] = np.where(data["expnum"] % 1000 < 3, 2.0, 30.0)
    fit = ig.fit_gradient_model(data, frame="ifu", breaks=[60465], crossval=False)
    examples = ig.choose_examples(fit, n=4)
    assert len(examples) == 4 and (examples["exptime"] >= 9).all()
    relative = ig._wrap_angle(examples["dir_sun"] - np.degrees(np.arctan2(examples["inst_y"], examples["inst_x"])))
    assert np.ptp(relative) > 180


def test_alignment_rose_and_ifu_examples():
    data = _gradients()
    fit = ig.fit_gradient_model(data, frame="ifu", breaks=[60465], crossval=False)
    rose = ig.figure_alignment_rose(fit)
    # the measured gradients spread, the gradients minus the instrument model point to the Sun (bin centered on 0 deg)
    sun = {t.name: np.asarray(t.r) for t in rose.data if t.subplot == "polar"}
    assert sun["g − I"].max() > 0.8 and sun["g"].max() < 0.5
    x, y = ig.ifu_fiber_positions()
    assert np.isclose(np.max(np.hypot(x, y)), 1.0)
    fig = ig.figure_ifu_examples(fit, (x, y), n=3)
    maps = [t for t in fig.data if t.mode == "markers"]
    assert len(maps) == 9
    # the last panel of each row is the first minus the second
    for row in range(3):
        g, inst, res = (np.asarray(maps[3 * row + c].marker.color) for c in range(3))
        assert np.allclose(res, g - inst)


def test_calibration_epoch_of():
    assert ig._mjd_list("60469, 60477") == [60469, 60477] and ig._mjd_list(60339) == [60339] and ig._mjd_list("60177-60179") == [60177, 60178, 60179]
    epochs = {60446: {"flavors": {"twilight": "60448, 60450"}}, 60465: {"flavors": {"twilight": "60469, 60477"}}, 60511: {"flavors": {"twilight": 60514}}}
    # listed nights go to their epoch, even after the next epoch starts; others to the last epoch before them
    assert ig.calibration_epoch_of([60448, 60469, 60477, 60490, 60514, 60520, 60400], epochs) == \
        {60400: 60446, 60448: 60446, 60469: 60465, 60477: 60465, 60490: 60465, 60514: 60511, 60520: 60511}


def test_epoch_browser_payload():
    import json

    data = _gradients()
    fit = ig.fit_gradient_model(data, frame="ifu", breaks=[60465], crossval=False)
    epochs = {60300: {"trigger": "Survey start", "flavors": {"twilight": "60300, 60320"}}, 60465: {"trigger": "Instrument intervention", "flavors": {}}}
    html = ig.epoch_browser({("twilight", "b"): fit, ("science", "b"): fit}, ig.ifu_fiber_positions(), epochs)
    payload = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert list(payload["sets"]) == ["twilight|b"] and payload["sets"]["twilight|b"]["reference"] == "sun"
    rows = payload["sets"]["twilight|b"]["rows"]
    assert len(rows) == len(fit["data"]) and {r[2] for r in rows} == {"60300", "60465"}
    assert all(r[2] == ("60300" if r[1] < 60465 else "60465") for r in rows)
    assert "Instrument intervention" in payload["epochs"]["60465"] and "</script>" in html
    # without epochs, the exposures are grouped by the segments of the fit
    html = ig.epoch_browser({("twilight", "b"): fit}, ig.ifu_fiber_positions())
    payload = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert {r[2] for r in payload["sets"]["twilight|b"]["rows"]} == {"60300", "60480"}
    assert ig.epoch_browser({("science", "b"): fit}, ig.ifu_fiber_positions()) == ""
