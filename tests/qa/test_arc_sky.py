# encoding: utf-8
#
# tests of the dashboard comparing the arc LSF with sky line widths

import numpy as np

import lvmdrp.qa.arc_sky as qas
from lvmdrp.core.rss import lvmArc


def test_select_fibers():
    """ by default four fibers of the sky and of the science telescopes, spread along the slit; requested ones otherwise """
    fibers = np.arange(100)
    telescopes = np.where(fibers % 10 == 0, "SkyE", np.where(fibers % 10 == 5, "SkyW", "Sci"))
    sky = {"fibers": fibers, "telescopes": telescopes}

    slots = qas.select_fibers(sky)
    assert list(slots) == ["sky1", "sky2", "sky3", "sky4", "sci1", "sci2", "sci3", "sci4"]
    assert all(telescopes[slots[f"sky{i}"]] != "Sci" and telescopes[slots[f"sci{i}"]] == "Sci" for i in range(1, 5))
    assert [slots[f"sci{i}"] for i in range(1, 5)] == sorted(slots[f"sci{i}"] for i in range(1, 5))
    assert slots["sci4"] - slots["sci1"] > 50

    # requested fibers are kept when they belong to the group, others are skipped
    slots = qas.select_fibers(sky, {"sky": [10, 15, 11], "sci": [1, 2]})
    assert slots == {"sky1": 10, "sky2": 15, "sci1": 1, "sci2": 2}


def test_arc_profiles_per_exposure(measured_arcs, make_arc):
    """ one profile per arc exposure and fiber, none combined, peak normalized and centered on the fitted centroid """
    lvmarcs = [lvmArc.from_file(larc) for larc in measured_arcs.larcs]
    # a science fiber, lit in every exposure (standard fibers are lit in one)
    fiber = int(np.flatnonzero(np.asarray(lvmarcs[0]._slitmap["telescope"]) == "Sci")[10])
    line = make_arc.waves[make_arc.waves.size // 2]
    disp = np.full((lvmarcs[0]._fibers, lvmarcs[0]._data.shape[1]), 0.5)
    profiles = qas._arc_profiles(lvmarcs, line, fiber, disp)

    assert [expnum for expnum, *_ in profiles] == [1, 2]
    for _, dwave, values, sigma in profiles:
        assert np.abs(dwave).max() <= 0.5 * (qas.FIT["aperture"] // 2 + 1)
        assert 0.85 < values[np.argmin(np.abs(dwave))] < 1.1
        assert np.isclose(sigma, make_arc.fwhm * 0.5 / qas.GAUSS_FWHM, rtol=0.1)
