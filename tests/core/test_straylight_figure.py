# encoding: utf-8
#
# test_straylight_figure.py
#
# The stray-light QA figure (Image.fit_straylight with axes) is drawn without changing the model

import numpy as np
from astropy.io import fits

from lvmdrp.core.image import Image
from lvmdrp.core.plot import create_straylight_axes, plt


def _frame():
    rng = np.random.default_rng(3)
    ny, nx = 120, 160
    yy, xx = np.mgrid[:ny, :nx]
    data = 5 + 2 * np.sin(xx / 40.0) + yy / 60.0 + rng.normal(0, 0.3, (ny, nx))
    error = np.full((ny, nx), 0.3)
    error[60, 10:13] = np.inf                  # masked pixels with infinite error, as in real frames
    mask = np.zeros((ny, nx), dtype=bool)
    header = fits.Header({'BUNIT': 'electron', 'FILENAME': 'test.fits'})
    img = Image(data=data, error=error, mask=mask, header=header)
    strips = []
    for y0 in range(5, ny, 20):
        s = np.zeros((ny, nx), dtype=bool)
        s[y0:y0 + 3] = True
        strips.append(s)
    return img, strips


def test_straylight_figure_does_not_change_model():
    kw = dict(n_knots_x=8, n_knots_y=8, lam_x=10, lam_y=10, nsigma=3.0, niter=2)
    img, strips = _frame()
    model = img.fit_straylight(strips, axs=None, **kw)
    fig, axs = create_straylight_axes(img, len(strips))
    try:
        model_qa = img.fit_straylight(strips, axs=axs, **kw)
        fig.canvas.draw()
    finally:
        plt.close(fig)
    assert np.array_equal(model._data, model_qa._data)
    # the per-column error band is drawn in each strip panel
    assert all(len(ax.collections) >= 1 for ax in axs['res'])
