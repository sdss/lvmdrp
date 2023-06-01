# encoding: utf-8
#
# @Author: Alfredo Mejia-Narvaez
# @Date: Mar 21, 2023
# @Filename: plot.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from astropy.visualization import PercentileInterval, AsinhStretch, ImageNormalize

plt.style.use("seaborn-v0_8-talk")


def plot_image(
    image, ax, title=None, use_mask=True, labels=True, colorbar=True, extension="data"
):
    """plots the given image, optionally specifying the extension of the same

    Parameters
    ----------
    image : lvmdrp.core.image.Image
        image from which to make a plot
    ax : plt.Axes
        _description_
    title : str, optional
        title of the axes, by default None
    use_mask : bool, optional
        whether to use masked arrays or not, by default True
    labels : bool, optional
        whether to display axes labels or not, by default True
    colorbar : bool, optional
        whether to show colorbar or not, by default True
    extension : str, optional
        image extension to plot, by default "data"

    Returns
    -------
    plt.Axes
        matplotlib axes containing the image

    Raises
    ------
    ValueError
        if the `extension` is not valid
    """
    # pick the extension image to plot
    if extension == "data":
        data = (
            np.ma.masked_array(image._data, mask=image._mask)
            if use_mask
            else image._data
        )
    elif extension == "error":
        data = (
            np.ma.masked_array(image._error, mask=image._mask)
            if use_mask
            else image._error
        )
    elif extension == "mask":
        data = image._mask
    else:
        raise ValueError(
            f"invalid value for {extension = }. Choices are: 'data', 'error' and 'mask'"
        )

    norm = ImageNormalize(data, interval=PercentileInterval(95), stretch=AsinhStretch())
    im = ax.imshow(
        data,
        origin="lower",
        cmap="binary",
        norm=norm,
        interpolation="none",
        aspect="auto",
    )

    fig = ax.get_figure()
    if labels:
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
    if colorbar:
        axins = inset_axes(ax, width="20%", height="3%", loc="upper right")
        axins.xaxis.set_ticks_position("bottom")
        cb = fig.colorbar(im, cax=axins, orientation="horizontal")
        if labels:
            cb.set_label("counts (e-)", size="small")

    if title is not None:
        ax.set_title(title, loc="left")

    return ax


def plot_strips(
    image, axis, nstrip, ax, mu_stat=np.median, sg_stat=np.std, n_sg=1, labels=False
):
    """plots a number of strips of the image along a given direction

    given an image, a number of strips, and central and deviation statistics,
    this function extracts a strips and plots those statistics along the given
    axis.

    Parameters
    ----------
    image : lvmdrp.core.image.Image
        the image from which the strips will be extracted
    axis : int
        the axis along which the strips are going to be extracted
    nstrip : int
        number of strips
    ax : plt.Axes
        matplotlib axes in which to plot the strips
    mu_stat : function, optional
        the function to compute the central statistic, by default np.median
    sg_stat : function, optional
        the function to compute the deviation statistic, by default np.std
    n_sg : int, optional
        the number of deviations from the median to plot, by default 1
    labels : bool, optional
        whether to show or not the plot legend, by default False

    Returns
    -------
    plt.Axes
        the axes updated with the plotted strips
    """
    data = image._data
    width = (data.shape[0] if axis == 1 else data.shape[1]) // nstrip
    for i in range(nstrip):
        strip_mu = mu_stat(data[i * width : (i + 1) * width], axis=axis)
        strip_sg = sg_stat(data[i * width : (i + 1) * width], axis=axis)

        pixels = np.arange(strip_mu.size)
        ax.fill_between(
            pixels,
            strip_mu - n_sg * strip_sg,
            strip_mu + n_sg * strip_sg,
            step="post",
            lw=0,
            fc="tab:blue",
            alpha=0.5,
            label=f"{mu_stat}" if labels else None,
        )
        ax.step(
            pixels,
            strip_mu,
            where="post",
            color="tab:red",
            lw=1,
            label=f"{sg_stat}" if labels else None,
        )

    return ax


def save_fig(fig, output_path, figure_path=None, label=None, fmt="png", close=True):
    """Saves the given matplotlib figure to the given output/figure path"""
    # define figure path
    if figure_path is not None:
        fig_path = os.path.join(os.path.dirname(output_path), figure_path)
    else:
        fig_path = os.path.dirname(output_path)
    # create figure path if needed
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path, exist_ok=True)

    # define figure name
    fig_name = os.path.basename(output_path)
    if label is not None:
        fig_name = f"{fig_name.replace('.fits', '')}_{label}.{fmt}"
    else:
        fig_name = f"{fig_name.replace('.fits', '')}.{fmt}"

    # define figure full path
    fig_path = os.path.join(fig_path, fig_name)

    # save fig and close if requested
    fig.savefig(fig_path, bbox_inches="tight")
    if close:
        plt.close(fig)
    else:
        plt.show()

    return fig_path
