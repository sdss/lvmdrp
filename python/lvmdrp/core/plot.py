# encoding: utf-8
#
# @Author: Alfredo Mejia-Narvaez
# @Date: Mar 21, 2023
# @Filename: plot.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

DEFAULT_BACKEND = plt.get_backend()
BACKEND = "Agg"

plt.switch_backend(newbackend=BACKEND)
plt.style.use("seaborn-v0_8-talk")


def _switch_backend(event):
    """switch matplotlib backend to performance one `BACKEND`

    Since a figure would be open on display, the current backend cannot be
    `BACKEND`. Hence we need to switch back to the performance `BACKEND` in
    case a different call of plt.figure is triggered.

    This function ensures we are always in the performance `BACKEND`.

    Parameters
    ----------
    event : matplotlib event
        figure close event
    """
    plt.switch_backend(newbackend=BACKEND)


def create_subplots(to_display, flatten_axes=True, **subplots_params):
    """creates a figure and axes given a plt.subplots set of parameters

    Parameters
    ----------
    to_display : bool
        whether this figure will be displayed or not. This controls the backend used
    flatten_axes : bool, optional
        whether to flatten the axes array or not, by default True

    Returns
    -------
    plt.Figure
        created figure
    array_like
        create array of plt.Axes
    """
    if to_display:
        plt.switch_backend(newbackend=DEFAULT_BACKEND)
    fig, axs = plt.subplots(**subplots_params)
    fig.canvas.mpl_connect("close_event", _switch_backend)
    if flatten_axes:
        axs = axs.flatten()
    return fig, axs


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

    im = ax.imshow(
        data,
        origin="lower",
        cmap="binary_r",
        norm=ImageNormalize(
            data, interval=PercentileInterval(95), stretch=AsinhStretch()
        )
        if extension != "mask"
        else None,
        interpolation="none",
        aspect="auto",
    )

    fig = ax.get_figure()
    if labels:
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
    if colorbar and extension != "mask":
        axins = inset_axes(ax, width="50%", height="3%", loc="upper center")
        axins.xaxis.set_ticks_position("bottom")
        axins.tick_params(colors="tab:red", labelsize="small")
        cb = fig.colorbar(im, cax=axins, orientation="horizontal")
        if labels:
            unit = image._header["BUNIT"]
            cb.set_label(f"counts ({unit})", size="small", color="tab:red")

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


def save_fig(fig, product_path, to_display, figure_path=None, label=None, fmt="png"):
    """Saves the given matplotlib figure to the given output/figure path"""
    # define figure path
    if figure_path is not None:
        fig_path = os.path.join(os.path.dirname(product_path), figure_path)
    else:
        fig_path = os.path.dirname(product_path)
    # create figure path if needed
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path, exist_ok=True)

    # define figure name
    fig_name = os.path.basename(product_path)
    if label is not None:
        fig_name = f"{fig_name.replace('.fits', '')}_{label}.{fmt}"
    else:
        fig_name = f"{fig_name.replace('.fits', '')}.{fmt}"

    # define figure full path
    fig_path = os.path.join(fig_path, fig_name)

    # save fig and close if requested
    fig.savefig(fig_path, bbox_inches="tight")
    if to_display:
        plt.show()
    else:
        plt.close(fig)

    return fig_path
