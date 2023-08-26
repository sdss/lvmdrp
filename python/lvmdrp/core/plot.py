# encoding: utf-8
#
# @Author: Alfredo Mejia-Narvaez
# @Date: Mar 21, 2023
# @Filename: plot.py
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings

warnings.filterwarnings(
    action="ignore",
    module="matplotlib.figure",
    category=UserWarning,
    message=(
        "This figure includes Axes that are not compatible with tight_layout, "
        "so results might be incorrect."
    ),
)

IS_INTERACTIVE = hasattr(sys, "ps1")
DEFAULT_BACKEND = plt.get_backend()
if not IS_INTERACTIVE:
    plt.switch_backend(newbackend="Agg")
plt.style.use("seaborn-v0_8-talk")


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
    fig, axs = plt.subplots(**subplots_params)
    if flatten_axes and isinstance(axs, np.ndarray):
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


def plot_wavesol_residuals(lines_pixels, lines_waves, model_waves, ax=None, labels=False):
    # lines_pixels [X, Y] of emission lines in a given lamp
    # wavelength model poly_model(lines_centroids)
    # lines_waves is an X array of the true wavelengths

    residuals = model_waves - lines_waves
    
    ax.axhline(ls="--", lw=1, color="0.8")
    ax.scatter(lines_pixels, residuals, s=10)
    for i in range(residuals.size):
        x, y = lines_pixels[i], residuals[i]
        ax.annotate(f"{lines_waves[i]:.2f}", (x, y), xytext=(9, -9),
                textcoords="offset pixels")
    
    if labels:
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Residuals (angstrom)")
    
    return ax


def plot_wavesol_coeffs(ypix, coeffs, axs, title=None, labels=False, to_display=False):
    # ypix is the Y coordinate of each fiber in the middle of the chip
    # coeffs is a 2D array of coefficients for each fiber [nfiber, ncoeff]

    for icoeff in range(coeffs.shape[1]):
        axs[icoeff].scatter(ypix, coeffs[:, icoeff], color="tab:blue")
        if labels:
            axs[icoeff].set_title(f"coeff # {icoeff+1}", loc="left")
    
    fig = axs[0].get_figure()
    if labels:
        fig.supxlabel("Fiber ID")
    if title is not None:
        fig.suptitle(title)
    
    return axs

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
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

    return fig_path
