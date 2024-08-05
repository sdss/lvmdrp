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
import bottleneck as bn
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


def plot_image_shift(ax, image, column_shift, xpos=None, inset_pos=(0.0,1.0-0.32), inset_box=(0.3,0.3), cmap="gray"):
    """plots the pixel shifts of the given image along the given column shifts positions"""
    if xpos is None:
        xpos = image.shape[1]//2
    deltas = np.gradient(column_shift)
    irows = np.where(deltas > 0)[0][::2][::-1]
    axis = []
    for i, irow in enumerate(irows):
        iy, fy, ix, fx = xpos-30, xpos+30, irow-30, irow+30
        image_region = image[ix:fx, iy:fy]
        vmin = np.abs(np.nanmean(image_region)-3*np.nanstd(image_region))
        vmax = np.abs(np.nanmean(image_region)+3*np.nanstd(image_region))

        axi = ax.inset_axes((inset_pos[0],inset_pos[1]-i/3.2, *inset_box))
        axi.imshow(image_region, extent=[iy,fy,ix,fx], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axi.tick_params(axis="both", labelsize=10)
        if i == 0 and irows.size > 1:
            axi.tick_params(axis="both", labelbottom=False)
        else:
            axi.set_xlabel("X (pixel)", fontsize=10)
        axi.set_ylabel("Y (pixel)", fontsize=10)
        axis.append(axi)
    return axis


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
    if extension not in ["data", "error", "mask"]:
        raise ValueError(
            f"invalid value for {extension = }. Choices are: 'data', 'error' and 'mask'"
        )

    # pick the extension image to plot
    if extension == "data":
        data = image._data
    elif extension == "error" and image._error is not None:
        data = image._error
    elif extension == "mask" and image._mask is not None:
        data = image._mask
    else:
        ax.set_visible("off")
        return

    # mask data if requested
    if use_mask and extension != "mask" and image._mask is not None:
        data[image._mask] = np.nan

    # plot image
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

    # add colorbar and labels if requested
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
    image, axis, nstrip, ax, mu_stat=np.median, sg_stat=np.std, n_sg=1, show_individuals=False, labels=False
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
    show_individuals : bool, optional
        whether to show or not the individual strips, by default False
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
        if show_individuals:
            ylims = ax.get_ylim()
            ax.plot(
                pixels,
                data[i * width : (i + 1) * width],
                lw=0.5,
                color="0.5",
                alpha=0.1,
                zorder=-999
            )
            ax.set_ylim(ylims)

    return ax


def plot_detrend(ori_image, det_image, axs, mbias=None, mdark=None, labels=False):
    """plots the original and detrended images, optionally with bias and dark levels

    Parameters
    ----------
    ori_image : lvmdrp.core.image.Image
        original image
    det_image : lvmdrp.core.image.Image
        detrended image
    axs : array_like
        array of matplotlib axes in which to plot the images
    mbias : lvmdrp.core.image.Image, optional
        bias image, by default None
    mdark : lvmdrp.core.image.Image, optional
        dark image, by default None
    labels : bool, optional
        whether to show or not the plot 'x' and 'y' labels, by default False

    Returns
    -------
    array_like
        array of matplotlib axes updated with the plotted images

    Raises
    ------
    ValueError
        if the number of axes is not 4
    """

    # define quadrant sections
    sections = ori_image.getHdrValue("AMP? TRIMSEC")

    # convert all images to adu
    unit = "adu"
    ori_image_ = ori_image.convertUnit("adu", inplace=False)
    det_image_ = det_image.convertUnit("adu", inplace=False)
    det_image_.apply_pixelmask()
    if mbias is not None:
        mbias_ = mbias.convertUnit("adu", inplace=False)
    if mdark is not None:
        mdark_ = mdark.convertUnit("adu", inplace=False)

    # define counts range
    counts_range = (np.nanpercentile(ori_image_._data, q=0.1), np.nanpercentile(ori_image_._data, q=95.0))

    for i in range(4):
        # plot original image
        qori = ori_image_.getSection(sections[i])
        _ = axs[i].hist(qori._data.ravel(), bins=100, range=counts_range, fc="0.9", histtype="stepfilled", label="original")

        # plot bias and dark levels if requested
        if mbias_ is not None:
            qbias = mbias_.getSection(sections[i])
            bias_level = bn.nanmedian(qbias._data)
            _ = axs[i].axvline(bias_level, lw=1, ls="--", color="tab:red", label=f"bias level ({bias_level:.2f} {unit})")
        if mdark_ is not None:
            qdark = mdark_.getSection(sections[i])
            dark_level = bn.nanmedian(qdark._data)
            _ = axs[i].axvline(dark_level, lw=1, ls="--", color="tab:purple", label=f"dark level ({dark_level:.2f} {unit})")

        # plot detrended image
        qdet = det_image.getSection(sections[i])
        _ = axs[i].hist(qdet._data.ravel(), bins=100, range=counts_range, fc="none", lw=1, ec="tab:blue", histtype="step", label="detrended")

        axs[i].legend(loc=1, frameon=False)
        axs[i].ticklabel_format(axis='y', style='sci', scilimits=(4,4))

    # add labels if requested
    if labels:
        fig = axs[0].get_figure()
        fig.supxlabel(f"counts ({unit})")
        fig.supylabel("#")

    return axs


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


def plot_wavesol_coeffs(ypix, coeffs, axs, color="tab:blue", title=None, labels=False):
    # ypix is the Y coordinate of each fiber in the middle of the chip
    # coeffs is a 2D array of coefficients for each fiber [nfiber, ncoeff]

    for icoeff in range(coeffs.shape[1]):
        axs[icoeff].scatter(coeffs[:, icoeff], ypix, s=7, lw=0, color=color)
        if labels:
            axs[icoeff].set_title(f"coeff # {icoeff+1}", loc="left")

    fig = axs[0].get_figure()
    if labels:
        fig.supylabel("Fiber ID")
    if title is not None:
        fig.suptitle(title)

    return axs


def plot_fiber_thermal_shift(columns, column_shifts, ax=None, labels=False):
    """"Plots the thermal shifts measured in the fiber centroids"""
    if ax is None:
        fig, ax = create_subplots(figsize=(15,5))

    mean_shifts = np.nanmean(column_shifts)
    std_shifts = np.nanstd(column_shifts)

    ax.plot(columns, column_shifts, "o-", color="tab:blue")
    ax.axhline(0, color="0.1", ls=":")
    ax.axhspan(mean_shifts-std_shifts, mean_shifts+std_shifts, color="tab:red", alpha=0.1)
    ax.axhline(mean_shifts, color="tab:red", lw=1, zorder=0)
    ax.set_title("Y shifts for each column")
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y shift (pixel)")

    if labels:
        ax.annotate(f"mean: {mean_shifts:.2f}", (0.9, 0.9), xycoords="axes fraction", ha="right", va="top", color="tab:red")
        ax.annotate(f"std: {std_shifts:.2f}", (0.9, 0.85), xycoords="axes fraction", ha="right", va="top", color="tab:red")

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
    fig.savefig(fig_path)#, bbox_inches="tight")
    if to_display:
        # plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

    return fig_path
