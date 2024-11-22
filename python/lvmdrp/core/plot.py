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


def plot_wavesol_residuals(fiber, ref_waves, lines_pixels, poly_cls, coeffs, ax=None, labels=False):
    """Plot residuals in wavelength polynomial fitting

    Parameters
    ----------
    fiber : int
        Reference fiber used in the fitting
    ref_waves : np.ndarray[float]
        Measured line centroids for all fibers
    lines_pixels : np.ndarray[float]
        Reference lines positions
    poly_cls : polynomial class in np.polynomial
        A polynomial class to evaluate wavelength solutions
    coeffs : np.ndarray[float]
        Coefficients to evaluate
    ax : plt.Axes, optional
        Axes where to plot, by default None
    labels : bool, optional
        Whether to draw labels or not, by default False

    Returns
    -------
    ax : plt.Axes
        Axes with plot in it
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(15,5), layout="constrained")

    colors = plt.cm.coolwarm(np.linspace(0, 1, lines_pixels.shape[0]))
    residuals = np.zeros((lines_pixels.shape[0], ref_waves.size))
    for ifiber in range(lines_pixels.shape[0]):
        residuals[ifiber] = poly_cls(coeffs[ifiber])(lines_pixels[ifiber]) - ref_waves
        if ifiber == fiber or (coeffs[ifiber] == 0).all():
            continue
        ax.plot(lines_pixels[ifiber], residuals[ifiber], ".", color=colors[ifiber], alpha=0.2)
    ax.plot(lines_pixels[fiber], residuals[fiber], "o", mec="k", mfc="none", ms=5, mew=1)
    ax.axhline(ls="--", lw=1, color="0.7")
    ax.axhline(-0.05, ls=":", lw=1, color="0.5")
    ax.axhline(+0.05, ls=":", lw=1, color="0.5")
    for i in range(ref_waves.size):
        x, y = lines_pixels[fiber, i], residuals[fiber, i]
        ax.annotate(f"{ref_waves[i]:.2f}", (x, y), xytext=(9, -9), textcoords="offset pixels")
    ax.set_ylim(-0.1, +0.1)

    if labels:
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Residuals (angstrom)")

    return ax


def plot_wavesol_spec(fiber, ref_pixels, aperture, mhat, bhat, arc, ax=None, log_scale=True, labels=False):
    """Display arc spectrum of given fiber along with the measured reference lines

    Parameters
    ----------
    fiber : int
        Fiber for which the arc spectrum will be displayed
    ref_pixels : np.ndarray[float]
        Pixel positions of reference arc lines
    aperture : float
        Size of the window around which the arc lines were fitted
    {mhat, bhat} : float
        Parameters of the linear (mhat*x + bhat) correction applied to the reference arc lines positions
    arc : lvmdrp.core.rss.RSS
        RSS of the arc frame
    ax : plt.Axes, optional
        Axes where to put the plot, by default None
    log_scale : bool, optional
        Whether to display Y axis in log-scale or not, by default True
    labels : bool, optional
        Whether to draw labels for the axes

    Returns
    -------
    ax : plt.Axes
        Axes containing the plot
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(15,5), layout="constrained")

    unit = arc._header["BUNIT"]
    good_pix = ~arc._mask
    for pix in ref_pixels:
        ax.axvspan(
            pix - (aperture - 1) // 2,
            pix + (aperture - 1) // 2,
            np.nanmin((arc._data * good_pix)[fiber]),
            np.nanmax((arc._data * good_pix)[fiber]),
            fc="0.7",
            alpha=0.5,
        )
    ax.vlines(
        (ref_pixels - bhat) / mhat,
        np.nanmin((arc._data * good_pix)[fiber]),
        np.nanmax((arc._data * good_pix)[fiber]),
        color="tab:red",
        lw=0.5,
        label="orig. ref. lines",
    )
    ax.vlines(
        ref_pixels,
        np.nanmin((arc._data * good_pix)[fiber]),
        np.nanmax((arc._data * good_pix)[fiber]),
        color="tab:blue",
        lw=0.5,
        label=f"corr. lines ({mhat = :.2f}, {bhat = :.2f})",
    )
    ax.step(arc._pixels, (arc._data * good_pix)[fiber], color="0.2", lw=1)
    if labels:
        ax.set_title(f"reference arc spectrum {fiber}", loc="left")
        ax.set_ylabel(f"count ({unit})")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(loc=1)

    return ax


def plot_wavesol_coeffs(ypix, coeffs, axs, color="tab:blue", labels=False):
    """Display plot of polynomial fitting coefficients against fiber positions/indices

    Parameters
    ----------
    ypix : np.ndarray[float|int]
        Fiber positions or indices
    coeffs : np.ndarray[float]
        Coefficients of the fitted polynomials for each fiber
    axs : np.ndarray[plt.Axes]
        Array of axes to plot each of the coefficients
    color : str, optional
        color of the points, by default 'tab:blue'
    labels : bool, optional
        Whether to draw labels for the axes or not, by default False

    Returns
    -------
    axs : np.ndarray[plt.Axes]
        Axes with the plots in them
    """
    if len(axs) != coeffs.shape[1]:
        raise ValueError(f"wrong number of axes {axs.size}, it does not match number of coefficients {coeffs.shape[1]}")

    for icoeff in range(coeffs.shape[1]):
        axs[icoeff].scatter(coeffs[:, icoeff], ypix, s=7, lw=0, color=color)
        if labels:
            axs[icoeff].set_title(f"coeff # {icoeff+1}", loc="left")

    if labels:
        axs[0].set_ylabel("Fiber ID")

    return axs


def plot_wavesol_wave(xpix, ref_waves, lines_pixels, wave_poly, wave_coeffs, ax=None, labels=False):
    """Display polynomial fitting without linear terms for the wavelength solutions

    Parameters
    ----------
    xpix : np.ndarray[float]
        X axis of the frame
    ref_waves : np.ndarray[float]
        Wavelengths of the reference arc lines
    lines_pixels : np.ndarray[float]
        Pixel positions of the measured arc lines
    wave_poly : np.polynomial class
        Polynomial class used to fit wavelengths
    wave_coeffs : np.ndarray[float]
        Coefficients of the polynomials for each fiber
    ax : plt.Axes, optional
        Axes where to draw the plot, by default None
    labels : bool, optional
        Whether to draw axes labels or not, by default False

    Returns
    -------
    ax : plt.Axes
        Axes containing the plot
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(15,5), layout="constrained")

    wave_sol = np.zeros((wave_coeffs.shape[0], xpix.size))
    wave_lin = np.zeros((wave_coeffs.shape[0], xpix.size))
    for i in np.arange(lines_pixels.shape[0]):
        if (wave_coeffs[i] == 0).all():
            continue

        wave_sol[i] = wave_poly(wave_coeffs[i])(xpix)
        wave_lin[i] = wave_poly(wave_coeffs[i]).truncate(2)(xpix)
        wave_lin_ref = wave_poly(wave_coeffs[i]).truncate(2)(lines_pixels)
        ax.plot(xpix, (wave_lin - wave_sol)[i], color="tab:blue", alpha=0.3, lw=1)

        ax.plot(
            lines_pixels[i],
            wave_lin_ref[i] - ref_waves,
            ".",
            ms=4,
            color="k",
            zorder=999
        )
    ax.plot(xpix, wave_lin.mean(0) - wave_sol.mean(0), lw=1, color="tab:blue")
    if labels:
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Wavelength model - linear terms (Angstrom)")
        ax.set_title(
            f"wavelength solutions with a {wave_coeffs.shape[1]-1}-deg polynomial",
            loc="left",
            color="tab:blue",
        )

    return ax


def plot_wavesol_lsf(xpix, lsf, lines_pixels, wave_poly, wave_coeffs, lsf_poly, lsf_coeffs, ax=None, labels=False):
    """Display polynomial fitting without linear terms for the LSF solutions

    Parameters
    ----------
    xpix : np.ndarray[float]
        X axis of the frame
    lsf : np.ndarray[float]
        FWHM of the measured arc lines for each fiber
    lines_pixels : np.ndarray[float]
        Pixel positions of the measured arc lines
    wave_poly : np.polynomial class
        Polynomial class used to fit wavelengths
    wave_coeffs : np.ndarray[float]
        Coefficients of the wavelength polynomials for each fiber
    lsf_poly : np.polynomial class
        Polynomial class used to fit LSF
    lsf_coeffs : np.ndarray[float]
        Coefficients of the LSF polynomials for each fiber
    ax : plt.Axes, optional
        Axes where to draw the plot, by default None
    labels : bool, optional
        Whether to draw axes labels or not, by default False

    Returns
    -------
    ax : plt.Axes
        Axes containing the plot
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(15,5), layout="constrained")

    dwave = np.zeros((wave_coeffs.shape[0], xpix.size))
    wave_sol = np.zeros((wave_coeffs.shape[0], xpix.size))
    lsf_sol = np.zeros((lsf_coeffs.shape[0], xpix.size))
    lsf_lin = np.zeros((lsf_coeffs.shape[0], xpix.size))
    for i in np.arange(lines_pixels.shape[0]):
        if (wave_coeffs[i] == 0).all():
            continue

        wave_sol[i] = wave_poly(wave_coeffs[i])(xpix)
        lsf_sol[i] = lsf_poly(lsf_coeffs[i])(xpix)
        lsf_lin[i] = lsf_poly(lsf_coeffs[i]).truncate(2)(xpix)
        lsf_lin_ref = lsf_poly(lsf_coeffs[i]).truncate(2)(lines_pixels)
        ax.plot(xpix, (lsf_lin - lsf_sol)[i], color="tab:red", alpha=0.3, lw=1)

        dwave[i] = np.abs(np.gradient(wave_sol[i]))
        dw = np.interp(lines_pixels[i], xpix, dwave[i])

        ax.plot(
            lines_pixels[i],
            lsf_lin_ref[i] - dw * lsf[i],
            ".",
            ms=4,
            color="k",
            zorder=999
        )
    ax.plot(xpix, lsf_lin.mean(0) - lsf_sol.mean(0), lw=1, color="tab:red")
    if labels:
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("LSF model - linear term (Angstrom)")
        ax.set_title(
            f"LSF solutions with a {lsf_coeffs.shape[1]-1}-deg polynomial",
            loc="left",
            color="tab:red",
        )

    return ax


def plot_fiber_thermal_shift(columns, column_shifts, median_shift, std_shift, ax=None, labels=False):
    """"Plots the thermal shifts measured in the fiber centroids"""
    if ax is None:
        _, ax = create_subplots(figsize=(15,5), layout="constrained")

    ax.plot(columns, column_shifts, "o-", color="tab:blue")
    ax.axhline(0, color="0.1", ls=":")
    ax.axhspan(median_shift-std_shift, median_shift+std_shift, color="tab:red", alpha=0.1)
    ax.axhline(median_shift, color="tab:red", lw=1, zorder=0)
    ax.set_title("Y shifts for each column")
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y shift (pixel)")
    ax.set_ylim(median_shift-4*std_shift, median_shift+4*std_shift)

    if labels:
        ax.annotate(f"mean: {median_shift:.2f}", (0.9, 0.9), xycoords="axes fraction", ha="right", va="top", color="tab:red")
        ax.annotate(f"std: {std_shift:.2f}", (0.9, 0.85), xycoords="axes fraction", ha="right", va="top", color="tab:red")

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
