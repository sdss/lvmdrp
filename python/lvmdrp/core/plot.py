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


plt.style.use("seaborn-v0_8-talk")


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

    return fig_path
