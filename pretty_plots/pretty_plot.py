from typing import List, Tuple

import cmasher as cmr
from matplotlib import pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

plt.rc("font", size=18)
rcParams["xtick.labelsize"] = 18
rcParams["ytick.labelsize"] = 18
cmr_name = cmr.__name__


def plot_settings(
    dpi: float = 175,
    fontsize: float = 20,
    labelsize: float = 20,
    figsize: List | Tuple = (10, 8),
    tex: bool = True,
):
    """
    Default settings for matplotlib plots
    Parameters
    ----------
    dpi : int or float
        Default : 175
    fontsize : int or float
        Default : 20
    labelsize : int or float
        Default : 20
    figsize : list or tuple
        Default : (10, 8)
    tex : Boolean
        Default : True
    """

    plt.rc("savefig", dpi=dpi)
    plt.rc("font", size=fontsize)
    plt.rc("font", family="serif")
    plt.rc("xtick.major", pad=5)
    plt.rc("xtick.minor", pad=5)
    plt.rc("ytick.major", pad=5)
    plt.rc("ytick.minor", pad=5)
    plt.rc("figure", figsize=figsize)
    if tex:
        plt.rc("mathtext", fontset="cm")

    rcParams["xtick.labelsize"] = labelsize
    rcParams["ytick.labelsize"] = labelsize
    rcParams.update({"figure.autolayout": True})


def colorbar(mappable):
    """
    Includes a colorbar.
    """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)
