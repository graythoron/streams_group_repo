from matplotlib import pyplot as plt, rcParams, lines as mlines
import matplotlib.colors as colors
import matplotlib.cm as cm
import cmasher as cmr
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


plt.rc('font', size=18)
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18

def plot_pretty(dpi=175, fontsize=15, labelsize=15, figsize=(10, 8), tex=True):
    # import pyplot and set some parameters to make plots prettier

    plt.rc('savefig', dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('font', family='serif')
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    plt.rc('figure', figsize=figsize)
    try:
        plt.rc('text', usetex=tex)
    except:
        plt.rc('mathtext', fontset="cm")

    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize
    rcParams.update({'figure.autolayout': True})


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)