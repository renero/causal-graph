#
# This file includes all the plot methods for the causal graph
#
# (C) J. Renero, 2022, 2023
#

from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


def setup_plot(tex=True, font="serif", dpi=100):
    """Customize figure settings.

    Args:
        tex (bool, optional): use LaTeX. Defaults to True.
        font (str, optional): font type. Defaults to "serif".
        dpi (int, optional): dots per inch. Defaults to 180.
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": font,
            "text.usetex": tex,
            "figure.subplot.top": 0.9,
            "figure.subplot.right": 0.9,
            "figure.subplot.left": 0.15,
            "figure.subplot.bottom": 0.12,
            "figure.subplot.hspace": 0.4,
            "figure.dpi": dpi,
            "savefig.dpi": 180,
            "savefig.format": "pdf",
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "axes.axisbelow": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "xtick.minor.size": 2.25,
            "xtick.major.pad": 7.5,
            "xtick.minor.pad": 7.5,
            "ytick.major.pad": 7.5,
            "ytick.minor.pad": 7.5,
            "ytick.major.size": 5,
            "ytick.minor.size": 2.25,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.framealpha": 1,
            "figure.titlesize": 12,
            "lines.linewidth": 2,
        }
    )


def add_grid(ax, lines=True, locations=None):
    """Add a grid to the current plot.

    Args:
        ax (Axis): axis object in which to draw the grid.
        lines (bool, optional): add lines to the grid. Defaults to True.
        locations (tuple, optional):
            (xminor, xmajor, yminor, ymajor). Defaults to None.
    """

    if lines:
        ax.grid(lines, alpha=0.5, which="minor", ls=":")
        ax.grid(lines, alpha=0.7, which="major")

    if locations is not None:

        assert (
            len(locations) == 4
        ), "Invalid entry for the locations of the markers"

        xmin, xmaj, ymin, ymaj = locations

        ax.xaxis.set_minor_locator(MultipleLocator(xmin))
        ax.xaxis.set_major_locator(MultipleLocator(xmaj))
        ax.yaxis.set_minor_locator(MultipleLocator(ymin))
        ax.yaxis.set_major_locator(MultipleLocator(ymaj))


def subplots(
        num_cols: int, 
        plot_func: Callable, 
        *plot_args: Any,
        fig_size: Tuple[int, int] = (10, 6),
        title: str = None,
        **kwargs: Any) -> None:
    """
    Plots a set of subplots.
    
    Arguments:
    ----------
        num_cols: int
            The number of columns in the subplot grid.
        plot_func: function
            The function to be used to plot each subplot.
        *plot_args: List
            The arguments to be passed to the plot function.
        fig_size: Tuple[int, int]
            The size of the figure.
        title: str
            The title of the figure.
        **kwargs: Dict
            Additional arguments to be passed to the plot function.

    Returns:
    --------
        fig: Figure
            The figure containing the subplots.
    """
    setup_plot(**kwargs)
    num_rows = len(plot_args) // num_cols
    if len(plot_args) % num_cols != 0:
        num_rows += 1

    def blank(ax):
        npArray = np.array([[[255, 255, 255, 255]]], dtype="uint8")
        ax.imshow(npArray, interpolation="nearest")
        ax.set_axis_off()

    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < len(plot_args):
                ax = axes[i][j]
                plot_func(plot_args[index], ax=ax)
            else:
                blank(axes[i][j])
                
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_shap_summary(
        feature_names: List[str],
        mean_shap_values, 
        feature_inds, 
        selected_features, 
        ax,
        **kwargs):
    """
    Plots the summary of the SHAP values for a given target.

    Arguments:
    ----------
        feature_names: List[str]
            The names of the features.
        mean_shap_values: np.array
            The mean SHAP values for each feature.
        feature_inds: np.array
            The indices of the features to be plotted.
        selected_features: List[str]
            The names of the selected features.
        ax: Axis
            The axis in which to plot the summary. If None, a new figure is created.
        **kwargs: Dict
            Additional arguments to be passed to the plot.
    """

    figsize_ = kwargs.get('figsize', (6, 3))
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_)

    y_pos = np.arange(len(feature_inds))
    ax.grid(True, axis='x')
    ax.barh(y_pos, mean_shap_values[feature_inds],
            0.7, align='center', color="#0e73fa", alpha=0.8)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    ax.set_yticks(y_pos, fontsize=13)
    ax.set_yticklabels([feature_names[i] for i in feature_inds])
    ax.set_xlabel("$\\frac{1}{m}\sum_{j=1}^p| \phi_j |$")
    # ax.set_title(target_name + " $\leftarrow$ " +
    #                 (','.join(selected_features) if len(selected_features) != 0 else 'ø'))
    fig = ax.figure if fig is None else fig
    return fig
