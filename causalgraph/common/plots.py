#
# This file includes all the plot methods for the causal graph
#
# (C) J. Renero, 2022, 2023
#

from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from typing import List


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


def multiline_plot(values: Dict[str, Any], num_cols: int, func: Callable,
                   title="Multiplot", extra_args: Dict = {}, **kwargs):
    """
    Plots multiple plots in a multiline fashion. For each plot the method `func` is
    called to produce each individual plot. The dictionary `values` contains pairs
    where the key is the title of each plot, and value contains the numeric values to
    be plotted.

    Args:
        values: A dictionary where the key is the name of each plot, and the value
            is whatever your plot `func` will represent on each individual plot.
        num_cols: The number of columns in the grid
        func: A function that will accept an `ax`, `values`, `target_name` and
            `labels` for the X axis, and will plot that information.
        title: The sup_title of the plot.
        extra_args: A dictionary with extra arguments to be passed to the
            function that draws each individual plot

    Returns:
        None

    Examples:
        >>> def single_plot(ax, values, target_name, labels):
        >>>     ax.plot(values, marker='.')
        >>>     ax.set_xticks(range(len(labels)))
        >>>     ax.set_xticklabels(labels)
        >>>     ax.set_title(target_name)
        >>>
        >>> my_dict = {k:my_value[k] for k in features}
        >>> multiplot(my_dict, 5, single_plot, title="SHAP values",\
                figsize=(12, 5), sharey=True)

    """
    dpi = kwargs.get('dpi', 100)
    feature_names = list(values.keys())
    num_plots = len(feature_names)
    num_rows = int(num_plots / num_cols) + int(num_plots % num_cols != 0)

    setup_plot(dpi=dpi)
    fig, ax = plt.subplots(num_rows, num_cols, **kwargs)
    row, col = 0, 0

    def blank(ax):
        npArray = np.array([[[255, 255, 255, 255]]], dtype="uint8")
        ax.imshow(npArray, interpolation="nearest")
        ax.set_axis_off()

    for i in range(num_rows * num_cols):
        if (i % num_cols == 0) and (i != 0):
            row += 1
            col = 0
        if i < num_plots:  # empty image https://stackoverflow.com/a/30073252
            target_name = feature_names[i]
            labels = [f for f in feature_names if f != target_name]
            func(ax[row, col], values[target_name],
                 target_name, labels, **extra_args)
        else:
            blank(ax[row, col])
        col += 1

    plt.suptitle(title)
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

    figsize_ = kwargs.get('figsize', (10, 5))
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
    if fig is not None:
        plt.show()
