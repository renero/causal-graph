#
# This file includes all the plot methods for the causal graph
#
# (C) J. Renero, 2022, 2023
#

from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

# Defaults for the graphs plotted
formatting_kwargs = {"node_size": 1000,
                     "node_color": "white",
                     "edgecolors": "black",
                     "font_family": "monospace",
                     "horizontalalignment": "center",
                     "verticalalignment": "center_baseline",
                     "with_labels": True
                     }


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


def plot_graph(
        dag: nx.DiGraph,
        reference: nx.DiGraph = None,
        names: List[str] = ["REX Prediction", "Ground truth"],
        figsize: Tuple[int, int] = (10, 5),
        dpi: int = 75,
        save_to_pdf: str = None,
        **kwargs):
    """
    Compare two graphs using dot.

    Parameters:
    -----------
    reference: The reference DAG.
    dag: The DAG to compare.
    names: The names of the reference graph and the dag.
    figsize: The size of the figure.
    **kwargs: Additional arguments to format the graphs:
        - "node_size": 500
        - "node_color": 'white'
        - "edgecolors": "black"
        - "font_family": "monospace"
        - "horizontalalignment": "center"
        - "verticalalignment": "center_baseline"
        - "with_labels": True
    """
    ncols = 1 if reference is None else 2

    # Overwrite formatting_kwargs with kwargs if they are provided
    formatting_kwargs.update(kwargs)

    G = nx.DiGraph()
    G.add_edges_from(dag.edges())
    if reference:
        Gt = _cleanup_graph(reference.copy())
        for missing in set(list(Gt.nodes)) - set(list(G.nodes)):
            G.add_node(missing)

        # Gt = _format_graph(Gt, Gt, inv_color="red", wrong_color="black")
        # G = _format_graph(G, Gt, inv_color="red", wrong_color="gray")
        Gt = _format_graph(Gt, G, inv_color="lightgreen", wrong_color="black")
        G = _format_graph(G, Gt, inv_color="orange", wrong_color="gray")
    else:
        G = _format_graph(G)

    ref_layout = None
    setup_plot(dpi=dpi)
    f, ax = plt.subplots(ncols=ncols, figsize=figsize)
    ax_graph = ax[1] if reference else ax
    if save_to_pdf is not None:
        with PdfPages(save_to_pdf) as pdf:
            if reference:
                ref_layout = nx.drawing.nx_agraph.graphviz_layout(
                    Gt, prog="dot")
                _draw_graph_subplot(Gt, layout=ref_layout, title=None, ax=ax[0],
                                    **formatting_kwargs)
            _draw_graph_subplot(G, layout=ref_layout, title=None, ax=ax_graph,
                                **formatting_kwargs)
            pdf.savefig(f, bbox_inches='tight', pad_inches=0)
            plt.close()
    else:
        if reference:
            ref_layout = nx.drawing.nx_agraph.graphviz_layout(Gt, prog="dot")
            _draw_graph_subplot(Gt, layout=ref_layout, title=names[1], ax=ax[0],
                                **formatting_kwargs)
        _draw_graph_subplot(G, layout=ref_layout, title=names[0], ax=ax_graph,
                            **formatting_kwargs)
        plt.show()


def _format_graph(
        G: nx.DiGraph,
        Gt: nx.DiGraph = None,
        ok_color="green",
        inv_color="lightgreen",
        wrong_color="black") -> nx.DiGraph:
    if Gt is None:
        for u, v in G.edges():
            G[u][v]['color'] = "black"
            G[u][v]['width'] = 1.0
            G[u][v]['style'] = 'solid'
            G[u][v]['alpha'] = 0.7
    else:
        for u, v in G.edges():
            if Gt.has_edge(u, v):
                G[u][v]['color'] = ok_color
                G[u][v]['width'] = 3.0
                G[u][v]['style'] = 'solid'
                G[u][v]['alpha'] = 1.0
            elif Gt.has_edge(v, u):
                G[u][v]['color'] = inv_color
                G[u][v]['width'] = 2.0
                G[u][v]['style'] = 'solid'
                G[u][v]['alpha'] = 0.8
            else:
                G[u][v]['color'] = wrong_color
                G[u][v]['width'] = 1.0
                G[u][v]['style'] = '--'
                G[u][v]['alpha'] = 0.6
    return G


def _draw_graph_subplot(G: nx.DiGraph, layout: dict, title: str, ax: plt.Axes, **formatting_kwargs):
    colors = list(nx.get_edge_attributes(G, 'color').values())
    widths = list(nx.get_edge_attributes(G, 'width').values())
    styles = list(nx.get_edge_attributes(G, 'style').values())
    nx.draw(G, pos=layout, edge_color=colors, width=widths, style=styles,
            **formatting_kwargs, ax=ax)
    if title is not None:
        ax.set_title(title, y=-0.1)


def _cleanup_graph(G: nx.DiGraph) -> nx.DiGraph:
    if '\\n' in G.nodes:
        G.remove_node('\\n')
    return G
