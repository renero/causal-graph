#
# This file includes all the plot methods for the causal graph
#
# (C) J. Renero, 2022, 2023
#

from copy import copy
from typing import Any, Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydotplus
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
from pydot import Dot


# Defaults for the graphs plotted
formatting_kwargs = {
    "node_size": 1000,
    "node_color": "white",
    "alpha": 0.8,
    "edgecolors": "black",
    "font_weight": "bold",
    "font_family": "monospace",
    "horizontalalignment": "center",
    "verticalalignment": "center_baseline",
    "with_labels": True
}


def setup_plot(**kwargs):  # tex=True, font="serif", dpi=75, font_size=10):
    """Customize figure settings.

    Args:
        tex (bool, optional): use LaTeX. Defaults to True.
        font (str, optional): font type. Defaults to "serif".
        dpi (int, optional): dots per inch. Defaults to 180.
    """
    font_size = kwargs.pop("font_size", 10)
    usetex = kwargs.pop("usetex", True)
    font_familiy = kwargs.pop("font_family", "serif")
    dpi = kwargs.pop("dpi", 75)
    format = kwargs.pop("format", "pdf")
    title_size = kwargs.pop("title_size", 8)
    axis_labelsize = kwargs.pop("axis_labelsize", 8)
    xtick_labelsize = kwargs.pop("xtick_labelsize", 8)
    ytick_labelsize = kwargs.pop("ytick_labelsize", 8)
    legend_fontsize = kwargs.pop("legend_fontsize", 8)
    axes_labelpad = kwargs.pop("axes_labelpad", 4)
    xtick_majorpad = kwargs.pop("xtick_majorpad", 3)
    ytick_majorpad = kwargs.pop("ytick_majorpad", 3)

    plt.rcParams.update(
        {
            "font.size": font_size,
            "font.family": font_familiy,
            "text.usetex": usetex,
            "figure.subplot.top": 0.9,
            "figure.subplot.right": 0.9,
            "figure.subplot.left": 0.15,
            "figure.subplot.bottom": 0.12,
            "figure.subplot.hspace": 0.4,
            "figure.dpi": dpi,
            "savefig.dpi": 180,
            "savefig.format": format,
            "axes.titlesize": title_size,
            "axes.labelsize": axis_labelsize,
            "axes.axisbelow": True,
            "axes.labelpad": axes_labelpad,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "xtick.minor.size": 2.25,
            "xtick.major.pad": xtick_majorpad,
            "xtick.minor.pad": 7.5,
            "ytick.major.pad": ytick_majorpad,
            "ytick.minor.pad": 7.5,
            "ytick.major.size": 5,
            "ytick.minor.size": 2.25,
            "xtick.labelsize": xtick_labelsize,
            "ytick.labelsize": ytick_labelsize,
            "legend.fontsize": legend_fontsize,
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
        plot_func: Callable,
        *plot_args: Any,
        **kwargs: Any) -> None:
    """
    Plots a set of subplots.

    Arguments:
    ----------
        plot_func: function
            The function to be used to plot each subplot.
        num_cols: int
            The number of columns in the subplot grid.
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
    fig_size = kwargs.pop("fig_size", (8, 6))
    title = kwargs.pop("title", None)
    num_cols = kwargs.pop("num_cols", 4)
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


def format_graph(
        G: nx.DiGraph,
        Gt: nx.DiGraph = None,
        ok_color="green",
        inv_color="lightgreen",
        wrong_color="black",
        missing_color="grey"
    ) -> nx.DiGraph:
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
            elif Gt.has_edge(v, u):           # The edge exists, but reversed
                G[u][v]['color'] = inv_color
                G[u][v]['width'] = 2.0
                G[u][v]['style'] = 'solid'
                G[u][v]['alpha'] = 0.8
            else:                             # The edge does not exist
                G[u][v]['color'] = wrong_color
                G[u][v]['width'] = 1.0
                G[u][v]['style'] = '--'
                G[u][v]['alpha'] = 0.6
        if missing_color is not None:
            for u, v in Gt.edges():
                if not G.has_edge(u, v) and not G.has_edge(v, u):
                    G[u][v]['color'] = missing_color
                    G[u][v]['width'] = 1.0
                    G[u][v]['style'] = '--'
                    G[u][v]['alpha'] = 0.6
    return G


def draw_graph_subplot(
        G: nx.DiGraph,
        layout: dict,
        title: str,
        ax: plt.Axes,
        **formatting_kwargs):
    """
    Draw a graph in a subplot.

    Parameters
    ----------
    G : nx.DiGraph
        The graph to be drawn.
    layout : dict
        The layout of the graph.
    title : str
        The title of the graph.
    ax : plt.Axes
        The axis in which to draw the graph.
    **formatting_kwargs : dict
        The formatting arguments for the graph.

    Returns
    -------
    None
    """
    edge_colors = list(nx.get_edge_attributes(G, 'color').values())
    widths = list(nx.get_edge_attributes(G, 'width').values())
    styles = list(nx.get_edge_attributes(G, 'style').values())

    # Create a colormap list with the colors of the nodes, based on the regr_score
    if all(['regr_score' in G.nodes[node] for node in G.nodes]):
        reg_scores = [G.nodes[node]['regr_score'] for node in G.nodes]
        max_cmap_value = max(max(reg_scores), 1.0)
        color_map = set_colormap(0.0, max_cmap_value, 'RdYlGn_r')
        # color_map_r = set_colormap(0.0, max_cmap_value, 'gist_gray')
        formatting_kwargs['font_color'] = "black"
        
        # Set with_labels to False if there is color information of each node, 
        # since I will draw the labels afterwards
        formatting_kwargs['with_labels'] = False

        # Set the node colors and the label colors according to the value of the
        # regr_score of each node.
        node_colors = []
        label_colors = []
        for node in G:
            node_colors.append(color_map(G.nodes[node]['regr_score']))
            # label_colors.append(color_map_r(G.nodes[node]['regr_score']))
        formatting_kwargs['node_color'] = node_colors

    nx.draw(G, pos=layout, edge_color=edge_colors,
            width=widths, style=styles, **formatting_kwargs, ax=ax)
    
    if formatting_kwargs['with_labels'] == False:
        for i, node in enumerate(G):
            # font_color = label_colors[i]
            nx.draw_networkx_labels(
                G, pos=layout, labels={node:node}, #font_color=font_color, 
                font_weight=formatting_kwargs['font_weight'],
                font_family=formatting_kwargs['font_family'],
                horizontalalignment=formatting_kwargs['horizontalalignment'],
                verticalalignment=formatting_kwargs['verticalalignment'],
                ax=ax)
    
    if title is not None:
        ax.set_title(title, y=-0.1)


def cleanup_graph(G: nx.DiGraph) -> nx.DiGraph:
    if '\\n' in G.nodes:
        G.remove_node('\\n')
    return G


def set_colormap(
        color_threshold=0.15,
        max_color=0.8,
        cmap_name: str = "OrRd") -> ListedColormap:
    """
    Set the colormap for the graph edges.

    Parameters
    ----------
    color_threshold : float
        The threshold for the color of the values in the plot, below which the color
        will be white.
    max_color : float
        The maximum color for the edges, above which the color will be red.

    Returns
    -------
    LinearColormap 
        The colormap to be used in the plot.
    """
    cw = plt.get_cmap(cmap_name)
    cmap = ListedColormap([cw(x)
                           for x in np.arange(color_threshold, max_color, 0.01)])
    cm = copy(cmap)
    cm.set_under(color='white')
    return cm


def dag2dot(
    G: nx.DiGraph,
    undirected=False,
    name: str = "my_dotgraph",
    odots: bool = True,
    **kwargs,
) -> Dot:
    """
    Display a DOT of the graph in the notebook.

    Args:
        G (nx.Graph or DiGraph): the graph to be represented.
        undirected (bool): default False, indicates whether the plot is forced
            to contain no arrows.
        plot (bool): default is True, this flag can be used to simply generate
            the object but not plot, in case the object is needed to generate
            a PNG version of the DOT, for instance.
        name (str): the name to be embedded in the Dot object for this graph.
        odots (bool): represent edges with biconnections with circles (odots). if
            this is set to false, then the edge simply has no arrowheads.

    Returns:
        pydot.Dot object
    """
    if len(list(G.edges())) == 0:
        return None
    # Obtain the DOT version of the NX.DiGraph and visualize it.
    if undirected:
        G = G.to_undirected()
        dot_object = nx.nx_pydot.to_pydot(G)
    else:
        # Make a dot Object with edges reflecting biconnections as non-connected edges
        # or arrowheads as circles.
        dot_str = "strict digraph" + name + "{\nconcentrate=true;\n"
        for node in G.nodes():
            dot_str += f"{node};\n"
        if odots:
            options = "[arrowhead=odot, arrowtail=odot, dir=both]"
        else:
            options = "[dir=none]"
        for u, v in G.edges():
            if G.has_edge(v, u):
                dot_str += f"{u} -> {v} {options};\n"
            else:
                dot_str += f"{u} -> {v};\n"
        dot_str += "}\n"
        dot_object = pydotplus.graph_from_dot_data(dot_str)

    # This is to display single arrows with two heads instead of two arrows with
    # one head towards each direction.
    dot_object.set_concentrate(True)
    dot_object.del_node('"\\n"')

    return dot_object


def values_distribution(values, **kwargs):
    # Plot two subplots: one with probability density of "all_mean_shap_values"
    # and another with the cumulative density of "all_shap_values"
    figsize = kwargs.get('figsize', (7, 5))
    _, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].set_title("Probability density")
    ax[1].set_title("Cumulative density")
    ax[0].set_xlabel("Mean SHAP values")
    ax[1].set_xlabel("Mean SHAP values")
    ax[0].set_ylabel("Probability")
    ax[1].set_ylabel("Cumulative probability")
    sns.histplot(data=values, ax=ax[0], kde=True)
    sns.ecdfplot(data=values, ax=ax[1])
    plt.tight_layout()
    plt.show()


def correlations(correlations, sorted_colnames, threshold, ax, **kwargs):
    """
    Plot the correlation matrix of the data.
    
    Parameters
    ----------
        - correlations (pd.DataFrame)
            Correlation matrix.
        - sorted_colnames (List[str])
            List of sorted column names.
        - threshold (float)
            Threshold for the correlation.
        - ax (matplotlib.axes.Axes)
            Axes to plot the correlation matrix.
        - **kwargs
            Keyword arguments to be passed to the plot_dendogram function.
            - title (str)
                Title of the plot.
            - fontsize (int)    
                Font size for the labels.
            - fontname (str)
                Font name for the labels.
            - xrot (int)
                Rotation of the labels.
                
    Returns
    -------
        None
    """
    title = kwargs.get('title', 'Correlation matrix')
    fontsize = kwargs.get('fontsize', 9)
    fontname = kwargs.get('fontname', "Arial")
    xrot = kwargs.get('xrot', 0)
    cm = set_colormap(color_threshold=threshold, max_color=0.9)
    precision = 2

    def myround(v, ndigits=2):
        if np.isclose(v, 0.0):
            return "0"
        return format(v, '.' + str(ndigits) + 'f')

    corr_data = np.abs(copy(correlations.values))
    ncols, nrows = corr_data.shape
    for x in range(ncols):
        for y in range(nrows):
            if x == y or corr_data[x, y] < threshold:
                corr_data[x, y] = 0

    ax.set_xticks(range(len(sorted_colnames)), sorted_colnames, rotation=xrot,
                    horizontalalignment='center',
                    fontsize=fontsize, fontname=fontname, color='black')
    ax.set_yticks(range(len(sorted_colnames)), sorted_colnames,
                    verticalalignment='center',
                    fontsize=fontsize, fontname=fontname, color='black')
    ax.imshow(corr_data, cmap=cm, vmin=threshold,
                vmax=1.0, aspect='equal')
    ax.grid(True, which='major', alpha=.25)
    for x in range(ncols):
        for y in range(nrows):
            if (x) == y:
                ax.annotate('x', xy=(y, x),
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=fontsize, fontname=fontname, color='black')
            if (x) != y and not np.isclose(round(corr_data[x, y], precision), 0.0):
                ax.annotate(myround(corr_data[x, y], precision), xy=(y, x),
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=fontsize, fontname=fontname, color='black')
    plt.tick_params(pad=10, axis='x', which='both')

    ax.spines['top'].set_linewidth(.3)
    ax.spines['right'].set_linewidth(.3)
    ax.spines['left'].set_linewidth(1)
    ax.spines['left'].set_edgecolor('grey')
    ax.spines['bottom'].set_linewidth(.3)
    ax.set_title(title)
