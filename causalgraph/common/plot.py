"""
This file includes all the plot methods for the causal graph

(C) J. Renero, 2022, 2023
"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name, E1121:too-many-function-args
# pylint: disable=C0116:missing-function-docstring, W0212:protected-access
# pylint: disable=R0913:too-many-arguments, disable:W0212:protected-access
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

from copy import copy
from typing import Any, Callable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pydotplus
import seaborn as sns
from matplotlib import axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
from pydot import Dot
from scipy.cluster.hierarchy import dendrogram
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from causalgraph.metrics.compare_graphs import evaluate_graph

# Defaults for the graphs plotted
formatting_kwargs = {
    "node_size": 800,
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
    file_format = kwargs.pop("file_format", "pdf")
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
            "savefig.format": file_format,
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
        *plot_args: List
            The arguments to be passed to the plot function.
        **kwargs: Dict
            Additional arguments to be passed to the plot function.

    Returns:
    --------
        fig: Figure
            The figure containing the subplots.
    """
    figsize = kwargs.pop("figsize", (8, 6))
    title = kwargs.pop("title", None)
    num_cols = kwargs.pop("num_cols", 4)
    setup_plot(**kwargs)
    num_rows = len(plot_args) // num_cols
    if len(plot_args) % num_cols != 0:
        num_rows += 1

    def blank(ax):
        """
        Create a blank subplot.
        """
        npArray = np.array([[[255, 255, 255, 255]]], dtype="uint8")
        ax.imshow(npArray, interpolation="nearest")
        ax.set_axis_off()

    def ax_index(i, j):
        """
        Return the axis index, considering special cases where the number of rows
        or columns is 1.
        """
        nonlocal ax
        if num_rows == 1:
            return ax[j]
        if num_cols == 1:
            return ax[i]
        return ax[i][j]

    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < len(plot_args):
                # axe = ax[i][j]
                plot_func(plot_args[index], ax=ax_index(i,j))
            else:
                blank(ax_index(i,j))

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
    missing_color=None
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
                G[u][v]['style'] = 'dashed'
                G[u][v]['alpha'] = 0.8
            else:                             # The edge does not exist
                G[u][v]['color'] = wrong_color
                G[u][v]['width'] = 1.0
                G[u][v]['style'] = 'dashdot'
                G[u][v]['alpha'] = 0.6
        if missing_color is not None:
            for u, v in Gt.edges():
                if not G.has_edge(u, v) and not G.has_edge(v, u):
                    G.add_edge(u, v)
                    G[u][v]['color'] = missing_color
                    G[u][v]['width'] = 1.0
                    G[u][v]['style'] = 'dotted'
                    G[u][v]['alpha'] = 0.5
    return G


def draw_graph_subplot(
        G: nx.DiGraph,
        root_causes: list = None,
        layout: dict = None,
        title: str = None,
        ax: plt.Axes = None,
        **kwargs):
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
    default_font_color = 'black'
    # create a dictionary with the default font color of each node
    node_font_color = {node: default_font_color for node in G.nodes}

    def luminance(color):
        r, g, b, _ = color
        return 0.299 * r + 0.587 * g + 0.114 * b

    # Create a colormap list with the colors of the nodes, based on the regr_score
    if all(['regr_score' in G.nodes[node] for node in G.nodes]):
        reg_scores: List[float] = [G.nodes[node]['regr_score']
                                   for node in G.nodes]
        max_cmap_value = max(*reg_scores, 1.0)
        color_map = set_colormap(0.0, max_cmap_value, 'RdYlGn_r')
        kwargs['font_color'] = "black"

        # Set with_labels to False if there is color information of each node,
        # since I will draw the labels afterwards
        kwargs['with_labels'] = False

        # Set the node colors and the label colors according to the value of the
        # regr_score of each node.
        node_colors = []
        linewidths = []
        for node in G:
            node_colors.append(color_map(G.nodes[node]['regr_score']))
            if root_causes is not None and node in root_causes:
                linewidths.append(3.0)
            else:
                linewidths.append(1.0)
            lum = luminance(color_map(G.nodes[node]['regr_score']))
            node_font_color[node] = 'black' if lum > 0.5 else 'white'
        kwargs['node_color'] = node_colors
        kwargs['linewidths'] = linewidths

    nx.draw(G, pos=layout, edge_color=edge_colors,
            width=widths, style=styles, **kwargs, ax=ax)

    if kwargs['with_labels'] is False:
        for _, node in enumerate(G):
            # font_color = label_colors[i]
            nx.draw_networkx_labels(
                G, pos=layout, labels={node: node},
                font_color=node_font_color[node],
                font_weight=kwargs['font_weight'],
                font_family=kwargs['font_family'],
                horizontalalignment=kwargs['horizontalalignment'],
                verticalalignment=kwargs['verticalalignment'],
                ax=ax)

    if title is not None:
        ax.set_title(title, fontsize=12, y=-0.1)


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
        odots: bool = True) -> Dot:
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


def values_distribution(values, threshold=None, **kwargs):
    """
    Plot the probability density and cumulative density of a given set of values.

    Parameters:
    ----------
        values (array-like): The values to be plotted.
        **kwargs: Additional keyword arguments for customizing the plot.

    Returns:
    -------
        None
    """
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
    if threshold is not None:
        ax[1].axvline(threshold, color='red', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def correlation_matrix(
        corr_matrix: pd.DataFrame,
        sorted_colnames: List[str] = None,
        threshold: float = 0.5,
        ax: axes.Axes = None,
        **kwargs) -> None:
    """
    Plot the correlation matrix of the data.

    Parameters
    ----------
        - corrs (pd.DataFrame)
            Correlation matrix.
        - sorted_colnames (List[str])
            List of sorted column names. If the dataframe contains the names of
                columns already sorted, then no need to pass this argument.
        - threshold (float)
            Threshold for the correlation. Values below this threshold will
                not be displayed
        - ax (matplotlib.axes.Axes)
            Axes to plot the correlation matrix, in case this is a plot to be
                embedded in a subplot. Otherwise, a new figure will be created and
                this argument is not necessary.
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
    if sorted_colnames is None:
        sorted_colnames = corr_matrix.columns

    if ax is None:
        _, ax = plt.subplots()

    title = kwargs.get('title', 'Correlation matrix')
    fontsize = kwargs.get('fontsize', 9)
    fontname = kwargs.get('fontname', "Arial")
    xrot = kwargs.get('xrot', 90)
    cm = set_colormap(color_threshold=threshold, max_color=0.9)
    precision = 2

    def myround(v, ndigits=2):
        if np.isclose(v, 0.0):
            return "0"
        return format(v, '.' + str(ndigits) + 'f')

    corr_data = np.abs(copy(corr_matrix.values))
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


def hierarchies(hierarchies, threshold=0.5, **kwargs):
    """
    Plot the hierarchical clustering and correlation matrix of the data.

    https://www.kaggle.com/code/sgalella/correlation-heatmaps-with-hierarchical-clustering/notebook

    Parameters
    ----------
        - hierarchies (HierarchicalClustering)
            Hierarchical clustering object.
        - threshold (float)
            Threshold for the correlation.
        - **kwargs
            Additional keyword arguments to be passed to the correlation_matrix function.

    Returns
    -------
        None

    Example
    -------
    ```python
    from causalgraph.common.notebook import Experiment
    from causalgraph.common import plot


    dataset_name = "my_dataset"
    output_path = "output"

    experiment = Experiment(dataset_name, output_path=output_path).load()
    rex = experiment.rex
    plot.correlations(rex.hierarchies)
    ```
    """
    f_size = kwargs.get('figsize', (9, 4))
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=f_size)
    dendrogram(hierarchies.linkage_mat, labels=hierarchies.data.columns,
               orientation='top', leaf_rotation=90, ax=ax1)
    ax1.set_title('Hierarchical Clustering Dendrogram')
    ax1.set_ylabel("Dissimilarity")
    ax1.set_ylim(0, 1)

    correlations, sorted_colnames = hierarchies._cluster_features(
        "spearman", threshold)
    correlation_matrix(
        correlations, sorted_colnames, threshold, ax2, **kwargs)


def dag(
        graph: nx.DiGraph,
        reference: nx.DiGraph = None,
        root_causes: list = None,
        show_metrics: bool = False,
        show_node_fill: bool = True,
        title: str = None,
        ax: plt.Axes = None,
        figsize: Tuple[int, int] = (5, 5),
        dpi: int = 75,
        save_to_pdf: str = None,
        **kwargs):
    """
    Compare two graphs using dot.

    Parameters:
    -----------
    graph: The DAG to compare.
    reference: The reference DAG.
    root_causes: The root causes of the graph.
    show_metrics: Whether to show the metrics of the graph.
    show_node_fill: Whether to show the node fill (corresponding to the root causes).
    title: The title of the graph.
    ax: The axis in which to draw the graph.
    figsize: The size of the figure.
    dpi: The dots per inch of the figure.
    **kwargs: Additional arguments to format the graphs:
        - "node_size": 500
        - "node_color": 'white'
        - "edgecolors": "black"
        - "font_family": "monospace"
        - "horizontalalignment": "center"
        - "verticalalignment": "center_baseline"
        - "with_labels": True
    """
    ncols = 1

    # Overwrite formatting_kwargs with kwargs if they are provided
    formatting_kwargs.update(kwargs)

    # Check consistency
    if show_metrics and reference is None:
        show_metrics = False

    G = nx.DiGraph()
    if show_node_fill:
        G.add_nodes_from(graph.nodes(data=True))
    else:
        G.add_nodes_from(graph.nodes())
    G.add_edges_from(graph.edges())
    if reference:
        # Clean up reference graph for inconsistencies along the DOT conversion
        # and add potential missing nodes to the predicted graph.
        Gt = cleanup_graph(reference.copy())
        for missing in set(list(Gt.nodes)) - set(list(G.nodes)):
            G.add_node(missing)
        G = format_graph(
            G, Gt, inv_color="orange", wrong_color="red", missing_color="lightgrey")
    else:
        G = format_graph(G)

    ref_layout = None
    setup_plot(dpi=dpi)
    if ax is None and show_metrics is False:
        f, axis = plt.subplots(ncols=ncols, figsize=figsize)
    elif ax is None and show_metrics is True:
        metric = evaluate_graph(reference, graph, list(reference.nodes))
        ax = plt.figure(
            figsize=(6, 4), layout="constrained").subplot_mosaic('AAB')
        axis = ax["A"]
        text_axis = ax["B"]
    elif ax is not None and show_metrics is False:
        axis = ax
    else:
        raise ValueError(
            "The 'ax' and 'show_metrics' arguments are mutually exclusive.")
    if save_to_pdf is not None:
        with PdfPages(save_to_pdf) as pdf:
            if reference:
                ref_layout = nx.drawing.nx_agraph.graphviz_layout(
                    Gt, prog="dot")
            draw_graph_subplot(
                G, layout=ref_layout, title=title, ax=axis, **formatting_kwargs)
            pdf.savefig(f, bbox_inches='tight', pad_inches=0)
            plt.close()
    else:
        if reference:
            ref_layout = nx.drawing.nx_agraph.graphviz_layout(
                Gt, prog="dot")
        else:
            ref_layout = nx.drawing.nx_agraph.graphviz_layout(
                G, prog="dot")

        draw_graph_subplot(
            G, root_causes=root_causes, layout=ref_layout, ax=axis, title=title,
            **formatting_kwargs)

        if show_metrics:
            plt.rcParams["font.family"] = "monospace"
            text_axis.text(0.1, 0.5, metric.matplotlib_repr(),
                           ha='left', va='center', size=12)
            text_axis.axis('off')

        if ax is None:
            plt.show()


def dags(
        dags: List[nx.DiGraph],
        ref_graph: nx.DiGraph,
        titles: List[str],
        figsize: Tuple[int, int] = (15, 12),
        dpi: int = 300) -> None:
    """
    Plots multiple directed acyclic graphs (DAGs) in a grid layout.

    Parameters:
    - dags (list): List of DAGs to plot.
    - ref_graph: Reference graph used for layout.
    - titles (list): List of titles for each DAG.
    - figsize (tuple): Figure size (default: (15, 12)).
    - dpi (int): Dots per inch (default: 300).

    Raises:
    - ValueError: If there are too many DAGs to plot.

    Returns:
    - None
    """
    assert len(titles) == len(dags), "Number of titles must match number of DAGs"
    assert len(dags) <=12 and len(dags) > 1, "Number of DAGs must be between 2 and 12"

    layout = _get_subplot_mosaic_layout(len(dags))

    axs = plt.figure(
        figsize=figsize, dpi=dpi, layout="constrained").subplot_mosaic(layout)

    ax_labels = list(axs.keys())
    for i, g in enumerate(dags):
        ax_ = axs[ax_labels[i]]
        dag(graph=g, reference=ref_graph, title=titles[i], ax=ax_)

    plt.tight_layout()
    plt.show()


def _get_subplot_mosaic_layout(n: int) -> str:
    """
    Get a layout string for a subplot mosaic with n subplots.

    The layout strings are hardcoded for 1 to 11 subplots. For more subplots,
    a ValueError is raised.

    Parameters:
    - n (int): The number of subplots.

    Returns:
    - str: The layout string for the subplot mosaic.

    Raises:
    - ValueError: If n is larger than 11.
    """
    layouts = [
        "AB",
        "ABC",
        "AB;CD",
        "AABBCC;.DDEE.",
        "ABC;DEF",
        "ABC;DEF;.G.",
        "ABCD;EFGH",
        "ABC;DEF;GHI",
        "AABBCCDD;EEFFGGHH;.IIJJ.",
        "ABCD;EFGH;IJK.",
        "ABCD;EFGH;IJKL",
    ]
    if n <= len(layouts):
        return layouts[n-2]
    else:
        raise ValueError("Too many DAGs to plot")


def shap_values(shaps: BaseEstimator, **kwargs):
    assert shaps.is_fitted_, "Model not fitted yet"
    plot_args = list(shaps.feature_names)
    return subplots(shaps._plot_shap_summary, *plot_args, **kwargs)


def shap_discrepancies(
        shaps: BaseEstimator,
        target_name: str,
        threshold: float = +100.0,
        regression_line:bool=False,
        reduced: bool = False,
        **kwargs):
    """
    Plot the discrepancies of the SHAP values.

    Parameters:
    -----------
        shaps: BaseEstimator
            The SHAP values to plot.
        target_name: str
            The name of the target variable.
        threshold: float, default=+100.0
            The threshold for the discrepancies. Only discrepancies below this
            threshold will be displayed. Typical values are in (0.0, 5.0), but those
            significant are in the values close to 0.0.
        regression_line: bool, default=False
            If True, include the regression line in the plot.
        Optional arguments:
        - figsize: Tuple[int, int], default=(10, 16)
            The size of the figure.
        - pdf_filename: str, default=None
            The name of the PDF file to save the plot to.
        - dpi: int, default=75
            The DPI of the plot.

    """
    assert shaps.is_fitted_, "Model not fitted yet"
    # shaps._plot_discrepancies(target_name, threshold, **kwargs)

    mpl.rcParams['figure.dpi'] = kwargs.get('dpi', 75)
    pdf_filename = kwargs.get('pdf_filename', None)
    feature_names = [
        f for f in shaps.feature_names if (
            (f != target_name) and
            ((1. - shaps.shap_discrepancies[target_name][f].shap_gof) < threshold)
        )
    ]
    # If no features are found, gracefully return
    if not feature_names:
        print(f"No features with discrepancy index below {threshold} found "
              f"for target {target_name}.")
        return

    # Set the height of the figure to 18 unless there're less than 9 features,
    # in which case, the height is 18/(9 - len(feature_names)).
    if len(feature_names) < 9:
        height = 2*len(feature_names)
    else:
        height = 16
    figsize_ = kwargs.get('figsize', (10, height))

    # The reduced version only plots the scatter and the discrepancies
    nr_axes = 1 if reduced else 4
    fig, ax = plt.subplots(len(feature_names), nr_axes, figsize=figsize_)

    # If the number of features is 1, I must keep ax indexable, so I put it
    # in a list.
    if len(feature_names) == 1:
        ax = [ax]

    for i, parent_name in enumerate(feature_names):
        r = shaps.shap_discrepancies[target_name][parent_name]
        x = shaps.X_test[parent_name].values.reshape(-1, 1)
        y = shaps.X_test[target_name].values.reshape(-1, 1)
        parent_pos = feature_names.index(parent_name)
        s = shaps.shap_scaled_values[target_name][:,
                                                  parent_pos].reshape(-1, 1)
        _plot_discrepancy(
            x, y, s, target_name, parent_name, r, ax[i], regression_line, reduced)

    plt.suptitle(f"Discrepancies for {target_name}")

    if pdf_filename is not None:
        plt.tight_layout(rect=[0, 0.0, 1, 0.97])
        plt.savefig(pdf_filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout(rect=[0, 0.0, 1, 0.97])
        fig.show()


def _plot_discrepancy(x, y, s, target_name, parent_name, r, ax, regression_line, reduced=False):
    """
    Plot the discrepancy between target and SHAP values.

    Args:
        x (array-like): The x-axis values.
        y (array-like): The target values.
        s (array-like): The SHAP values.
        target_name (str): The name of the target variable.
        parent_name (str): The name of the parent variable.
        r (object): The result object containing model parameters and statistics.
        ax (array-like): The array of subplots.
        regression_line (bool): If True, include the regression line in the plot.
        reduced (bool): If True, plot the reduced version of the SHAP values.

    Returns:
        None
    """
    def _remove_ticks_and_box(ax):
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)

    b0_s, b1_s = r.shap_model.params[0], r.shap_model.params[1]
    b0_y, b1_y = r.parent_model.params[0], r.parent_model.params[1]

    mpl.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')

    # If the number of axes is 1, it is not indexable, so I change the only one
    ax_0 = ax if reduced else ax[0]

    # Represent scatter plots
    ax_0.scatter(x, s, alpha=0.5, marker='+')
    ax_0.scatter(x, y, alpha=0.5, marker='.')
    if regression_line:
        ax_0.plot(x, b1_s * x + b0_s, color='blue', linewidth=.5)
        ax_0.plot(x, b1_y * x + b0_y, color='red', linewidth=.5)
    ax_0.set_title(
        fr'$ X_j\, \textrm{{vs.}}\, [X_i | \phi_j] $, '
        fr'\quad $\delta_j^{{(i)}}\!: {1 - r.shap_gof:.2f}$',
        fontsize=10)
    ax_0.set_xlabel(f'${parent_name}$')
    ax_0.set_ylabel(
        fr'$ \mathrm{{{target_name}}} / \phi_{{{parent_name}}} $')

    # Represent target vs. SHAP values
    if not reduced:
        ax[1].scatter(s, y, alpha=0.3, marker='.', color='tab:green')
        ax[1].set_title(
            rf'$\delta_j^{{(i)}}: {1 - r.shap_gof:.2f}$', fontsize=10)
        ax[1].set_xlabel(fr'$ \phi_{{{parent_name}}} $')
        ax[1].set_ylabel(fr'$ \mathrm{{{target_name}}} $')

        # Represent distributions
        pd.DataFrame(s).plot(kind='density', ax=ax[1], label="shap")
        pd.DataFrame(y).plot(kind='density', ax=ax[1], label="parent")
        ax[2].legend().set_visible(False)
        ax[2].set_ylabel('')
        ax[2].set_xlabel(
            fr'$ \mathrm{{{target_name}}} /  \phi_{{{parent_name}}} $')
        ax[2].set_title(rf'$\mathrm{{KS}}({r.ks_pvalue:.2g})$', fontsize=10)

        # Represent fitted vs. residuals
        s_resid = r.shap_model.get_influence().resid_studentized_internal
        y_resid = r.parent_model.get_influence().resid_studentized_internal
        scaler = StandardScaler()
        s_fitted_scaled = scaler.fit_transform(
            r.shap_model.fittedvalues.reshape(-1, 1))
        y_fitted_scaled = scaler.fit_transform(
            r.parent_model.fittedvalues.reshape(-1, 1))
        ax[3].scatter(s_fitted_scaled, s_resid, alpha=0.5, marker='+')
        ax[3].scatter(y_fitted_scaled, y_resid, alpha=0.5,
                        marker='.', color='tab:orange')
        ax[3].set_title(r'$\mathrm{Residuals}$', fontsize=10)
        ax[3].set_xlabel(
            rf'$ \mathrm{{{target_name}}} /  \phi_{{{parent_name}}} $')
        ax[3].set_ylabel(rf'$ \epsilon_{{{target_name}}} / \epsilon_\phi $')

    if not reduced:
        for ax_idx in range(4):
            _remove_ticks_and_box(ax[ax_idx])
    else:
        _remove_ticks_and_box(ax_0)


def deprecated_dags(
        graph: nx.DiGraph,
        reference: nx.DiGraph = None,
        names: List[str] = None,
        figsize: Tuple[int, int] = (10, 5),
        dpi: int = 75,
        save_to_pdf: str = None,
        **kwargs):
    """
    Compare two graphs using dot.

    Parameters:
    -----------
    graph: The DAG to compare.
    reference: The reference DAG.
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
    print("This method is deprecated. Use plot_dags() instead.")
    return

    ncols = 1 if reference is None else 2
    if names is None:
        names = ["Prediction", "Ground truth"]

    # Overwrite formatting_kwargs with kwargs if they are provided
    formatting_kwargs.update(kwargs)

    G = nx.DiGraph()
    G.add_nodes_from(graph.nodes(data=True))
    G.add_edges_from(graph.edges())
    if reference:
        # Clean up reference graph for inconsistencies along the DOT conversion
        # and add potential missing nodes to the predicted graph.
        Gt = cleanup_graph(reference.copy())
        for missing in set(list(Gt.nodes)) - set(list(G.nodes)):
            G.add_node(missing)
        # Gt = _format_graph(Gt, Gt, inv_color="red", wrong_color="black")
        # G  = _format_graph(G, Gt, inv_color="red", wrong_color="gray")
        Gt = format_graph(
            Gt, G, inv_color="lightgreen", wrong_color="lightgreen")
        G = format_graph(G, Gt, inv_color="orange", wrong_color="red")
    else:
        G = format_graph(G)

    ref_layout = None
    setup_plot(dpi=dpi)
    f, ax = plt.subplots(ncols=ncols, figsize=figsize)
    ax_graph = ax[1] if reference else ax
    if save_to_pdf is not None:
        with PdfPages(save_to_pdf) as pdf:
            if reference:
                ref_layout = nx.drawing.nx_agraph.graphviz_layout(
                    Gt, prog="dot")
                draw_graph_subplot(Gt, layout=ref_layout, title=None, ax=ax[0],
                                   **formatting_kwargs)
            draw_graph_subplot(G, layout=ref_layout, title=None, ax=ax_graph,
                               **formatting_kwargs)
            pdf.savefig(f, bbox_inches='tight', pad_inches=0)
            plt.close()
    else:
        if reference:
            ref_layout = nx.drawing.nx_agraph.graphviz_layout(
                Gt, prog="dot")
            draw_graph_subplot(Gt, layout=ref_layout, title=names[1], ax=ax[0],
                               **formatting_kwargs)
        draw_graph_subplot(G, layout=ref_layout, title=names[0], ax=ax_graph,
                           **formatting_kwargs)
        plt.show()
