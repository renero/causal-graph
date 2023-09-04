"""
Hierarchy of links

Can I use the information above to decide wether to connect groups 
of variables linked together?

"""
from collections import defaultdict
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd

from copy import copy
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from causalgraph.independence.mic import pairwise_mic


class Hierarchies:

    correlations = None
    linkage_mat = None

    def __init__(
            self,
            method: str = 'spearman',
            alpha: float = 0.6,
            c: int = 15,
            linkage_method: str = 'complete',
            correlation_th: float = None,
            prog_bar: bool = False,
            verbose: bool = False,
            silent: bool = False):
        """
        Parameters
        ----------
            method (str or Callable) : Method to use to compute the correlation. 
                Default is 'spearman', but can also be 'pearson', 'kendall' or 'mic'. 
            alpha (float) : Threshold for the correlation. Default is 0.6.
            c (int) : Number of clusters to be formed. Default is 15. Only valid with MIC.
            linkage_method (str) : Method to use to compute the linkage. 
                Default is 'complete'.
            **kwargs : Keyword arguments to be passed to the plot_dendogram function.
        """
        self.method = method
        self.alpha = alpha
        self.c = c
        self.linkage_method = linkage_method
        self.correlation_th = correlation_th
        self.prog_bar = prog_bar
        self.verbose_ = verbose
        self.silent_ = silent

    def fit(self, X: pd.DataFrame, y=None):
        """
        Compute the hierarchy of links between variables using the correlation method
        specified in `corr_method`.


        Returns
        -------
            - np.array, np.array
                Arrays with the correlation matrix and the linkage matrix.

        """
        # copy X into self.data
        self.data = X.copy()
        self.feature_names = list(self.data.columns)
        
        # Set the list of correlated features for each target
        self.correlations = self.compute_correlation_matrix(
            self.data, method=self.method, prog_bar=self.prog_bar)
        self.correlated_features = self.compute_correlated_features(
            self.correlations, self.correlation_th, self.feature_names)

        # Compute the dissimilarity matrix
        self.dissimilarity = 1 - np.abs(self.correlations)
        close_to_zero = self.dissimilarity < 1.0e-6
        self.dissimilarity[close_to_zero] = 0.0
        self.linkage_mat = linkage(squareform(
            self.dissimilarity), self.linkage_method)

        return self 

    @staticmethod
    def compute_correlation_matrix(data: pd.DataFrame, method='spearman', prog_bar=False):
        if method in ['spearman', 'pearson', 'kendall']:
            correlations = data.corr(method=method)
        elif method == 'mic':
            correlations = pairwise_mic(
                data, alpha=alpha, c=c, progbar=prog_bar)
        else:
            raise ValueError(
                f"Unknown correlation method: {method}. \
                    Use 'spearman', 'pearson', 'kendall' or 'mic'.")
            
        return correlations

    @staticmethod
    def compute_correlated_features(correlations, correlation_th, feature_names, verbose=False):
        correlated_features = defaultdict(list)
        if correlation_th:
            for target_name in feature_names:
                corr_features = list(
                    correlations[(correlations[target_name] > correlation_th)
                                      & (correlations[target_name] < 1.0)].index)
                if len(corr_features) > 0:
                    correlated_features[target_name] = corr_features
                    if verbose:
                        print(
                            f"CORRELATED FEATS for {target_name}: {corr_features}")

        return correlated_features

    def expand_clusters_perm_importance(self, pi, ground_truth=None):
        """
        Expand the clusters of the linkage matrix to include the features that are
        in the same cluster in the permutation importance matrix. It expands, for
        each cluster, with the metrics related to correlation, deltas, backward PI, etc.
        Used to determine if some criteria can be extracted 

        Parameters
        ----------
            pi : pd.DataFrame   
                Permutation importance matrix.
            ground_truth : pd.DataFrame
                Ground truth matrix.

        Returns
        -------
            None
        """
        clusters = self._clusters_from_linkage(
            self.linkage_mat, self.data.columns)
        correlations = np.abs(self.data.corr("spearman"))
        for i, feature in enumerate(self.data.columns):
            print(f"{feature}")
            conn = "└─"
            for j, (name, fwd_PI) in enumerate(pi.feature_importances_[feature].items()):
                # check if this the last item in the list
                conn = "└─> " if j == len(pi.feature_importances_[
                    feature]) - 1 else "├─> "
                print(f" {conn}{name:>4s}: {fwd_PI:.02f}", end="")
                if feature in pi.feature_importances_[name].keys():
                    bwd_PI = pi.feature_importances_[name][feature]
                    print(
                        f" r({bwd_PI:.02f},∂={np.abs(fwd_PI-bwd_PI):.02f})", end="")
                    reverse = True
                else:
                    print(" r(0.00,∂=0.00)", end="")
                    reverse = False
                print(f" | c({correlations.loc[feature, name]:.02f})", end="")
                print(f" | R2({pi.regression_importances_[i]:.2f})", end="")
                degree = self._are_connected(clusters, feature, name)
                connected = True if degree is not None else False
                print(f" | d{degree:.02f}", end="") if connected else print(
                    f" | d0.00", end="")
                if ground_truth is None:
                    print("")
                if reverse:
                    if fwd_PI > bwd_PI:
                        check = "✅" if ground_truth.has_edge(
                            feature, name) else "❌"
                        print(f" | {feature:4s} -> {name:4s} {check}")
                    else:
                        check = "✅" if ground_truth.has_edge(
                            name, feature) else "❌"
                        print(f" | {feature:4s} <- {name:4s} {check}")
                else:
                    print("")

    def _cluster_features(self, method, threshold):
        """
        This function clusters the features of the data based on the linkage matrix
        obtained from the hierarchical clustering. It is used in the method 
        plot_correlations.
        """
        # Keep the indices to sort labels
        labels = fcluster(self.linkage_mat, threshold, criterion='distance')
        labels_order = np.argsort(labels)
        sorted_colnames = self.data.columns[labels_order]

        # Build a new dataframe with the sorted columns
        for idx, i in enumerate(sorted_colnames):
            if idx == 0:
                clustered = pd.DataFrame(self.data[i])
            else:
                df_to_append = pd.DataFrame(self.data[i])
                clustered = pd.concat([clustered, df_to_append], axis=1)
        return clustered.corr(method), sorted_colnames

    def hierarchical_dissimilarities(self):
        """
        Compute the dissimilarities between features in a hierarchical clustering.
        Two features' dissimilarity is the distance between them in the same cluster 
        of the dedrogram or to the inmediately superior cluster. If they are connected
        by a path of clusters, the dissimilarity is the maximum dissimilarity between
        the features in the path.

        Returns
        -------
            pd.DataFrame
                Dissimilarities between features.

        """
        feature_names = list(self.data.columns)
        hierarchical_dissimilarity = pd.DataFrame(columns=feature_names)
        clusters = self._clusters_from_linkage(self.linkage_mat, feature_names)
        for feature in feature_names:
            remaining_features = [f for f in feature_names if f != feature]
            for name in remaining_features:
                hierarchical_dissimilarity.loc[feature, name] = self._are_connected(
                    clusters, feature, name)

        hierarchical_dissimilarity[hierarchical_dissimilarity.isna()] = 0.0

        return hierarchical_dissimilarity

    def _clusters_from_linkage(linkage_mat, features):
        """
        Get the clusters from the linkage matrix, in the form:
            {'K10': [('V8', 'V9'), 0.32309495637982555],
            'K11': [('V1', 'V2'), 0.3416643106572427],
            'K12': [('V5', 'K11'), 0.4056455905823624],
            'K13': [('V4', 'V7'), 0.4388866835467342],
            'K14': [('K10', 'K12'), 0.46126091704366823]}
        """
        clusters = {}
        Z = linkage_mat
        new_cluster_id = len(features)
        for n in range(Z.shape[0]):
            new_cluster = new_cluster_id
            new_cluster_id += 1
            feature_a = int(Z[n][0])
            feature_b = int(Z[n][1])
            if feature_a < len(features):
                feature_a = features[feature_a]
            else:
                feature_a = f'K{feature_a}'
            if feature_b < len(features):
                feature_b = features[feature_b]
            else:
                feature_b = f'K{feature_b}'
            # print(f"K{new_cluster}: {feature_a:<3s} {feature_b:>3s}, sim: {Z[n][2]:.02f}")
            clusters[f"K{new_cluster}"] = [(feature_a, feature_b), Z[n][2]]

        return clusters

    @staticmethod
    def _get_cluster(clusters: List[str], node: str) -> str:
        for k in clusters.keys():
            if node in clusters[k][0]:
                return k
        return None

    @staticmethod
    def _is_cluster(node: str) -> bool:
        return node.startswith('K')

    @staticmethod
    def _contains_a_cluster(clusters: List[str], node: str) -> bool:
        return any([hierarchies._is_cluster(n) for n in clusters[node][0]])

    @staticmethod
    def _get_cluster_element(clusters: List[str], node: str) -> str:
        for n in clusters[node][0]:
            if hierarchies._is_cluster(n):
                return n
        return None

    @staticmethod
    def _in_cluster(cluster, node):
        return node in cluster[0]

    @staticmethod
    def _in_same_cluster(clusters: List[str], node1: str, node2: str) -> str:
        for k in clusters.keys():
            if node1 in clusters[k][0] and node2 in clusters[k][0]:
                return k
        return None

    @staticmethod
    def _are_connected(clusters: List[str], node1: str, node2: str) -> Tuple[bool, float]:
        """
        Determine if two nodes are connected in the hierarchical clustering represented
        in the clusters dictionary, obtained from the `linkage` function.

        Parameters
        ----------
            - clusters (List[str])
                List of clusters.
            - node1 (str)   
                First node.
            - node2 (str)   
                Second node.

        Returns
        -------
            - float
                Degree of disimilarity between the nodes. The higher the value, the more
                dissimilar the nodes are. If None is returned, the nodes are not connected.
        """
        k = Hierarchies._in_same_cluster(clusters, node1, node2)
        if k is not None:
            # Obtain the second valud in the dictionary of clusters for the pair of nodes
            return clusters[k][1]
        cluster1 = Hierarchies._get_cluster(clusters, node1)
        cluster2 = Hierarchies._get_cluster(clusters, node2)
        if Hierarchies._contains_a_cluster(clusters, cluster1):
            ref_cluster = Hierarchies._get_cluster_element(clusters, cluster1)
            if Hierarchies._in_cluster(clusters[ref_cluster], node2):
                disimilarity = clusters[Hierarchies._get_cluster(
                    clusters, node1)][1]
                return disimilarity
        if Hierarchies._contains_a_cluster(clusters, cluster2):
            ref_cluster = Hierarchies._get_cluster_element(clusters, cluster2)
            if Hierarchies._in_cluster(clusters[ref_cluster], node1):
                disimilarity = clusters[Hierarchies._get_cluster(
                    clusters, node2)][1]
                return disimilarity
        return None

    @staticmethod
    def _set_colormap(color_threshold=0.15, max_color=0.8) -> ListedColormap:
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
        cw = plt.get_cmap('coolwarm')
        cmap = ListedColormap([cw(x)
                               for x in np.arange(color_threshold, max_color, 0.01)])
        cm = copy(cmap)
        cm.set_under(color='white')
        return cm

    def plot(self, threshold=0.5, **kwargs):
        """
        Plot the hierarchical clustering and correlation matrix of the data.

        https://www.kaggle.com/code/sgalella/correlation-heatmaps-with-hierarchical-clustering/notebook
        """
        f_size = kwargs.get('figsize', (9, 4))
        title = kwargs.get('title', 'Correlation matrix')
        fontsize = kwargs.get('fontsize', 9)
        xrot = kwargs.get('xrot', 0)
        cm = Hierarchies._set_colormap(
            color_threshold=threshold, max_color=0.9)
        precision = 2

        def myround(v, ndigits=2):
            if np.isclose(v, 0.0):
                return "0"
            return format(v, '.' + str(ndigits) + 'f')

        _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=f_size)
        dendrogram(self.linkage_mat, labels=self.data.columns, orientation='top',
                   leaf_rotation=90, ax=ax1)
        ax1.set_title('Hierarchical Clustering Dendrogram')
        ax1.set_ylabel("Dissimilarity")
        ax1.set_ylim(0, 1)

        correlations, sorted_colnames = self._cluster_features(
            "spearman", threshold)

        corr_data = np.abs(copy(correlations.values))
        ncols, nrows = corr_data.shape
        for x in range(ncols):
            for y in range(nrows):
                if x == y or corr_data[x, y] < threshold:
                    corr_data[x, y] = 0

        ax2.set_xticks(range(len(sorted_colnames)), sorted_colnames, rotation=xrot,
                       horizontalalignment='center',
                       fontsize=fontsize, fontname="Arial", color='black')
        ax2.set_yticks(range(len(sorted_colnames)), sorted_colnames,
                       verticalalignment='center',
                       fontsize=fontsize, fontname="Arial", color='black')
        ax2.imshow(corr_data, cmap=cm, vmin=threshold,
                   vmax=1.0, aspect='equal')
        ax2.grid(True, which='major', alpha=.25)
        for x in range(ncols):
            for y in range(nrows):
                if (x) == y:
                    ax2.annotate('x', xy=(y, x),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontsize=fontsize, fontname="Arial", color='black')
                if (x) != y and not np.isclose(round(corr_data[x, y], precision), 0.0):
                    ax2.annotate(myround(corr_data[x, y], precision), xy=(y, x),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontsize=fontsize, fontname="Arial", color='black')
        plt.tick_params(pad=10, axis='x', which='both')

        ax2.spines['top'].set_linewidth(.3)
        ax2.spines['right'].set_linewidth(.3)
        ax2.spines['left'].set_linewidth(1)
        ax2.spines['left'].set_edgecolor('grey')
        ax2.spines['bottom'].set_linewidth(.3)
        ax2.set_title(title)


def _get_directed_pair(g, u, v):
    if g.has_edge(u, v) and not g.has_edge(v, u):
        return (u, v)
    elif g.has_edge(v, u) and not g.has_edge(u, v):
        return (v, u)
    else:
        return None


def _get_direction(g: nx.DiGraph, u: str, v: str) -> Tuple[str, Union[int, None]]:
    """
    Returns the direction of the edge between u and v.

    Arguments:
    ----------
        - g (networkx.DiGraph): The graph to search in.
        - u (str): The source node.
        - v (str): The target node.

    Returns:
    --------
        - (str, int): A string respresenting the direction of the edge 
            between u and v, and an integer representing it. The string 
            can be either '-->', '<--', '<->' or ' · '. And the integer 
            can be either 1, -1, 0 or None for each possibility.

    """

    if not g.has_node(u) or not g.has_node(v):
        raise ValueError(f"Node(s) {u} and/or {v} not in graph.")

    if g.has_edge(u, v) and not g.has_edge(v, u):
        return "-->", 1
    elif g.has_edge(v, u) and not g.has_edge(u, v):
        return "<--", -1
    elif g.has_edge(u, v) and g.has_edge(v, u):
        return "<->", 0
    else:
        return " · ", None


def connect_isolated_nodes(G, linkage_mat, feature_names, verbose=False):
    """
    Connect isolated nodes in the graph, based on their relationship in the
    hierarchical clustering provided through the linkage_mat.

    Arguments:
    ----------
        - G (networkx.DiGraph): The graph to search in.
        - linkage_mat (np.ndarray): The linkage matrix.
        - feature_names (List[str]): The list of feature names.
        - verbose (bool): Whether to print information about the process.

    Returns:
    --------
        - networkx.DiGraph: The graph with connected isolated nodes.

    Notes:
    ------
        The linkage matrix is a matrix of the following form:

            [i, j, distance, n_items, direction]

        where:
            - i, j: The indices of the two nodes in the linkage.
            - distance: The distance between the two nodes.
            - n_items: The number of items in the two nodes.
            - direction: The direction of the edge between the two nodes.

        and it can be obtained using the function: `compute_hierarchies`
    """
    node_names = [f for f in feature_names]
    G_h = nx.DiGraph()
    G_h.add_edges_from(G.edges(data=False))

    for i in range(linkage_mat.shape[0]):
        _, num = linkage_mat[i][2], int(linkage_mat[i][3])
        if num > 2:
            continue
        u, v = node_names[int(linkage_mat[i][0])
                          ], node_names[int(linkage_mat[i][1])]
        arrow, direction = _get_direction(G_h, u, v)

        # Consider only clusters formed by two nodes, not other clusters.
        features_cluster = True if num == 2 else False
        if direction is None and features_cluster:
            G_h.add_edge(u, v, weight=None)
            if verbose:
                print(f"Adding edge {u} {arrow} {v}")

    return G_h


def connect_hierarchies(G, linkage_mat, feature_names, verbose=True):
    cluster_id = len(feature_names)
    node_names = [f for f in feature_names]
    clusters = {}
    G_h = nx.DiGraph()
    G_h.add_edges_from(G.edges(data=True))
    if verbose:
        print(
            f"{'from':>4s}  :  {'to':<4s}  {'weight':6s}  {'n.items':7s} {'names':5s}")
    for i in range(linkage_mat.shape[0]):
        u, v = node_names[int(linkage_mat[i][0])
                          ], node_names[int(linkage_mat[i][1])]
        weight, num = linkage_mat[i][2], int(linkage_mat[i][3])
        arrow, direction = _get_direction(G_h, u, v)
        kname = f"K#{cluster_id}"
        node_names.append(kname)
        clusters[kname] = (u, v)
        if verbose:
            print(f"{u:>4s} {arrow:^s} {v:<4s}  {weight:6.4f}  {str(num):^7s} {kname:^5s}",
                  end="")
        cluster_id += 1

        # If I'm forming the first cluster, nothing to do.
        if cluster_id <= len(feature_names)+1:
            if verbose:
                print()
            continue

        # Determine if source is a node or a cluster
        if u in clusters.keys():
            pair = _get_directed_pair(G_h, *clusters[u])
            if pair is not None:
                source = pair[1]
        else:
            source = u

        # Determine if target is a node or cluster. If node, continue
        if v in clusters.keys():
            pair = _get_directed_pair(G_h, *clusters[v])
            if pair is not None:
                target = pair[0]
            else:
                if verbose:
                    print()
                continue
        else:
            if verbose:
                print()
            continue

        if _get_directed_pair(G_h, source, target) is None:
            if verbose:
                print(f" Add {source} --> {target}")
            G_h.add_edge(source, target)
        else:
            if verbose:
                print()
            continue

    return G_h


def plot_dendogram_correlations(correlations, feature_names: List[str], **kwargs):
    """
    Plot the dendrogram of the correlation matrix.

    Parameters
    ----------
        - correlations (pd.DataFrame)
            Correlation matrix.
        - feature_names (List[str])
            List of feature names.
        - kwargs
            Keyword arguments to be passed to the plot_dendogram function.
    """
    figsize = kwargs.get("figsize", (5, 3))
    dissimilarity = 1 - abs(correlations)
    Z = linkage(squareform(dissimilarity), 'single')
    plt.figure(figsize=figsize)
    ax = plt.gca()
    dendrogram(Z, labels=feature_names, orientation='left', ax=ax)
    plt.show()
    return Z


if __name__ == "__main__":
    alpha = 0.8
    c = 15

    test_data = pd.read_csv("/Users/renero/phd/data/generated_linear_10.csv")
    h = Hierarchies(method='mic', alpha=alpha, c=c)
    h.fit(test_data)
    h.plot()
    plt.show()
