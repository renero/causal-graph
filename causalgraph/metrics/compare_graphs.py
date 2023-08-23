#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This method computes metrics between a pair of graphs.
    To call this method, simply pass the reference graph, and the
    predicted graph (the one you want to make as much similar to the first one as
    possible), and all metrics will be computed.

    Use:
    >>>> target = nx.DiGraph()
    >>>> target.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
    >>>> target.add_weighted_edges_from([
        ('A', 'B', rand()), ('B', 'D', rand()),('C', 'B', rand()),
        ('D', 'E', rand()), ('C', 'E', rand())])

    >>>> predicted = nx.DiGraph()
    >>>> predicted.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
    >>>> predicted.add_weighted_edges_from([
        ('A', 'B', rand()), ('A', 'C', rand()), ('E', 'A', rand()),
        ('E', 'B', rand()), ('C', 'D', rand())])
    >>>> result = compare_graphs(target, predicted)
    >>>> result
    {'Tp': 1, 'Fn': 4, 'Fp': 4, 'precision': 0.2, 'recall': 0.2,
    'AuPR': 0.18000000000000005, 'f1': 0.20000000000000004, 'SHD': 8}
    >>>> result._precision
    0.2
    >>>> result._recall
    0.2
    >>>> result._f1
    0.2
    >>>> result._SHD
    8
"""

import networkx as nx
import numpy as np

from dataclasses import dataclass
from networkx.linalg import adjacency_matrix
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, Union, List

from causalgraph.metrics.SID import SID

# Code for metrics...
AnyGraph = Union[nx.Graph, nx.DiGraph]


@dataclass
class Metrics:
    """
        This class contains all the metrics computed by the evaluate_graph method.
    """

    Tp: int = 0
    Tn: int = 0
    Fn: int = 0
    Fp: int = 0
    precision: float = 0.0
    recall: float = 0.0
    aupr: float = 0.0
    f1: float = 0.0
    shd: float = 0.0
    sid: float = 0.0

    def __init__(
            self,
            Tp: int,
            Tn: int,
            Fn: int,
            Fp: int,
            precision: float,
            recall: float,
            aupr: float,
            f1: float,
            shd: int,
            sid: Dict[str, float]
    ):
        self.Tp = Tp
        self.Tn = Tn
        self.Fn = Fn
        self.Fp = Fp
        self.precision = precision
        self.recall = recall
        self.aupr = aupr
        self.f1 = f1
        self.shd = shd
        if self.Fn + self.Fp == 0:
            self.ishd = 0.0
        else:
            self.ishd = 1 - (self.shd / (self.Fn + self.Fp))
        self.sid = sid['sid']
        self.sid_lower = sid['sidLowerBound']
        self.sid_upper = sid['sidUpperBound']

    def __str__(self):
        s = ""
        s += f"Predicted    1     0      Precision: {self.precision:.3g}\n"
        s += f"Actual     -----------    Recall...: {self.recall:.3g}\n"
        s += f"  1       |{self.Tp:^5d} {self.Fn:^5d}|   F1.......: {self.f1:.3g}\n"
        s += f"  0       |{self.Fp:^5d} {self.Tn:^5d}|   AuPR.....: {self.aupr:.3g}\n"
        s += f"           -----------    SHD......: {self.shd:.3g}\n"
        if self.sid_upper == self.sid_lower:
            s += f"                          SID......: {self.sid:.3g}\n"
        else:
            s += f"                          SID......: {self.sid:.3g} [{self.sid_lower}..{self.sid_upper}]\n"
        return s


def _shallow_copy(G: AnyGraph):
    """
    This method returns a shallow copy of the graph G.

    Parameters
    ----------
    G : AnyGraph
        The graph to copy.

    Returns
    -------
    AnyGraph
        The shallow copy of the graph.
    """
    G_new = nx.DiGraph()
    G_new.add_edges_from(G.edges())
    return G_new


def evaluate_graph(
    ground_truth: AnyGraph,
    predicted_graph: AnyGraph,
    feature_names: List = None,
    threshold: float = 0.0,
    absolute: bool = False,
    double_for_anticausal: bool = True,
) -> Metrics:
    """
        This method computes metrics between a pair of graphs: the ground truth and
        the predicted graph. To call this method, simply pass the reference graph, and
        the predicted graph (the one you want to make as much similar to the first one 
        as possible), and all metrics will be computed.

        Parameters
        ----------
        ground_truth : AnyGraph
            The ground truth graph.
        predicted_graph : AnyGraph
            The predicted graph.
        feature_names : List, optional
            The list of feature names, by default None
        threshold : float, optional
            The threshold to use for the precision-recall curve, by default 0.0
        absolute : bool, optional
            Whether to use the absolute value of the weights, by default False
        double_for_anticausal : bool, optional  
            Whether to double the weights of anticausal edges, by default True

        Returns
        -------
        Metrics
            The metrics object containing all the metrics: Tp, Tn, Fn, Fp, precision,
            recall, AuPR, F1, SHD, ISHD, UMI, OMI.
    """
    if ground_truth is None:
        return None

    # If one graphs is directed and the other is not, then consider them both as
    # undirected.
    if (nx.is_directed(ground_truth) and not nx.is_directed(predicted_graph)) or (
            not nx.is_directed(ground_truth) and nx.is_directed(
                predicted_graph)
    ):
        truth = ground_truth.to_undirected() if nx.is_directed(
            ground_truth) else _shallow_copy(ground_truth)
        predicted = predicted_graph.to_undirected() if nx.is_directed(
            predicted_graph) else _shallow_copy(ground_truth)
    else:
        truth = ground_truth.copy()
        predicted = predicted_graph.copy()

    # Checking whether both contain the same nr of nodes.
    # If not, add isolated nodes to the 'g' graph
    missing_nodes = list(set(ground_truth.nodes()) -
                         set(predicted_graph.nodes()))
    if len(missing_nodes) != 0:
        predicted.add_nodes_from(missing_nodes)

    if feature_names is None:
        feature_names = list(predicted_graph.nodes())

    metrics = dict()
    metrics["_Gm"] = _adjacency(truth, feature_names, threshold, absolute)
    metrics["_gm"] = _adjacency(predicted, feature_names, threshold, absolute)
    if _is_weighted(predicted):
        metrics["_preds"] = _weighted_adjacency(
            predicted, feature_names, threshold, absolute
        )
    else:
        metrics["_preds"] = metrics["_gm"].copy()
    metrics["_double_for_anticausal"] = double_for_anticausal

    metrics["Tp"], metrics["Tn"], metrics["Fn"], metrics["Fp"] = _conf_mat(
        truth, predicted, feature_names
    )

    metrics["precision"] = _precision(metrics)
    metrics["recall"] = _recall(metrics)
    metrics["f1"] = _f1(metrics)
    metrics["aupr"] = _aupr(metrics)
    metrics["SHD"] = _SHD(metrics)
    # Cross check if the number of nodes in the predicted graph is the same as the
    # number of nodes in the ground truth graph
    if len(predicted_graph.nodes()) < len(ground_truth.nodes()):
        # Determine what are the nodes missing in the predicted graph
        missing_nodes = list(set(ground_truth.nodes()) -
                             set(predicted_graph.nodes()))
        # Create a copy of the predicted graph
        predicted_graph_copy = predicted_graph.copy()
        # Add the missing nodes to the predicted graph
        predicted_graph_copy.add_nodes_from(missing_nodes)
        predicted_graph_array = nx.to_numpy_array(predicted_graph_copy)
    else:
        predicted_graph_array = nx.to_numpy_array(predicted_graph)
    metrics["SID"] = SID(trueGraph=nx.to_numpy_array(ground_truth),
                         estGraph=predicted_graph_array)

    return Metrics(metrics["Tp"], metrics["Tn"], metrics["Fn"], metrics["Fp"],
                   metrics["precision"], metrics["recall"], metrics["aupr"],
                   metrics["f1"], metrics["SHD"], metrics["SID"])


def _is_weighted(G: AnyGraph) -> bool:
    """
    Returns if a graph is weighted by checking if list of edges has data

    Arguments:
        - G (Graph or DiGraph): a graph or digraph

    Returns:
        True if edges have no data.
    """
    ddicts = map(lambda t: bool(t[2]), list(G.edges().data()))
    return any(ddicts)


def _binary_adj_matrix(
        G: AnyGraph, order: List = None, threshold: float = 0.0, absolute: bool = False
) -> np.ndarray:
    """
    Returns a binary adjacency matrix from a weighted adjacency matrix. If the
    values in the adjacency matrix are greater than the threshold (default 0.0) then
    that value is tarnsformed into 1.0.

    Arguments:
        - G (Graph or DiGraph): Graph or Digraph
        - threshold (float): Min value of weight to be valued as 1 in the binary
            matrix.
        - absolute (bool): Whether performing the comparison of weights against the
            threshold using absolute value. Default is false.

    Returns:
        np.ndarray with the binary version of the weights.
    """
    m = adjacency_matrix(G, nodelist=order).todense()

    def f(m, threshold, absolute):
        if absolute:
            return (np.abs(m) > threshold).astype(np.int16)
        else:
            return (m > threshold).astype(np.int16)

    return f(m, threshold, absolute)


def _adjacency(G: AnyGraph, order: List = None, threshold=0.0, absolute=False):
    """
    Retrieves the adjacency matrix from the graph using NetworkX method if the
    graph is not weighted. This method produces a matrix with 1/0 values only
    since it is used to compute metrics.

    Arguments:
        - G (Graph or DiGraph): the graph to extract the adj. matrix from.
        - order (list): The order of nodes to be used in the adjacency matrix. If
            none specified the node list will be sorted in ascending order.
        - threshold (float): The minimum weight for an edge to be considered as
            present in the adjacency matrix. Default is 0.0.
        - absolute (bool): Default is false. Whether to consider absoute values
            when comparing edge weights agains the threshold.

    Returns:
        numpy.matrix: An adjacency matrix formed by 0s and 1s where the order of
            the nodes is given as argument to the class, and the minimum weight
            must be above the threshold also specified.
    """
    if order is None:
        order = sorted(list(G.nodes()))
    if nx.is_weighted(G):
        result = _binary_adj_matrix(G, order, threshold, absolute)
    else:
        # Scipy adjacency sometimes throws an exception based on type.
        # result = adjacency_matrix(G, nodelist=order).todense()
        nodeset = set(order)
        if nodeset - set(G):
            # delete from order any nodes not in G
            order = [n for n in order if n in G.nodes()]
        adj_matrix = nx.to_numpy_array(G, nodelist=order)
        adj_matrix[np.isnan(adj_matrix)] = 0
        result = adj_matrix

    # If we have symmetric elements, that means that the edge has no direction.
    # for i in range(result.shape[0]):
    #     for j in range(result.shape[1]):
    #         if result[i, j] == 1 and result[j, i] == 1:
    #             result[i, j] = 0
    #             result[j, i] = 0

    return result


def _weighted_adjacency(G: AnyGraph, order: List = None, threshold=0.0, absolute=False):
    if order is None:
        order = sorted(list(G.nodes()))
    F = G.copy()
    def value(x): return abs(x) if absolute else x
    for p, q, w in F.edges(data="weight"):
        if w is None:
            continue
        if value(w) < threshold:
            F.remove_edge(p, q)

    adj_matrix = nx.to_numpy_matrix(F, nodelist=order)
    adj_matrix[np.isnan(adj_matrix)] = 0

    return adj_matrix


def _intersect_matrices(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """
    Computes the intersection of two matrices.

    Parameters:
        mat1: (np.matrix) the first matrix
        mat2: (np.matrix) the second matrix

    Returns:
        np.ndarray: A new matrix with the values in common between the two matrices.
    """
    if not (mat1.shape == mat2.shape):
        raise ValueError("Both graphs must have the same number of nodes")
    mat_intersect = np.where((mat1 == mat2), mat1, 0)

    return mat_intersect


def _positive(matrix: np.matrix) -> np.matrix:
    """
    Returns a matrix where negative values are converted to zero,
    and positive values remain the same
    """
    return (matrix > 0).astype(int)


def _negative(matrix: np.matrix) -> np.matrix:
    """
    Returns a matrix where positive values are converted to zero,
    and negative values remain the same.
    """
    return (matrix < 0).astype(int)


def _conf_mat(truth, est, feature_names):
    if truth.is_directed() and est.is_directed():
        return _conf_mat_directed(truth, est, feature_names)
    else:
        return _conf_mat_undirected(truth, est, feature_names)


def _conf_mat_directed(truth, est, feature_names):
    """
    Computes the confusion matrix for two directed graphs.
    """
    def is_arrow(G, u, v):
        if u not in G.nodes() or v not in G.nodes():
            return False
        return G.has_edge(u, v)  # and not G.has_edge(v, u)

    num_elems = len(feature_names)

    truePositives = np.zeros(
        (num_elems, num_elems)).astype(int)
    estPositives = np.zeros(
        (num_elems, num_elems)).astype(int)
    for i, u in enumerate(feature_names):
        for j, v in enumerate(feature_names):
            if is_arrow(truth, u, v):
                truePositives[i, j] = 1
            if is_arrow(est, u, v):
                estPositives[i, j] = 1

    zeros = np.zeros((num_elems, num_elems))

    Fp = int((np.maximum(estPositives - truePositives, zeros)).sum())
    Fn = int((np.maximum(truePositives - estPositives, zeros)).sum())
    Tp = (np.minimum(truePositives == estPositives, truePositives)).sum()
    Tn = (truePositives == estPositives).sum() - Tp - num_elems

    return Tp, Tn, Fn, Fp


def _conf_mat_undirected(truth, est, feature_names):

    def is_edge(G, u, v):
        if u not in G.nodes() or v not in G.nodes():
            return False
        return G.has_edge(u, v) or G.has_edge(v, u)

    truePositives = np.zeros(
        (len(feature_names), len(feature_names))).astype(int)
    estPositives = np.zeros(
        (len(feature_names), len(feature_names))).astype(int)
    for i, u in enumerate(feature_names):
        for j, v in enumerate(feature_names):
            if is_edge(truth, u, v):
                truePositives[i, j] = 1
            if is_edge(est, u, v):
                estPositives[i, j] = 1

    zeros = np.zeros((len(feature_names), len(feature_names)))

    Tp = int((np.minimum(truePositives == estPositives, truePositives)).sum() / 2)
    Fp = int((np.maximum(estPositives - truePositives, zeros)).sum() / 2)
    Fn = int((np.maximum(truePositives - estPositives, zeros)).sum() / 2)
    Tn = int(((truePositives == estPositives).sum() - Tp) / 2)

    return Tp, Tn, Fn, Fp


def _confusion_matrix(G, metrics):
    def comp(M):
        return 1 - M

    # These are the intersection between the adjacency matrices. The second one
    # is between the complementary matrices
    G_n_g = _intersect_matrices(metrics["_Gm"], metrics["_gm"])
    cG_n_cg = _intersect_matrices(comp(metrics["_Gm"]), comp(metrics["_gm"]))
    N = metrics["_Gm"].shape[0]
    if isinstance(G, nx.DiGraph):
        _Tp = int(G_n_g.sum())
        _Tn = (
            cG_n_cg[np.triu_indices(n=N, k=+1)].sum()
            + cG_n_cg[np.tril_indices(n=N, k=-1)].sum()
        )
        _Fp = int(_negative(metrics["_Gm"] - metrics["_gm"]).sum())
        _Fn = int(_negative(metrics["_gm"] - metrics["_Gm"]).sum())
    elif isinstance(G, nx.Graph):
        _Tp = int(G_n_g.sum() / 2)
        _Tn = cG_n_cg[np.triu_indices(n=N, k=1)].sum()
        _Fp = int(_negative(metrics["_Gm"] - metrics["_gm"]).sum() / 2)
        _Fn = int(_negative(metrics["_gm"] - metrics["_Gm"]).sum() / 2)
    else:
        raise TypeError("Only networkx Graph and DiGraph graphs supported")
    return _Tp, _Tn, _Fn, _Fp


def _precision(metrics):
    if (metrics["Tp"] + metrics["Fp"]) == 0:
        return 0
    else:
        return metrics["Tp"] / (metrics["Tp"] + metrics["Fp"])


def _recall(metrics):
    if (metrics["Tp"] + metrics["Fn"]) == 0.:
        return 0
    else:
        return metrics["Tp"] / (metrics["Tp"] + metrics["Fn"])


def _aupr(metrics):
    true_labels = np.squeeze(np.asarray(metrics["_Gm"].ravel()))
    predicted = np.squeeze(np.asarray(metrics["_preds"].ravel()))
    _sk_precision, _sk_recall, _sk_ths = precision_recall_curve(
        true_labels, predicted)
    return auc(_sk_recall, _sk_precision)


def _f1(metrics):
    if metrics["recall"] + metrics["precision"] == 0.0:
        return 0.0
    else:
        return (
            2
            * (metrics["recall"] * metrics["precision"])
            / (metrics["recall"] + metrics["precision"])
        )


def _SHD(metrics):
    r"""
    Compute the Structural Hamming Distance.
    The Structural Hamming Distance (SHD) is a standard distance to compare
    graphs by their adjacency matrix. It consists in computing the difference
    between the two (binary) adjacency matrixes: every edge that is either
    missing or not in the target graph is counted as a mistake. Note that
    for directed graph, two mistakes can be counted as the edge in the wrong
    direction is false and the edge in the good direction is missing ; the
    `double_for_anticausal` argument accounts for this remark. Setting it to
    `False` will count this as a single mistake.

    Returns:
        int: Structural Hamming Distance (int).
            The value tends to zero as the graphs tend to be identical.
    """
    diff = np.abs(metrics["_Gm"] - metrics["_gm"])
    if metrics["_double_for_anticausal"]:
        _shd = np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1
        _shd = np.sum(diff) / 2

    return int(_shd)
