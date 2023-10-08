import numpy as np
import networkx as nx
from causalgraph.metrics.compare_graphs import _conf_mat_directed


# Two identical graphs are compared.
def test_identical_graphs():
    truth = nx.DiGraph()
    truth.add_edge("a", "b")
    truth.add_edge("b", "c")
    est = nx.DiGraph()
    est.add_edge("a", "b")
    est.add_edge("b", "c")
    feature_names = ["a", "b", "c"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 2
    assert fp == 0
    assert tn == 4
    assert fn == 0

# Two graphs, same nodes, with one edge different are compared.
def test_one_edge_different():
    truth = nx.DiGraph()
    truth.add_edge("a", "b")
    truth.add_edge("b", "c")
    est = nx.DiGraph()
    est.add_edge("a", "b")
    est.add_edge("c", "a")
    feature_names = ["a", "b", "c"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 1
    assert fp == 1
    assert tn == 3
    assert fn == 1

# Two graphs with different edges are compared.
def test_different_edges():
    truth = nx.DiGraph()
    truth.add_edge("a", "b")
    truth.add_edge("b", "c")
    est = nx.DiGraph()
    est.add_edge("a", "b")
    est.add_edge("a", "c")
    feature_names = ["a", "b", "c"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 1
    assert fp == 1
    assert tn == 3
    assert fn == 1

# Two graphs with different nodes are compared.
def test_different_nodes_est_more_connections():
    truth = nx.DiGraph()
    truth.add_edge("a", "b")
    truth.add_edge("b", "c")
    truth.add_node("d")
    est = nx.DiGraph()
    est.add_edge("a", "b")
    est.add_edge("b", "c")
    est.add_edge("c", "d")
    feature_names = ["a", "b", "c", "d"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 2
    assert tn == 9
    assert fp == 0
    assert fn == 1

# Two graphs with different nodes (truth has more connection) are compared.
def test_different_nodes_truth_more_connections():
    truth = nx.DiGraph()
    truth.add_edge("a", "b")
    truth.add_edge("b", "c")
    truth.add_edge("c", "d")
    est = nx.DiGraph()
    est.add_edge("a", "b")
    est.add_edge("b", "c")
    est.add_node("d")
    feature_names = ["a", "b", "c", "d"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 2
    assert fp == 1
    assert tn == 9
    assert fn == 0


# Two graphs with one node and no edges are compared.
def test_one_node_no_edges():
    truth = nx.DiGraph()
    truth.add_node("a")
    est = nx.DiGraph()
    est.add_node("a")
    feature_names = ["a"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 0
    assert fp == 0
    assert tn == 0
    assert fn == 0

# Two graphs with one node and a self-loop are compared.
def test_one_node_self_loop():
    truth = nx.DiGraph()
    truth.add_edge("a", "a")
    est = nx.DiGraph()
    est.add_edge("a", "a")
    feature_names = ["a"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 1
    assert fp == 0
    assert tn == 0
    assert fn == 0

# Two graphs with multiple nodes and no edges are compared.
def test_multiple_nodes_no_edges():
    truth = nx.DiGraph()
    truth.add_nodes_from(["a", "b", "c"])
    est = nx.DiGraph()
    est.add_nodes_from(["a", "b", "c"])
    feature_names = ["a", "b", "c"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 0
    assert fp == 0
    assert tn == 6
    assert fn == 0