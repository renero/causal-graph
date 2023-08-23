import numpy as np
import networkx as nx
from causalgraph.metrics.compare_graphs import _conf_mat_directed


def test_conf_mat_directed():
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
