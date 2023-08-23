import numpy as np

from causalgraph.metrics.compare_graphs import _conf_mat_directed


def test_conf_mat_directed():
    truth = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    est = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    feature_names = ["a", "b", "c"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 2
    assert tn == 1
    assert fn == 0
    assert fp == 0
