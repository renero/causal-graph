import numpy as np
import pytest

from causalgraph.metrics.SID import SID


def test_SID():
    G = np.array([[0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    H1 = np.array([[0, 1, 1, 1, 1],
                   [0, 0, 1, 1, 1],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])
    H2 = np.array([[0, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])
    H3 = np.array([[0, 1, 1, 1, 1],
                   [1, 0, 1, 1, 1],
                   [1, 1, 0, 1, 0],
                   [1, 1, 1, 0, 0],
                   [1, 1, 0, 0, 0]])
    H4 = np.array([[0, 0, 1, 1, 1],
                   [0, 0, 0, 1, 0],
                   [1, 1, 0, 1, 0],
                   [0, 1, 0, 0, 1],
                   [1, 0, 1, 0, 0]])
    H5 = np.array([[0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])

    sid1 = SID(G, H1)
    assert sid1['sid'] == 0.0
    sid2 = SID(G, H2)
    assert sid2['sid'] == 8.0
    sid3 = SID(G, H3)
    assert sid3['sidLowerBound'] == 0.0
    assert sid3['sidUpperBound'] == 15.0
    sid4 = SID(G, H4)
    assert sid4['sidLowerBound'] == 8.0
    assert sid4['sidUpperBound'] == 16.0
    sid5 = SID(G, H5)
    assert sid5['sid'] == 12.0
