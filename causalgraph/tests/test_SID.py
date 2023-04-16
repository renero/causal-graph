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

    H1c = np.array([[0, 1, 1, 1, 1],
                    [1, 0, 1, 1, 1],
                    [1, 1, 0, 1, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0]])

    sid1 = SID(G, H1)
    assert sid1['sid'] == 0.0
