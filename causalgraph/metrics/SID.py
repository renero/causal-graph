# Structural Intervention Distance (SID) metric
#
# Based on original R implementation by Jonas Peters
# https://rdrr.io/cran/SID/f/
# (C) Python Version by J. Renero, 2023
#

from typing import Dict
import numpy as np
from gadjid import sid


def SID(trueGraph: np.ndarray, estGraph: np.ndarray) -> Dict[str, float]:
    # Check if estGraph is a DAG
    if not np.all(np.linalg.eig(estGraph)[0] > 0):
        return {
            'sid': 0.0,
            'sidLowerBound': 0.0,
            'sidUpperBound': 0.0
        }

    s = sid(trueGraph, estGraph, edge_direction="from row to column")
    return {
        'sid': s[1],
        'sidLowerBound': s[0],
        'sidUpperBound': s[0]
    }
