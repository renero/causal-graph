"""
- **Function Definition**: Translated the R function `computeScoreMat` to Python as
`compute_score_mat`.
- **Combinations**: Used `itertools.combinations` to generate parent combinations.
- **DataFrame**: Used `pandas.DataFrame` to create the grid of indices.
- **Parallel Processing**: Used `multiprocessing.Pool` for parallel processing.
- **Variance Calculation**: Used `numpy.var` to calculate variance and adjusted the
score matrix accordingly.

Make sure the `compute_score_mat_parallel` function is correctly defined in
`computeScoreMatParallel.py`. If you need further assistance with that function,
please provide its R code.
"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name, W0221:arguments-differ
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, E0401:import-error
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches


from itertools import combinations
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .computeScoreMatParallel import computeScoreMatParallel


def computeScoreMat(
        X,
        score_name,
        num_parents,
        verbose,
        num_cores,
        sel_mat,
        pars_score,
        interv_mat,
        interv_data):
    """_summary_

    Args:
        X (_type_): _description_
        score_name (_type_): _description_
        num_parents (_type_): _description_
        output (_type_): _description_
        num_cores (_type_): _description_
        sel_mat (_type_): _description_
        pars_score (_type_): _description_
        interv_mat (_type_): _description_
        interv_data (_type_): _description_

    Returns:
        _type_: _description_
    """

    p = X.shape[1]
    n = X.shape[0]
    row_parents = np.array(
        list(combinations(range(p), num_parents)), dtype=int)

    # XXX
    if verbose:
        print(f". p: {p}")
        print(f". n: {n}")
        print(f". row_parents: {row_parents.flatten()}")

    tt = pd.DataFrame(list(itertools.product(
        range(0, row_parents.shape[0]), range(0, p))),
        columns=['i', 'j'])
    all_node2 = tt['i'].values
    all_i = tt['j'].values

    if num_cores == 1:
        score_mat = np.array(
            [computeScoreMatParallel(
                row_parents, score_name, X, sel_mat, verbose, node2,
                i, pars_score, interv_mat, interv_data)
             for node2, i in zip(all_node2, all_i)])
    else:
        with Pool(num_cores) as pool:
            score_mat = np.array(pool.starmap(computeScoreMatParallel, [(
                row_parents, score_name, X, sel_mat, verbose, node2, i,
                pars_score, interv_mat, interv_data)
                for node2, i in zip(all_node2, all_i)]))

    score_mat = score_mat.reshape(len(row_parents), p)

    init_score = np.empty(p)
    for i in range(p):
        if interv_data:
            X2 = X[~interv_mat[:, i], :]
        else:
            X2 = X
        vartmp = np.var(X2[:, i])
        init_score[i] = -np.log(vartmp)
        score_mat[:, i] -= init_score[i]

    return {'scoreMat': score_mat,
            'rowParents': row_parents,
            'scoreEmptyNodes': init_score}
