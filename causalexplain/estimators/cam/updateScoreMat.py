"""
1. **Imports**: Imported necessary modules and the `compute_score_mat_parallel`
function.
2. **Function Definition**: Translated the R function to Python, maintaining the same
logic and structure.
3. **Matrix Operations**: Used NumPy for matrix operations.
4. **Parallel Processing**: Used Python's `multiprocessing.Pool` for parallel
processing, similar to R's `mcmapply`.

Make sure to have the `compute_score_mat_parallel` function defined in a file named
`compute_score_mat_parallel.py`.
"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name, W0221:arguments-differ
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, E0401:import-error
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

from multiprocessing import Pool

import numpy as np

from .computeScoreMatParallel import computeScoreMatParallel


def updateScoreMat(
        score_mat,
        X,
        score_name,
        i,
        j,
        score_nodes,
        adj,
        verbose,
        num_cores,
        max_num_parents,
        pars_score,
        interv_mat,
        interv_data):
    """_summary_

    Args:
        score_mat (_type_): _description_
        X (_type_): _description_
        score_name (_type_): _description_
        i (_type_): _description_
        j (_type_): _description_
        score_nodes (_type_): _description_
        adj (_type_): _description_
        output (_type_): _description_
        num_cores (_type_): _description_
        max_num_parents (_type_): _description_
        pars_score (_type_): _description_
        interv_mat (_type_): _description_
        interv_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    # new edge: from i to j

    p = X.shape[1]
    if verbose:
        print(f"p: {p}")
        print("Adjacency matrix---")
        for r in range(adj.shape[0]):
            for c in range(p):
                print(f"{adj[r, c]} ", end="")
            print()

    existing_par_of_j = np.where(adj[:, j] == 1)[0]
    not_allowed_par_of_j = np.setdiff1d(
        np.where(score_mat[:, j] == -np.inf)[0], np.append(existing_par_of_j, j))
    if verbose:
        print(f"existing_par_of_j: {existing_par_of_j}")
        print(f"not_allowed_par_of_j: {not_allowed_par_of_j}")

    # if there is something left that we need to update
    if len(existing_par_of_j) + len(not_allowed_par_of_j) < p - 1:
        # update column for j
        row_parents = np.full((p, len(existing_par_of_j) + 1), np.nan)
        row_parents[:, :len(existing_par_of_j)] = existing_par_of_j
        row_parents[:, len(existing_par_of_j)] = np.arange(p)
        to_update = np.setdiff1d(np.arange(p), np.append(
            [j], np.append(existing_par_of_j, not_allowed_par_of_j)))

        if len(existing_par_of_j) < max_num_parents:
            if num_cores == 1:
                score_update = [
                    computeScoreMatParallel(
                        row_parents=row_parents,
                        sel_mat=np.ones((p, p), dtype=bool),
                        score_name=score_name, X=X, verbose=verbose, node2=j,
                        pars_score=pars_score, interv_mat=interv_mat,
                        interv_data=interv_data, i=idx) \
                            for idx in to_update
                    ]
            else:
                with Pool(num_cores) as pool:
                    score_update = pool.starmap(
                        computeScoreMatParallel, [
                            (row_parents, np.ones((p, p), dtype=bool),
                             score_name, X, verbose, j, pars_score, interv_mat,
                             interv_data, idx) \
                                 for idx in to_update
                        ])
        else:
            score_update = -np.inf

        score_mat[to_update, j] = score_update - score_nodes[j]

    return score_mat
