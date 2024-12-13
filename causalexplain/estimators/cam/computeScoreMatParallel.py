"""
- R's `cat` is replaced with Python's `print`.
- R's `stop` is replaced with Python's `raise` for exceptions.
- R's `!` is replaced with Python's `~` for logical negation.
- R's `prod` is replaced with `np.prod` from NumPy.
- R's `var` is replaced with `np.var` from NumPy.
"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name, W0221:arguments-differ
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, E0401:import-error
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches


import numpy as np
from causalexplain.estimators.cam.train_gam import train_gam
from causalexplain.estimators.cam.train_linear import train_linear


def computeScoreMatParallel(
        row_parents,
        score_name,
        X,
        sel_mat,
        verbose,
        node2,
        i,
        pars_score,
        interv_mat,
        interv_data):

    # the i-th row of row_parents contains possible parents of node2 (we call
    # them "parents_to_check")
    parents_to_check = row_parents[i, :]
    if verbose:
        print(f"parents_to_check: {parents_to_check.flatten()}")

    if verbose:
        print(
            f". Compute score entry for regressing {node2} on {parents_to_check}")

    if interv_data:
        X2 = X[~interv_mat[:, node2], :]
    else:
        X2 = X

    parents_to_check = parents_to_check.astype(int)
    if node2 not in parents_to_check and \
            np.prod(sel_mat[parents_to_check, node2]) == 1:
        if verbose:
            print(". . node2 not in parentsToCheck AND all parents are selected")
            print(
                f". . . '{node2}' not in parents_to_check ({parents_to_check}) OR\n"
                f". . . selMat[parentsToCheck, node2] = "
                f"{sel_mat[parents_to_check, node2]}")
        if score_name == "SEMSEV":
            raise ValueError("This score does not work. It does not decouple.")
        elif score_name == "SEMIND":
            raise NotImplementedError("NOT IMPLEMENTED YET")
        elif score_name == "SEMGAM":
            if verbose:
                print(f". . . parentsToCheck: {parents_to_check.flatten()}\n"
                    f". . . node2: {node2}")
            mod_gam = train_gam(X2[:, parents_to_check],
                                X2[:, node2], pars=pars_score, verbose=verbose)
            score = -np.log(np.var(mod_gam['residuals']))
            print(f". . . score: {score}") if verbose else None
        elif score_name == "SEMLIN":
            mod_gam = train_linear(X2[:, parents_to_check], X2[:, node2])
            score = -np.log(np.var(mod_gam['residuals']))
        else:
            raise ValueError("I do not know this score function.")
    else:
        print(". . NOT matching\n"
              f". . . '{node2}' not in? parents_to_check ({parents_to_check}) "
              f"{node2 not in parents_to_check} OR\n"
              f". . . selMat[parentsToCheck, node2] = "
              f"{sel_mat[parents_to_check, node2]}") if verbose else None
        score = -np.inf

    return score
