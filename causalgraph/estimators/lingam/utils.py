"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""
import numpy as np
from sklearn.linear_model import LassoLarsIC, LinearRegression


def find_all_paths(dag, from_index, to_index, min_causal_effect=0.0):
    """Find all paths from point to point in DAG.

    Parameters
    ----------
    dag : array-like, shape (n_features, n_features)
        The adjacency matrix to fine all paths, where n_features is the number of features.
    from_index : int
        Index of the variable at the start of the path.
    to_index : int
        Index of the variable at the end of the path.
    min_causal_effect : float, optional (default=0.0)
        Threshold for detecting causal direction.
        Causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

    Returns
    -------
    paths : array-like, shape (n_paths)
        List of found path, where n_paths is the number of paths.
    effects : array-like, shape (n_paths)
        List of causal effect, where n_paths is the number of paths.
    """
    # Extract all edges
    edges = np.array(np.where(np.abs(np.nan_to_num(dag)) > min_causal_effect)).T

    # Aggregate edges by start point
    to_indices = []
    for i in range(dag.shape[0]):
        adj_list = edges[edges[:, 1] == i][:, 0].tolist()
        if len(adj_list) != 0:
            to_indices.append(adj_list)
        else:
            to_indices.append([])

    # DFS
    paths = []
    stack = [from_index]
    stack_to_indice = [to_indices[from_index]]
    while stack:
        if len(stack) > dag.shape[0]:
            raise ValueError(
                "Unable to find the path because a cyclic graph has been specified."
            )

        cur_index = stack[-1]
        to_indice = stack_to_indice[-1]

        if cur_index == to_index:
            paths.append(stack.copy())
            stack.pop()
            stack_to_indice.pop()
        else:
            if len(to_indice) > 0:
                next_index = to_indice.pop(0)
                stack.append(next_index)
                stack_to_indice.append(to_indices[next_index].copy())
            else:
                stack.pop()
                stack_to_indice.pop()

    # Calculate the causal effect for each path
    effects = []
    for p in paths:
        coefs = [dag[p[i + 1], p[i]] for i in range(len(p) - 1)]
        effects.append(np.cumprod(coefs)[-1])

    return paths, effects


def predict_adaptive_lasso(X, predictors, target, gamma=1.0):
    """Predict with Adaptive Lasso.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    predictors : array-like, shape (n_predictors)
        Indices of predictor variable.
    target : int
        Index of target variable.

    Returns
    -------
    coef : array-like, shape (n_features)
        Coefficients of predictor variable.
    """
    lr = LinearRegression()
    lr.fit(X[:, predictors], X[:, target])
    weight = np.power(np.abs(lr.coef_), gamma)
    reg = LassoLarsIC(criterion="bic")
    reg.fit(X[:, predictors] * weight, X[:, target])
    return reg.coef_ * weight
