"""
This Python version of `selGamBoost` follows the structure and logic of the original
R function. Here are some key points about the translation:

1. We import `numpy` for array operations and assume that `train_GAMboost` is
imported from a separate file.
2. The function signature remains similar, with default values for `pars` and `output`.
3. R's matrix indexing is replaced with NumPy array indexing.
4. The `cat` function for output is replaced with Python's `print` function.
5. List comprehensions and NumPy functions are used to replace some R-specific
    operations.
6. The `xselect()` method is assumed to exist in the model returned by `train_GAMboost`.
You may need to adjust this based on the actual implementation.
7. The boolean indexing and selection logic is adapted to work with NumPy arrays.

Note that this translation assumes that the `train_GAMboost` function in Python returns
an object with similar properties to its R counterpart. You may need to adjust the
code further based on the exact implementation of `train_GAMboost` in Python.
"""
import numpy as np
from .train_GAMboost import train_GAMboost


def selGamBoost(X, pars=None, output=False, k=None):
    """_summary_

    Args:
        X (_type_): _description_
        pars (_type_, optional): _description_. Defaults to None.
        output (bool, optional): _description_. Defaults to False.
        k (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if pars is None:
        pars = {'atLeastThatMuchSelected': 0.02, 'atMostThatManyNeighbors': 10}

    if output:
        print(f"Performing variable selection for variable {k}:")

    result = {}
    X = np.array(X)
    p = X.shape

    if p[1] > 1:
        selVec = [False] * p[1]
        X_without_k = np.delete(X, k, axis=1)
        X_k = X[:, k]
        modfitGam = train_GAMboost(X_without_k, X_k, pars)

        # Replace the xselect() call with feature_importances_
        feature_importances = modfitGam['model'].feature_importances_
        # Indices of features sorted by importance
        cc = np.argsort(feature_importances)[::-1]
        if output:
            print("The following variables")
            print(cc)

        nstep = len(feature_importances)
        howOftenSelected = feature_importances[cc]

        if output:
            print("... have been selected that many times:")
            print(howOftenSelected)

        howOftenSelectedSorted = sorted(howOftenSelected, reverse=True)

        if sum(np.array(howOftenSelected) > pars['atLeastThatMuchSelected']) \
                > pars['atMostThatManyNeighbors']:
            cc = cc[np.array(howOftenSelected) >
                    howOftenSelectedSorted[pars['atMostThatManyNeighbors']]]
        else:
            cc = cc[np.array(howOftenSelected) >
                    pars['atLeastThatMuchSelected']]

        if output:
            print("We finally choose as possible parents:")
            print(cc)
            print()

        tmp = [False] * (p[1] - 1)
        for i in cc:
            tmp[i] = True
        selVec[:k] + tmp + selVec[k+1:]
    else:
        selVec = []

    return selVec
