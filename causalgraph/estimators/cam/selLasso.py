"""
Key changes and explanations:

1. Imported `numpy` for array operations and `train_lasso` from `train_lasso.py`.
2. Changed function signature to use Python conventions (e.g., `None` as default
for `pars`).
3. Used f-string for formatted output.
4. Converted `X` to a numpy array for easier indexing and shape retrieval.
5. Adjusted indexing to account for Python's 0-based indexing (e.g., `X[:, :k]`
instead of `X[,-k]`).
6. Implemented the selection vector creation using list comprehensions and boolean
operations.
7. Adjusted the `selVec` assignment to account for Python's slicing behavior.

Note that this translation assumes that the `train_lasso` function in `train_lasso.py`
returns a dictionary with a nested 'model' dictionary containing a 'beta' list.
You may need to adjust the `train_lasso` call and result handling if its
implementation differs from this assumption.
"""
import numpy as np

from .train_lasso import train_lasso


def selLasso(X, pars=None, output=False, k=None):
    if output:
        print(f"Performing variable selection for variable {k}:")

    result = {}
    X = np.asarray(X)
    p = X.shape

    if p[1] > 1:
        selVec = [False] * p[1]
        modfitGam = train_lasso(
            X[:, :k].tolist() + X[:, k+1:].tolist(), X[:, k].tolist(), pars)
        selVecTmp = [False] * (p[1] - 1)

        # Access the coefficients from the Lasso model
        coefficients = modfitGam['model'].coef_
        for i, coef in enumerate(coefficients):
            if coef != 0:
                selVecTmp[i] = True

        selVec[:k] = selVecTmp[:k]
        selVec[k+1:] = selVecTmp[k:]
    else:
        selVec = []

    return selVec
