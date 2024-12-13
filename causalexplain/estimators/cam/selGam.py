"""
This Python version aims to replicate the functionality of the R function.
Here are some key points about the translation:

1. We use NumPy for array operations.
2. Instead of `gam` from R, we use `pygam` library which provides similar
    functionality in Python.
3. The p-values are extracted from the fitted GAM model's statistics.
4. The logic for creating and updating `selVec` is adjusted to work with Python's
    0-based indexing.

Note that this translation assumes that the `pygam` library is installed and imported.
You may need to install it using `pip install pygam`.

Also, be aware that there might be some differences in the exact implementation
details between R's `gam` and Python's `pygam`. You may need to fine-tune the GAM
model creation and fitting process to match the exact behavior of the R version.
"""
import numpy as np

from causalexplain.estimators.cam.train_gam import train_gam


def selGam(X, pars=None, verbose=False, k=None):
    """
    This method selects features based on GAM p-values. It returns a vector
    of selected features whose p-values are less than the cutOffPVal.

    Args:
        X (_type_): _description_
        pars (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.
        k (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if pars is None:
        pars = {'cutOffPVal': 0.001, 'numBasisFcts': 10}

    X = np.asarray(X)
    p = X.shape

    if p[1] > 1:
        selVec = [False] * p[1]
        X_without_k = np.delete(X, k-1, axis=1)
        y = X[:, k-1]

        # Use train_gam function instead of directly creating and fitting GAM
        gam_result = train_gam(X_without_k, y, pars=pars, verbose=verbose)

        # Extract p-values from the gam_result.
        # PyGAM returns the p-values of all predictors, followed by the p-value of
        # the intercept, which we don't need.
        pValVec = gam_result['p_values']
        pValVec = pValVec[:-1]


        if len(pValVec) != len(selVec) - 1:
            print("This should never happen (function selGam).")

        selVec_without_k = [p < pars['cutOffPVal'] for p in pValVec]
        selVec[:k] = selVec_without_k[:k]
        selVec[k+1:] = selVec_without_k[k:]
        if verbose:
            print(f". . . . . SelGAM(k={k})")
            print(f". . . . . . Vector of p-values: {pValVec}")
            print(f". . . . . . Selected indices: {selVec_without_k}")

    else:
        selVec = []

    return selVec
