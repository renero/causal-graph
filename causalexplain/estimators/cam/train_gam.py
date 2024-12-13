"""
Key differences and notes:

- We use NumPy arrays instead of R matrices.
- The pygam library is used instead of R's gam function.
- The formula creation is different. In pygam, we create a list of smooth terms.
- Error handling is done with a try-except block instead of R's try().
- The df, edf, and edf1 calculations are approximations, as pygam doesn't provide
    exact equivalents to R's GAM implementation.
- The function signature includes type hints for better code clarity.

To use this function, you'll need to install the required libraries: pygam

This Python version should provide similar functionality to the R version,
but there might be some differences in the exact numerical results due to the
different implementations of GAM in R and Python.
"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name, W0221:arguments-differ
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, E0401:import-error
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

from typing import Any, Dict

import numpy as np
from pygam import GAM, terms
from statsmodels.gam.api import BSplines, GLMGam


def _build_gam_formula(p, num_splines, lam=False):
    """
    Create a pyGAM formula for a GAM model with 'p' smooth terms, each with
    'num_splines' basis functions. If 'lam' is True, the smooth terms are
    also regularized with a penalty term of 0, which is different from
    R's `gam` function.

    Parameters
    ----------
    p : int
        The number of smooth terms in the GAM model.
    num_splines : int
        The number of basis functions for each smooth term.
    lam : bool, optional
        Whether to add a penalty term to each smooth term. Defaults to
        False.

    Returns
    -------
    formula : pyGAM formula
        The pyGAM formula for the GAM model.
    """
    gam_formula = None
    for i in range(p):
        if gam_formula is None:
            if not lam:
                gam_formula = terms.s(i, n_splines=num_splines)
            else:
                gam_formula = terms.s(i, n_splines=num_splines, lam=0)
        else:
            if not lam:
                gam_formula = gam_formula + \
                    terms.s(i, n_splines=num_splines)
            else:
                gam_formula = gam_formula + \
                    terms.s(i, n_splines=num_splines, lam=0)

    return gam_formula


def train_gam(
        X: np.ndarray,
        y: np.ndarray,
        pars: Dict[str, Any] = None,
        verbose: bool=False) -> Dict[str, Any]:
    """
    Train a Generalized Additive Model using pyGAM.

    Parameters
    ----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target variable.
    pars : Dict[str, Any], optional
        Model parameters. Defaults to None.

    Returns
    -------
    result : Dict[str, Any]
        Model results.
    """
    if pars is None:
        pars = {"num_basis_fcts": 10}
    if "num_basis_fcts" not in pars:
        pars["num_basis_fcts"] = 10

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)

    if verbose:
        print(f". . . . Fitting GAM on {X.shape[1]} variables")
        print(". . . . . First 8 elements of each column of X:")
        for col in range(X.shape[1]):
            print(f". . . . . . Column {col}: {X[:8, col]}")
        print(f". . . . . First 8 elements of Y: {y[:8].flatten()}")

    n, p = X.shape

    if n / p < 3 * pars["num_basis_fcts"]:
        pars["num_basis_fcts"] = int(np.ceil(n / (3 * p)))
        print(
            f"Changed number of basis functions to {pars['num_basis_fcts']} in order "
            f"to have enough samples per basis function") if verbose else None
    gam_formula = _build_gam_formula(p, pars["num_basis_fcts"])
    try:
        expr = ""
        for i in range(p):
            expr += f"s({i})"
            if i == p-1:
                break
            expr += " + "
        print(f". . . . . Fitting GAM: {expr}") if verbose else None
        model = GAM(gam_formula)
        model.fit(X, y)
    except Exception:
        print(". . . . . There was some error with GAM. The smoothing parameter is set to zero.")
        gam_formula = _build_gam_formula(p, pars["num_basis_fcts"], lam=True)
        model = GAM(gam_formula)
        model.fit(X, y)

    result = {
        "Yfit": model.predict(X).reshape(-1, 1),
        "residuals": (y - model.predict(X)).reshape(-1, 1),
        "model": model,
        # Approximate
        "df": model.statistics_["n_samples"] - model.statistics_["edof"],
        "edf": model.statistics_["edof"],
        # pygam doesn't have a direct equivalent to edf1
        "edf1": model.statistics_["edof"],
        "p_values": model.statistics_["p_values"]
    }

    return result


# ... existing code ...


def train_gam_sm(
        X: np.ndarray,
        y: np.ndarray,
        pars: Dict[str, Any] = None) -> Dict[str, Any]:
    """Train a Generalized Additive Model using statsmodels.gam.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target variable.
        pars (Dict[str, Any], optional): Model parameters. Defaults to None.

    Returns:
        Dict[str, Any]: Model results.
    """

    if pars is None:
        pars = {"num_basis_fcts": 10}
    if "num_basis_fcts" not in pars:
        pars["num_basis_fcts"] = 10

    y = y.reshape(-1, 1)  # Change this line to flatten y

    print(f". . . . Fitting GAM on {X.shape[1]} variables")
    print(". . . . . First 8 elements of each column of X:")
    for col in range(X.shape[1]):
        print(f". . . . . . Column {col}: {X[:8, col]}")
    print(f". . . . . First 8 elements of Y: {y[:8].flatten()}")

    n, p = X.shape

    if n / p < 3 * pars["num_basis_fcts"]:
        pars["num_basis_fcts"] = int(np.ceil(n / (3 * p)))
        print(
            f"Changed number of basis functions to {pars['num_basis_fcts']} in order "
            f"to have enough samples per basis function")

    # Create B-spline basis
    basis = BSplines(X, df=pars["num_basis_fcts"] * p, degree=[3] * p)

    try:
        mod_gam = GLMGam(y, smoother=basis)
        mod_gam.fit()
    except Exception:
        print(". . . . . There was some error with GAM. The smoothing parameter is set to zero.")
        # Adjust basis for zero smoothing (if applicable)
        basis = BSplines(X, df=[1] * p, degree=[3] * p)
        mod_gam = GLMGam(y, smoother=basis)
        mod_gam.fit()

    result = {
        # Ensure Yfit is reshaped correctly
        "Yfit": mod_gam.predict(X),
        # Ensure residuals are reshaped correctly
        "residuals": (y - mod_gam.predict(X)).reshape(-1, 1),
        "model": mod_gam,
        # Approximate
        "df": mod_gam.df_resid,
        "edf": mod_gam.df_model,
        # pygam doesn't have a direct equivalent to edf1
        "edf1": mod_gam.df_model
    }

    return result
