"""
This Python version accomplishes the same task as the R function:

1. It uses `LassoCV` from scikit-learn to perform cross-validation and find the 
optimal regularization parameter (lambda in R, alpha in Python).
2. It then trains a final Lasso model using the optimal alpha.
3. The function returns a dictionary containing the fitted values, residuals, and 
the trained model.

Note that:
- The `cv.glmnet` in R is replaced by `LassoCV` in Python.
- The `glmnet` in R is replaced by `Lasso` in Python.
- In scikit-learn, the regularization parameter is called `alpha` instead of `lambda`.
- The `pars` parameter is kept for consistency, but it's not used in this 
implementation. You can extend the function to use additional parameters if needed.
- The cross-validation is set to 5-fold (you can adjust this if needed).
- A random state is set for reproducibility.

This Python version should provide equivalent functionality to the original R function.
"""
from typing import Dict, Any
from sklearn.linear_model import LassoCV, Lasso
import numpy as np


def train_lasso(X, y, pars=None) -> Dict[str, Any]:
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        pars (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Perform cross-validation to find optimal lambda
    cv_model = LassoCV(cv=5, random_state=42)
    cv_model.fit(X, y)

    # Train the final model using the optimal lambda (alpha in sklearn)
    mod = Lasso(alpha=cv_model.alpha_)
    mod.fit(X, y)

    # Prepare results
    result = {
        'Yfit': mod.predict(X),
        'residuals': y - mod.predict(X),
        'model': mod
    }

    return result
