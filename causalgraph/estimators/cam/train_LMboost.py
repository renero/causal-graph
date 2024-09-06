"""
Here's an explanation of the changes:

1. We import `numpy` for array operations and `GradientBoostingRegressor` from 
scikit-learn as an equivalent to R's `glmboost`.
2. The function signature is similar, but we use `None` as the default for `pars` 
instead of an empty list.
3. We convert inputs to numpy arrays to ensure compatibility.
4. We center `y` by subtracting its mean.
5. We create and fit a `GradientBoostingRegressor`, which is similar to `glmboost` in R.
6. We create a dictionary `result` with the fitted values, residuals, and the model 
itself.
7. The `center=TRUE` parameter in the R version is not needed as scikit-learn's 
`GradientBoostingRegressor` handles feature centering internally.

Note that this Python version might not be exactly equivalent to the R version, 
as there could be differences in the underlying algorithms and default parameters.
You may need to adjust the `GradientBoostingRegressor` parameters to match the 
behavior of `glmboost` more closely if needed.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def train_LMboost(X, y, pars=None):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        pars (dict, optional): Parameters

    Returns:
        dict: result
    """
    # Convert inputs to numpy arrays if they're not already
    X = np.array(X)
    y = np.array(y)

    # Center y
    y = y - np.mean(y)

    # Create and fit the model
    gb = GradientBoostingRegressor()
    gb.fit(X, y)

    # Prepare the result
    result = {
        'Yfit': gb.predict(X),
        'residuals': y - gb.predict(X),
        'model': gb
    }

    return result
