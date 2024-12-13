import numpy as np
from sklearn.linear_model import LinearRegression


def train_linear(X, y, pars=None):
    """
    Train a linear regression model.

    Args:
        X (numpy.ndarray): Input features, shape (n_samples, n_features).
        y (numpy.ndarray): Target values, shape (n_samples, 1).
        pars (dict, optional): Additional parameters for the model. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - 'Yfit' (numpy.ndarray): Predicted values, shape (n_samples, 1).
            - 'residuals' (numpy.ndarray): Residuals (y - y_pred), shape (n_samples, 1).
            - 'model' (LinearRegression): Fitted sklearn LinearRegression model.

    Note:
        The coefficients of the model can be accessed via the 'model' key in the 
        returned dictionary, specifically using `result['model'].coef_`.
    """
    if pars is None:
        pars = {}

    mod = LinearRegression().fit(X, y)

    result = {
        'Yfit': mod.predict(X).reshape(-1, 1),
        'residuals': (y - mod.predict(X)).reshape(-1, 1),
        'model': mod
    }

    # For coefficients, use mod.coef_
    return result
