"""
Here are the main changes and explanations:

- We use `numpy` for array operations and `scipy.stats` for linear regression.
- The default parameter `pars` is set to `None` and then initialized if not provided.
- We use f-strings for string formatting in the print statement.
- The input `X` is converted to a numpy array.
- Instead of using `lm`, we use `scipy.stats.linregress` for linear regression.
- We manually add a constant term to `X` for the intercept in the regression.
- The p-values are extracted directly from the `linregress` result.
- The selection vector is updated using list slicing to exclude the k-th element.

Note that this Python version assumes that the input `X` is a 2D array-like object. 
The function will work similarly to the R version, but there might be slight 
differences in the exact numerical results due to different underlying implementations 
of the linear regression.
"""
import numpy as np
from scipy import stats


def selLm(X, pars=None, output=False, k: int = None):
    """_summary_

    Args:
        X (np.ndarray): a 2D numpy array with the variables
        pars (dict): Parameters
        output (bool, optional): _description_. Defaults to False.
        k (int, optional): The index of the variable

    Returns:
        _type_: _description_
    """
    if pars is None:
        pars = {'cut_off_p_val': 0.001}

    if output:
        print(f"Performing variable selection for variable {k}:")

    X = np.array(X)
    p = X.shape

    if p[1] > 1:
        sel_vec = [False] * p[1]
        x = np.delete(X, k, axis=1)
        y = X[:, k]

        # Perform linear regression
        X_with_const = np.column_stack((np.ones(len(x)), x))
        model = stats.linregress(X_with_const, y)

        # Calculate p-values for coefficients (excluding intercept)
        p_val_vec = model.pvalue[1:]

        # Update sel_vec based on p-values
        # We need to assign the boolean values to sel_vec, excluding the k-th element
        # First, we create a boolean array from p_val_vec
        significant = p_val_vec < pars['cut_off_p_val']

        # Then, we assign these values to sel_vec, skipping the k-th element
        sel_vec[:k] = significant[:k]
        sel_vec[k+1:] = significant[k:]
    else:
        sel_vec = []

    return sel_vec
