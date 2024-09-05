"""
This Python translation aims to maintain the functionality of the original R code. 
Here are some key points about the translation:

- The function signature remains similar, with default values for `pars` and `output`.
- We use numpy arrays instead of R matrices.
- The `xselect()` method is assumed to be part of the model returned by 
  `train_LMboost`. You may need to adjust this based on the actual implementation.
- List comprehensions and numpy operations are used to replace some R-specific 
  operations.
- The indexing is adjusted to account for Python's 0-based indexing (compared to R's 
  1-based indexing).

Note that this translation assumes that the `train_LMboost` function in Python returns 
a dictionary with a 'model' key, which has an `xselect()` method. You may need to 
adjust this part based on the actual implementation of `train_LMboost` in Python.
"""
import numpy as np

from Python.CAM.train_LMboost import train_LMboost


def selLmBoost(X, pars=None, output=False, k: int = None):
    if pars is None:
        pars = {'atLeastThatMuchSelected': 0.02, 'atMostThatManyNeighbors': 10}

    if output:
        print(f"Performing variable selection for variable {k}:")

    X = np.array(X)
    p = X.shape

    if p[1] > 1:
        selVec = [False] * p[1]

        modfit_lm = train_LMboost(np.delete(X, k, axis=1), X[:, k], pars)
        # Get the indices of the important features
        cc = np.argsort(modfit_lm['model'].feature_importances_)[::-1]

        if output:
            print("The following variables")
            print(cc)

        # Create a list to store the importance of each feature in 'cc'
        how_often_selected = modfit_lm['model'].feature_importances_

        if output:
            print("... have been selected that many times:")
            print(how_often_selected)

        how_often_selected_sorted = sorted(how_often_selected, reverse=True)

        if sum(np.array(how_often_selected) > pars['atLeastThatMuchSelected']) \
                > pars['atMostThatManyNeighbors']:
            cc = cc[np.array(how_often_selected) >
                    how_often_selected_sorted[pars['atMostThatManyNeighbors']]]
        else:
            cc = cc[np.array(how_often_selected) >
                    pars['atLeastThatMuchSelected']]

        if output:
            print("We finally choose as possible parents:")
            print(cc)
            print()

        tmp = [False] * (p[1] - 1)
        for idx in cc:
            tmp[idx] = True

        selVec[:k] = tmp[:k]
        selVec[k+1:] = tmp[k:]
    else:
        selVec = []

    return selVec
