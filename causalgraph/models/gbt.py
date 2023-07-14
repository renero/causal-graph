import types
import warnings
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from tqdm.auto import tqdm

from causalgraph.common import GRAY, GREEN, RESET, tqdm_params
from causalgraph.models._models import MLPModel

# import sys
# sys.path.append("../dnn")



warnings.filterwarnings("ignore")

# TODO: Make this class to inherit from a generic class for all regressors


class GBTRegressor(BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        pass

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        return super().__repr__(N_CHAR_MAX)