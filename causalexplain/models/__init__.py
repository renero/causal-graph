"""
CausalExplain Models
==================

This module contains the models for causal discovery methods. All the models
are implemented in the scikit-learn style, with a `fit` method to fit the model
to the data, a `predict` method to make predictions, and a `score` method to
evaluate the model performance.

The models are:

- `GBTRegressor`: Gradient Boosting Trees Regressor
- `NNRegressor`: Neural Network Regressor
"""

from .gbt import GBTRegressor
from .dnn import NNRegressor

__all__ = [
    'GBTRegressor',
    'NNRegressor',
]