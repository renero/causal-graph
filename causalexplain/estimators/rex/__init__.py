"""
REX (Regression with Explainability) estimator module.

This module provides the REX estimator, which uses regression models and 
explainability techniques to discover causal relationships in data.
"""

from .rex import Rex
from .knowledge import Knowledge

__all__ = [
    'Rex',
    'Knowledge'
]
