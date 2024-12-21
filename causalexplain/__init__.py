"""
CausalExplain: A Python package for causal discovery and inference.

This package provides tools for discovering and analyzing causal relationships
in data using various methods and algorithms.
"""

from . import common
from . import estimators
from . import explainability
from . import generators
from . import independence
from . import metrics
from . import models
from .causalexplainer import GraphDiscovery

__all__ = [
    'common',
    'estimators',
    'explainability',
    'generators',
    'independence',
    'metrics',
    'models',
    'GraphDiscovery',
    '__version__',
]

from ._version import __version__
