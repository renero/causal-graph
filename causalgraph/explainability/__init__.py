"""
Explainability techniques used for causal discovery.

This module contains various techniques and tools for explaining and 
interpreting causal discovery results:

- shapley: Implements Shapley value-based methods for attributing importance to
  features in causal models.
- regression_quality: Provides metrics and tools for assessing the quality of
  regression models used in causal discovery.
- perm_importance: Implements permutation importance methods for feature
  importance in causal models.
- hierarchies: Contains tools for analyzing and visualizing hierarchical
  structures in causal relationships.

These submodules offer a range of approaches to enhance the interpretability
and understanding of causal discovery results, aiding in the validation and
refinement of causal models.
"""

from . import shapley
from . import regression_quality
from . import perm_importance
from . import hierarchies

__all__ = [
    "shapley",
    "regression_quality",
    "perm_importance",
    "hierarchies",
]
