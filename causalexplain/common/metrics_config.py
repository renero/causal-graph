"""
Configuration for metric types used across the causalexplain package.
"""

# Non-causal metric types
global_nc_metric_types = [
    'r2', 'mse', 'rmse', 'mae', 'mape', 'medae',
    'evs', 'maxe', 'msle', 'male', 'mda', 'gmae'
]

# Causal metric types
global_metric_types = [
    'shd', 'precision', 'recall', 'f1', 'fdr', 'tpr',
    'fpr', 'nnz', 'extra', 'missing', 'reversed',
    'pred_size', 'pred_time', 'diff_edges'
]

__all__ = ['global_nc_metric_types', 'global_metric_types']
