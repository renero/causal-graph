import colorama
import networkx as nx
import numpy as np
import pandas as pd
import sklearn
import kneed

from sklearn.model_selection import cross_validate
from typing import List


BLACK = colorama.Fore.BLACK
RED = colorama.Fore.RED
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Style.RESET_ALL


def select_features(values,
                    feature_names,
                    method='knee',
                    tolerance=None,
                    sensitivity=1.0,
                    return_shaps=False,
                    descending=False,
                    strict=True,
                    min_impact: float=1e-06,
                    verbose=False) -> List[str]:
    """
    Sort the values and select those before (strict) the point of max. curvature,
    according to the Knee algorithm. If strict is False, the point of max curv. is also 
    selected. When the method is 'abrupt' the selection method is based on taking
    only those feature up-to (down-to) a certain percentage of change in their values.

    Arguments:
        - values (np.ndarray): The values for each of the features. This can 
            be anything that should be used to determine what features are more 
            important than others.
        - feature_names (list): Names of the variables corresponding to the shap values 
        - method: 'knee' (default) or 'abrupt'. The former uses the algorithm in `kneed`
            package, while the later finds an abrupt change greater than `tolerance`
            in the sequence of values.
        - tolerance (float): The percentage of change in value along the sequence of 
            mean shap values that is considered an abrupt change (shaps are normalized 
            to compute this abrupt change). If not specified the tolerance will be
            the determined by the range between max and min divided by the nr. of 
            features. Default value is None.
        - sensitivity (float): The sensitivity to pass to the Kneed algorithm. 
            Default is 1.0, and larger values produce more extreme (higher) values 
            in the curve.
        - return_shaps (bool): Whether returning the mean shap values together with 
            order of the features.
        - descending (bool): Default is False. If True, only applies to `abrupt` method
            and indicates to detect abrupt changes from higher to lower values.
        - strict (bool): Default True, excludes the feature where the elbow/knee is found
        - min_impact (float): Default 1e-06. The minimum impact of a feature to be
            selected. If all features are below this value, none are selected.
        - verbose: guess what.

    """
    shift = 1 if strict else 0
    feature_order = np.argsort(np.sum(np.abs(values), axis=0))
    mean_shap_values = np.abs(values).mean(0)
    sorted_impact_values = [mean_shap_values[idx] for idx in feature_order]

    # In some cases, the mean SHAP values are 0. We return an empty list in that case.
    if np.all(mean_shap_values < min_impact):
        return []

    if tolerance is not None:
        assert tolerance < 1.0 and tolerance > 0.0
    else:
        amplitude = np.max(sorted_impact_values) - np.min(sorted_impact_values)
        tolerance = (amplitude / len(sorted_impact_values)) / amplitude

    if descending:
        sorted_impact_values = sorted_impact_values[::-1]
    if verbose:
        print("Sorted shaps......:", end="")
        print(','.join([f"({f}:{s:.03f})" for f, s in zip(
            feature_names, np.sum(np.abs(values), axis=0))]))
        print(
            f"feature_order.....: {','.join([f'{feature_names[i]}' for i in feature_order])}")
        print(
            f"sorted_mean_values: {','.join([f'{x:.6f}' for x in sorted_impact_values])}")

    if method == 'knee':
        cutoff = kneed.KneeLocator(x=range(len(sorted_impact_values)),
                                   y=sorted_impact_values,
                                   S=sensitivity,
                                   curve='convex', direction='decreasing')
        if cutoff.knee is not None:
            limit_idx = cutoff.knee + \
                shift if cutoff.knee < len(
                    feature_order-1) else len(feature_order-1)
        else:
            # take only the last one.
            limit_idx = len(feature_order) - 2
    elif method == 'abrupt':
        limit_idx = abrupt_change(
            sorted_impact_values, tolerance=tolerance, verbose=verbose)
    else:
        raise ValueError(
            f"Unknown method ({method}). Only 'knee' or 'abrupt' accepted")

    if descending:
        selected_features = list(reversed(
            [feature_names[i] for i in feature_order[-limit_idx:]]))
    else:
        selected_features = list(reversed(
            [feature_names[i] for i in feature_order[limit_idx:]]))

    if verbose:
        print(f"limit_idx..........: {limit_idx}")
        print(f"selected_features..: {selected_features}")
    if return_shaps:
        return selected_features, list(reversed(sorted(mean_shap_values)[limit_idx:]))

    return selected_features

def abrupt_change(X: np.array, tolerance: float = 0.1, verbose=False) -> int:
    """
    Given an array of values in increasing or decreasing order, detect what is the
    element at which an abrupt change of more than `tolerance` is given. The
    tolerance is expressed as a percentage of the range between max and min values
    in the series.

    Arguments:
        - X (np.array): the series of values to detect the abrupt change.
        - tolerance (float): the max percentage of change tolerated.
        - verbose (bool): Verbose output.

    Returns:
        The position in the array where an abrupt change is produced. If there's
            no change in consecutive values greater than the tolerance passed then
            the last element of the array is returned.
    """
    # assert monotonic(X), "The series is not monotonic"
    prev = X[0]
    interval = max(X) - min(X)
    if verbose:
        print(f"Tolerance: {tolerance*100}%")
    for cutoff, x in enumerate(X):
        delta = np.abs((prev - x) / interval)
        if verbose:
            print(f"- pos.{cutoff:02d} ({x:.4f}), ???={delta * 100.0:+.2f}")
        if delta > tolerance:
            break
        prev = x

    return cutoff
