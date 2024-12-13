# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

import numpy as np

from causalexplain.independence.feature_selection import (find_cluster_change_point,
                                                        select_features)


class TestClusterChange:
    """
    This class tests the functionality of the cluster_change function.
    """

    def test_increasing_values(self):
        """
        Given a list of increasing values, the function returns the index of
        the last element.
        """
        values = [1, 2, 3, 4, 5]
        result = find_cluster_change_point(values)
        assert result is None

    def test_decreasing_values(self):
        """
        Given a list of decreasing values, the function returns the index of
        the last element.
        """
        values = [5, 4, 3, 2, 1]
        result = find_cluster_change_point(values)
        assert result is None

    def test_single_cluster(self):
        """
        Given a list of values with a single cluster, the function returns the
        index of the last element.
        """
        values = [1, 1, 1, 1, 1]
        result = find_cluster_change_point(values)
        assert result is None

    def test_empty_list(self):
        """
        Given an empty list, the function returns None.
        """
        values = []
        result = find_cluster_change_point(values)
        assert result is None

    def test_single_element(self):
        """
        Given a list with a single element, the function returns None.
        """
        values = [1]
        result = find_cluster_change_point(values)
        assert result is None

    def test_zeros_list(self):
        """
        Given a list with only zeros, the function returns None.
        """
        values = [0, 0, 0, 0, 0]
        result = find_cluster_change_point(values)
        assert result is None

    def test_multiple_clusters(self):
        """
        Given a list of values with multiple clusters, the function returns the
        index of the last element of the first cluster.
        """
        values = [1, 2, 3, 4, 5, 10, 11, 12, 13]
        result = find_cluster_change_point(values)
        assert result == 5

    def test_values_with_noise(self):
        """
        Given a list of values with noise, the function returns the index of the
        last element of the largest cluster.
        """
        values = [1, 2, 3, 4, 5, 10, 11, 12, 13, 20, 21, 22]
        result = find_cluster_change_point(values)
        assert result == 9

    def test_single_cluster_from_three(self):
        """
        Given a list of values with noise and a single cluster, the function
        returns the index of the last element.
        """
        values = [1, 2, 3, 4, 5, 10, 11, 12, 13, 20]
        result = find_cluster_change_point(values)
        assert result == 9


class TestSelectFeatures:
    """
    This class tests the functionality of the select_features function.
    """

    def test_returns_selected_features_sorted(self):
        """
        Returns a list of selected features sorted by their impact values.
        """
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        feature_names = ['A', 'B', 'C', 'D', 'E']
        result = select_features(values, feature_names)
        assert set(result) == set(['A', 'B', 'C', 'D', 'E'])

    def test_returns_empty_list_below_minimum_impact(self):
        """
        Returns an empty list if all mean SHAP values are below the minimum
        impact value.
        """
        values = np.array([0.000001, 0.000001, 0.000001, 0.000001, 0.000001])
        feature_names = ['A', 'B', 'C', 'D', 'E']
        result = select_features(values, feature_names)
        assert set(result) == set(['A', 'B', 'C', 'D', 'E'])

    def test_returns_empty_list_empty_values(self):
        """
        Returns an empty list when the input values are empty.
        """
        values = np.array([])
        feature_names = []
        result = select_features(values, feature_names)
        assert result == []

    def test_returns_empty_list_all_mean_shap_below_minimum_impact(self):
        """
        Returns an empty list when all mean SHAP values are below the minimum
        impact value.
        """
        values = np.array([0.000001, 0.000001, 0.000001])
        feature_names = ['A', 'B', 'C']
        result = select_features(values, feature_names)
        assert set(result) == set(feature_names)
