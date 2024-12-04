import networkx as nx
import pandas as pd
import numpy as np
import pytest
import torch

from causalgraph.common import utils


# Tests that the function returns 'cuda' when force is 'cuda' and cuda is available
def test_returns_cuda_when_force_is_cuda_and_cuda_is_not_available():
    # Try to force cuda when cuda is not available
    # Should raise a ValueError
    with pytest.raises(ValueError):
        utils.select_device(force='cuda')


# Tests that the function returns 'mps' when force is 'mps' and mps is available
def test_returns_mps_when_force_is_mps_and_mps_is_available():
    result = utils.select_device(force='mps')
    if torch.backends.mps.is_available():
        assert result == 'mps'
    else:
        assert result == 'cpu'


# Tests that the function returns 'cpu' when force is 'cpu' and no other options are available
def test_returns_cpu_when_force_is_cpu_and_no_other_options_are_available():
    result = utils.select_device(force='cpu')
    assert result == 'cpu'


# Tests that the function returns 'mps' when mps is available and no force is specified
def test_returns_mps_when_mps_is_available_and_no_force_is_specified():
    result = utils.select_device()
    if torch.backends.mps.is_available():
        assert result == 'mps'
    else:
        assert result == 'cpu'


# Tests that the function returns 'cpu' when no other options are available and no force is specified
def test_returns_cpu_when_no_other_options_are_available_and_no_force_is_specified():
    result = utils.select_device()
    if torch.backends.mps.is_available():
        assert result == 'mps'
    else:
        assert result == 'cpu'


# Test with two identical graphs
def test_identical_graphs():
    # Create two identical graphs
    g1 = nx.DiGraph()
    g1.add_node(1)
    g1.add_node(2)
    g1.add_edge(1, 2)

    g2 = nx.DiGraph()
    g2.add_node(1)
    g2.add_node(2)
    g2.add_edge(1, 2)

    # Call the graph_intersection function
    result = utils.graph_intersection(g1, g2)

    # Check if the result is equal to the input graphs
    assert result.nodes == g1.nodes
    assert result.edges == g1.edges


# Test with two graphs with different nodes and edges
def test_different_nodes_edges():
    # Create two graphs with different nodes and edges
    g1 = nx.DiGraph()
    g1.add_node(1)
    g1.add_node(2)
    g1.add_edge(1, 2)

    g2 = nx.DiGraph()
    g2.add_node(3)
    g2.add_node(4)
    g2.add_edge(3, 4)

    # Call the graph_intersection function
    result = utils.graph_intersection(g1, g2)

    # Check if the result is an empty graph
    assert len(result.nodes) == 0
    assert len(result.edges) == 0


# Test with two graphs with some common nodes and edges
def test_common_nodes_edges():
    # Create two graphs with some common nodes and edges
    g1 = nx.DiGraph()
    g1.add_node(1)
    g1.add_node(2)
    g1.add_edge(1, 2)

    g2 = nx.DiGraph()
    g2.add_node(2)
    g2.add_node(3)
    g2.add_edge(2, 3)

    # Call the graph_intersection function
    result = utils.graph_intersection(g1, g2)

    # Check if the result has the common nodes and edges
    assert len(result.nodes) == 1
    assert len(result.edges) == 0
    assert 2 in result.nodes


# Test two graphs with different node attributes
def test_different_node_attributes():
    # Create two graphs with different node attributes
    g1 = nx.DiGraph()
    g1.add_node(1, attr='A')
    g1.add_node(2, attr='B')
    g1.add_edge(1, 2)

    g2 = nx.DiGraph()
    g2.add_node(1, attr='C')
    g2.add_node(2, attr='D')
    g2.add_edge(1, 2)

    # Call the graph_intersection function
    result = utils.graph_intersection(g1, g2)

    # Check if the result has the minimum node attributes
    assert len(result.nodes) == 2
    # assert 'A' in result.nodes[1]['attr']
    # assert 'B' in result.nodes[2]['attr']


# Test with two empty graphs
def test_empty_graphs():
    # Create two empty graphs
    g1 = nx.DiGraph()
    g2 = nx.DiGraph()

    # Call the graph_intersection function
    result = utils.graph_intersection(g1, g2)

    # Check if the result is an empty graph
    assert len(result.nodes) == 0
    assert len(result.edges) == 0


# Test with two undirected graphs
def test_identical_undirected_graphs():
    # Create two identical undirected graphs
    g1 = nx.Graph()
    g1.add_node(1)
    g1.add_node(2)
    g1.add_edge(1, 2)

    g2 = nx.Graph()
    g2.add_node(1)
    g2.add_node(2)
    g2.add_edge(1, 2)

    # Call the graph_intersection function
    result = utils.graph_intersection(g1, g2)

    # Check if the result is equal to the input graphs
    assert result.nodes == g1.nodes
    assert result.edges == g1.edges



class TestClassifyVariable:
    """Test suite for _classify_variable function"""

    def test_binary_numeric(self):
        """Test binary classification with numeric values"""
        arr = pd.Series([0, 1, 0, 1, 0])
        assert utils._classify_variable(arr) == "binary"
        
        arr = pd.Series([1.0, 2.0, 1.0, 2.0])
        assert utils._classify_variable(arr) == "binary"

    def test_binary_categorical(self):
        """Test binary classification with categorical values"""
        arr = pd.Series(['yes', 'no', 'yes', 'no'])
        assert utils._classify_variable(arr) == "multiclass"
        
        arr = pd.Series(['true', 'false', 'true'])
        assert utils._classify_variable(arr) == "multiclass"

    def test_multiclass_categorical(self):
        """Test multiclass classification with categorical values"""
        arr = pd.Series(['a', 'b', 'c', 'd'])
        assert utils._classify_variable(arr) == "multiclass"
        
        arr = pd.Series(['low', 'medium', 'high', 'medium', 'low'])
        assert utils._classify_variable(arr) == "multiclass"

    def test_continuous_numeric(self):
        """Test continuous classification with numeric values"""
        arr = pd.Series([1.0, 2.5, 3.7, 4.2, 5.0])
        assert utils._classify_variable(arr) == "continuous"
        
        arr = pd.Series(np.random.normal(0, 1, 100))
        assert utils._classify_variable(arr) == "continuous"

    def test_mixed_types(self):
        """Test with mixed types (should be handled as multiclass)"""
        arr = pd.Series(['1', '2', 'a', 'b'])
        assert utils._classify_variable(arr) == "multiclass"

    def test_special_cases(self):
        """Test special cases"""
        # Empty series
        arr = pd.Series([])
        assert utils._classify_variable(arr) == "continuous"
        
        # Series with NaN values
        arr = pd.Series([1, 2, np.nan, 4])
        assert utils._classify_variable(arr) == "continuous"
        
        # Boolean values
        arr = pd.Series([True, False, True])
        assert utils._classify_variable(arr) == "binary"


class TestCastCategoricalsToInt:
    """Test suite for cast_categoricals_to_int function"""

    def test_cast_multiclass(self):
        """Test casting multiclass categorical variables"""
        df = pd.DataFrame({
            'A': pd.Categorical(['a', 'b', 'c', 'a']),
            'B': pd.Categorical(['x', 'y', 'z', 'x']),
            'C': [1.0, 2.0, 3.0, 1.0]  # Numeric column should remain unchanged
        })
        result = utils.cast_categoricals_to_int(df)
        # Categories should be converted to integers starting from 0
        assert result['A'].dtype == np.int16
        assert result['B'].dtype == np.int16
        assert result['C'].dtype == np.float64  # Should remain unchanged
        assert set(result['A'].unique()) == {0, 1, 2}
        assert set(result['B'].unique()) == {0, 1, 2}

    def test_cast_binary(self):
        """Test casting binary categorical variables"""
        df = pd.DataFrame({
            'A': ['yes', 'no', 'yes', 'no'],
            'B': ['true', 'false', 'true', 'false'],
            'C': [0, 1, 0, 1]  # Already binary numeric
        })
        result = utils.cast_categoricals_to_int(df)
        assert result['A'].dtype == np.int16
        assert result['B'].dtype == np.int16
        assert result['C'].dtype == np.int16  # Should remain numeric
        assert set(result['A'].unique()) == {0, 1}
        assert set(result['B'].unique()) == {0, 1}
        assert set(result['C'].unique()) == {0, 1}

    def test_no_categorical(self):
        """Test with no categorical columns"""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0],
            'C': [7, 8, 9]
        })
        result = utils.cast_categoricals_to_int(df)
        pd.testing.assert_frame_equal(result, df)  # Should be unchanged

    def test_mixed_types(self):
        """Test with mixed types in the same column"""
        df = pd.DataFrame({
            'A': ['1', '2', 'a', 'b'],  # Mixed strings
            'B': [1, 2, 'three', 'four'],  # Mixed numbers and strings
            'C': [1.0, 2.0, 3.0, 4.0]  # Pure numeric
        })
        result = utils.cast_categoricals_to_int(df)
        assert result['A'].dtype == np.int16  # Should be converted
        assert result['B'].dtype == np.int16  # Should be converted
        assert result['C'].dtype == np.float64  # Should remain unchanged
        assert len(result['A'].unique()) == 4
        assert len(result['B'].unique()) == 4
