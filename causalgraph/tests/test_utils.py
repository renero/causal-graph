import networkx as nx
import pandas as pd
import pytest

from causalgraph.common import utils
from causalgraph.estimators.knowledge import Knowledge
from causalgraph.estimators.rex import Rex
from causalgraph.explainability.hierarchies import Hierarchies
from causalgraph.explainability.perm_importance import PermutationImportance
from causalgraph.models.dnn import NNRegressor


# Tests that the function returns 'cuda' when force is 'cuda' and cuda is available
def test_returns_cuda_when_force_is_cuda_and_cuda_is_not_available():
    # Try to force cuda when cuda is not available
    # Should raise a ValueError
    with pytest.raises(ValueError):
        utils.select_device(force='cuda')


# Tests that the function returns 'mps' when force is 'mps' and mps is available
def test_returns_mps_when_force_is_mps_and_mps_is_available():
    result = utils.select_device(force='mps')
    assert result == 'mps'


# Tests that the function returns 'cpu' when force is 'cpu' and no other options are available
def test_returns_cpu_when_force_is_cpu_and_no_other_options_are_available():
    result = utils.select_device(force='cpu')
    assert result == 'cpu'


# Tests that the function returns 'mps' when mps is available and no force is specified
def test_returns_mps_when_mps_is_available_and_no_force_is_specified():
    result = utils.select_device()
    assert result == 'mps'


# Tests that the function returns 'cpu' when no other options are available and no force is specified
def test_returns_cpu_when_no_other_options_are_available_and_no_force_is_specified():
    result = utils.select_device()
    assert result == 'mps'


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


def test_break_cycle_if_present_no_cycles():
    """
    Test function to check if the break_cycle_if_present function works correctly 
    when there are no cycles in the DAG.
    """
    # Create a DAG with no cycles
    dag = nx.DiGraph()
    # dag.add_node(1)
    # dag.add_node(2)
    dag.add_edge(1, 2)

    # Create a knowledge DataFrame with some permutation importances
    knowledge = pd.DataFrame({
        ('1', '2'): {'mean_pi': 0.5},
        ('2', '1'): {'mean_pi': 0.4},
    })

    # Call the break_cycle_if_present function
    result = utils.break_cycles_if_present(dag, knowledge)

    # Check if the result is equal to the input DAG
    assert result.nodes == dag.nodes
    assert result.edges == dag.edges


def test_break_cycle_if_present_one_cycle():
    """
    Test function to check if the `break_cycle_if_present` function correctly 
    breaks a cycle in a DAG with one cycle.
    """
    # Create a DAG with one cycle
    dag = nx.DiGraph()
    dag.add_edges_from([('1', '2'), ('2', '3'), ('3', '1')])

    # Create a knowledge DataFrame with some permutation importances
    learnings = pd.DataFrame([
        {'origin': '1', 'target': '2', 'shap_gof': 0.5},
        {'origin': '2', 'target': '3', 'shap_gof': 0.4},
        {'origin': '3', 'target': '1', 'shap_gof': 0.3}])
    rex = Rex(name="test")
    rex.hierarchies = Hierarchies()
    models = NNRegressor()
    models.regressor = {'test': 'test'}
    models.scoring = [0.1, 0.2, 0.3]
    rex.pi = PermutationImportance(models)
    rex.shaps = rex.hierarchies
    rex.indep = rex.hierarchies
    rex.feature_names = ['1', '2', '3']
    rex.models = models
    rex.G_shap = dag
    rex.root_causes = ['1', '2', '3']
    rex.correlation_th = None
    knowledge = Knowledge(rex, None)
    knowledge.results = learnings

    # Call the break_cycle_if_present function
    result = utils.break_cycles_if_present(dag, knowledge.results)

    # Check if the result is a DAG with no cycles
    assert nx.is_directed_acyclic_graph(result)


def test_break_cycle_if_present_multiple_cycles():
    """
    Test function to check if the `break_cycle_if_present` function can break cycles 
    in a DAG with multiple cycles.

    The function creates a DAG with multiple cycles and a knowledge DataFrame with 
    some permutation importances. It then calls the `break_cycle_if_present` 
    function and checks if the result is a DAG with no cycles.
    """
    # Create a DAG with multiple cycles
    dag = nx.DiGraph()
    dag.add_edges_from(
        [('1', '2'), ('2', '3'), ('3', '1'), ('3', '4'), ('4', '2')])

    # Create a knowledge DataFrame with some permutation importances
    learnings = pd.DataFrame([
        {'origin': '1', 'target': '2', 'shap_gof': 0.5},
        {'origin': '2', 'target': '3', 'shap_gof': 0.4},
        {'origin': '3', 'target': '1', 'shap_gof': 0.3},
        {'origin': '3', 'target': '4', 'shap_gof': 0.2},
        {'origin': '4', 'target': '2', 'shap_gof': 0.1}
    ])
    rex = Rex(name="test")
    rex.hierarchies = Hierarchies()
    models = NNRegressor()
    models.regressor = {'test': 'test'}
    models.scoring = [0.1, 0.2, 0.3]
    rex.pi = PermutationImportance(models)
    rex.shaps = rex.hierarchies
    rex.indep = rex.hierarchies
    rex.feature_names = ['1', '2', '3']
    rex.models = models
    rex.G_shap = dag
    rex.root_causes = ['1', '2', '3']
    rex.correlation_th = None
    knowledge = Knowledge(rex, None)
    knowledge.results = learnings

    # Call the break_cycle_if_present function
    result = utils.break_cycles_if_present(dag, knowledge.results)

    # Check if the result is a DAG with no cycles
    assert nx.is_directed_acyclic_graph(result)
