import networkx as nx
from causalexplain.independence.cond_indep import (
    ConditionalIndependencies, get_backdoor_paths, get_paths,
    find_colliders_in_path, get_sufficient_sets_for_pair
)

# Returns an empty list if x and y are not connected


def test_returns_empty_list_if_not_connected():
    """
    Test that the function get_backdoor_paths returns an empty list when there 
    is no path between x and y in the DAG.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)])

    # Test case: x and y are not connected
    x = 1
    y = 6
    paths = get_backdoor_paths(dag, x, y)
    assert paths == []

# Returns an empty list if there are no backdoor paths between x and y


def test_returns_empty_list_if_no_backdoor_paths():
    """
    Test that the function get_backdoor_paths returns an empty list when 
    there are no backdoor paths between two nodes.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)])

    # Test case: x and y are connected, but there are no backdoor paths
    x = 1
    y = 2
    paths = get_backdoor_paths(dag, x, y)
    assert paths == []

# Returns a list with one path if there is one backdoor path between x and y


def test_returns_list_with_one_path_if_one_backdoor_path():
    """
    Test that the function get_backdoor_paths returns a list with one path if 
    there is only one backdoor path between two nodes.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(2, 1), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)])

    # Test case: x and y are connected, and there is one backdoor path
    x = 1
    y = 4
    paths = get_backdoor_paths(dag, x, y)
    assert set(map(tuple, paths)) == {tuple([1, 2, 4]), tuple([1, 2, 3, 4])}

# Returns an empty list if x and y are the same node


def test_returns_condsets_when_collider():
    """
    Test function to check if the function get_backdoor_paths returns the 
    correct set of paths
    when there is a collider in the graph.
    """
    dag = nx.DiGraph()
    dag.add_edges_from([('r', 'x'), ('r', 's'), ('t', 's'), ('t', 'y')])

    x = 'x'
    y = 'y'
    paths = get_backdoor_paths(dag, x, y)
    print(paths)
    assert set(map(tuple, paths)) == {tuple(['x', 'r', 's', 't', 'y'])}


def test_returns_empty_list_if_same_node():
    """
    Test that the function get_backdoor_paths returns an empty list when the 
    input nodes are the same.

    The function should return an empty list because there are no backdoor paths 
    between a node and itself.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)])

    # Test case: x and y are the same node
    x = 1
    y = 1
    paths = get_backdoor_paths(dag, x, y)
    assert paths == []

# Returns an empty list if x and y are not in the graph


def test_returns_empty_list_if_not_in_graph():
    """
    Test that the function get_backdoor_paths returns an empty list when 
    x and y are not in the graph.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)])

    # Test case: x and y are not in the graph
    x = 7
    y = 8
    paths = get_backdoor_paths(dag, x, y)
    assert paths == []

# Returns an empty list if there is no path between x and y


def test_returns_empty_list_if_no_path():
    """
    Test that the function get_backdoor_paths returns an empty list when 
    there is no path between x and y in the DAG.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)])

    # Test case: there is no path between x and y
    x = 1
    y = 6
    paths = get_backdoor_paths(dag, x, y)
    assert paths == []


def test_get_paths_returns_empty_list_if_not_connected():
    """
    Test that the function get_paths returns an empty list when there 
    is no path between x and y in the DAG.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(2, 3), (2, 4), (3, 4), (4, 5), (5, 6)])

    # Test case: x and y are not connected
    x = 1
    y = 6
    paths = get_paths(dag, x, y)
    assert paths == []


def test_get_paths_returns_list_with_one_path_if_one_path():
    """
    Test that the function get_paths returns a list with one path if 
    there is only one path between two nodes.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])

    # Test case: x and y are connected, and there is one path
    x = 1
    y = 6
    paths = get_paths(dag, x, y)
    assert paths == [[1, 2, 3, 4, 5, 6]]


def test_get_paths_returns_list_with_multiple_paths():
    """
    Test that the function get_paths returns a list with multiple paths if 
    there are multiple paths between two nodes.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)])

    # Test case: x and y are connected, and there are multiple paths
    x = 1
    y = 6
    paths = get_paths(dag, x, y)
    assert set(map(tuple, paths)) == {
        tuple([1, 2, 4, 5, 6]), tuple([1, 2, 3, 4, 5, 6])}


def test_get_paths_returns_empty_list_if_same_node():
    """
    Test that the function get_paths returns an empty list when the 
    input nodes are the same.

    The function should return an empty list because there are no paths 
    between a node and itself.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)])

    # Test case: x and y are the same node
    x = 1
    y = 1
    paths = get_paths(dag, x, y)
    assert paths == []


def test_get_paths_returns_empty_list_if_not_in_graph():
    """
    Test that the function get_paths returns an empty list when 
    x and y are not in the graph.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)])

    # Test case: x and y are not in the graph
    x = 7
    y = 8
    paths = get_paths(dag, x, y)
    assert paths == []


def test_get_paths_returns_empty_list_if_no_path():
    """
    Test that the function get_paths returns an empty list when 
    there is no path between x and y in the DAG.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (7, 6)])

    # Test case: there is no path between x and y
    x = 1
    y = 6
    paths = get_paths(dag, x, y)
    assert paths == []


def test_find_colliders_in_path_returns_empty_set_when_no_colliders():
    """
    Test that the function find_colliders_in_path returns an empty set when 
    there are no colliders in the path.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])

    # Test case: there are no colliders in the path
    path = [1, 2, 3, 4, 5, 6]
    colliders = find_colliders_in_path(dag, path)
    assert colliders == set()


def test_find_colliders_in_path_returns_set_of_colliders():
    """
    Test that the function find_colliders_in_path returns a set of colliders 
    in the path.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 4), (5, 6)])

    # Test case: there are colliders in the path
    path = [1, 2, 3, 4, 5, 6]
    colliders = find_colliders_in_path(dag, path)
    assert colliders == {4}


def test_find_colliders_in_path_returns_empty_set_when_path_is_too_short():
    """
    Test that the function find_colliders_in_path returns an empty set when 
    the path is too short to contain a collider.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])

    # Test case: the path is too short to contain a collider
    path = [1, 2]
    colliders = find_colliders_in_path(dag, path)
    assert colliders == set()


def test_find_colliders_in_path_returns_empty_set_when_path_is_not_in_dag():
    """
    Test that the function find_colliders_in_path returns an empty set when 
    the path is not in the DAG.
    """
    # Create a simple DAG
    dag = nx.DiGraph()
    dag.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])

    # Test case: the path is not in the DAG
    path = [1, 2, 3, 7]
    colliders = find_colliders_in_path(dag, path)
    assert colliders == set()


def test_add_new_independence():
    ci = ConditionalIndependencies()
    ci.add('X', 'Y', ['Z'])
    assert ci._cache[('X', 'Y', ('Z',))] is True


def test_add_existing_independence():
    ci = ConditionalIndependencies()
    ci.add('X', 'Y', ['Z'])
    ci.add('X', 'Y', ['Z'])  # Adding the same independence again
    assert len(ci._cache) == 1  # Cache size should remain 1


def test_add_empty_conditioning_set():
    ci = ConditionalIndependencies()
    ci.add('A', 'B', [])
    assert ci._cache[('A', 'B', ())] is True


def test_no_backdoor_paths():
    dag = nx.DiGraph()
    dag.add_edges_from([('A', 'B'), ('C', 'D')])
    sufficient_sets = get_sufficient_sets_for_pair(dag, 'A', 'D')
    assert sufficient_sets == []


def test_single_backdoor_path():
    dag = nx.DiGraph()
    dag.add_edges_from([('Z', 'X'), ('Z', 'Y')])
    sufficient_sets = get_sufficient_sets_for_pair(dag, 'X', 'Y')
    assert sufficient_sets == [['Z']]


def test_path_with_descendants():
    dag = nx.DiGraph()
    dag.add_edges_from([('X', 'Z'), ('Z', 'Y'), ('X', 'Z2')])
    sufficient_sets = get_sufficient_sets_for_pair(dag, 'X', 'Y')
    assert sufficient_sets == []


def test_path_with_colliders():
    dag = nx.DiGraph()
    dag.add_edges_from([('X', 'Z'), ('Z', 'Y'), ('Z', 'X')])
    sufficient_sets = get_sufficient_sets_for_pair(dag, 'X', 'Y')
    assert sufficient_sets == []
