import networkx as nx

from causalgraph.independence.cond_indep import get_backdoor_paths

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
