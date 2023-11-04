#
# Module to compute conditional independencencies in a DAG, and
# sufficient tests.
#
# Author: Jesús Renero
#
import itertools
from typing import List
import networkx as nx


class ConditionalIndependencies:
    """
    A class to store conditional independencies in a graph.

    Attributes
    ----------
    _cache : dict
        A dictionary representing the conditional independencies.

    Methods
    -------
    add(x, y, z)
        Adds a new conditional independence to the cache.
    __str__()
        Returns a string representation of the conditional independencies.
    __repr__()
        Returns a string representation of the conditional independencies.
    """

    def __init__(self):
        self._cache = {}

    def __str__(self):
        s = "Conditional Independencies:\n"
        for (x, y), z in self._cache.items():
            if z == set():
                s += f"  {x} ⊥ {y}\n"
            else:
                s += f"  {x} ⊥ {y} | {z}\n"
        return s

    def __repr__(self):
        s = "{"
        last = False
        if self._cache:
            for (x, y), z in self._cache.items():
                if z == set():
                    s += f"({x}, {y}):{[]}"
                else:
                    s += f"({x}, {y}):{list(z)}"
                # check if this is the last element in the dict
                if (x, y) == list(self._cache.keys())[-1]:
                    last = True
                if not last:
                    s += ", "
        s += "}"
        return s

    def add(self, x: str, y: str, z: set):
        """
        Adds a new conditional independence to the cache.

        Parameters
        ----------
        x : str
            A node in the graph.
        y : str
            A node in the graph.
        z : set
            A set of nodes in the graph.
        """
        if (x, y) in self._cache:
            self._cache[(x, y)] = self._cache[(x, y)].union(z)
        else:
            self._cache[(x, y)] = z


class SufficientSets:
    """
    A class to represent the sufficient sets of a conditional independence test.

    Attributes
    ----------
    _cache : list
        A list of tuples representing the sufficient sets.

    Methods
    -------
    add(suff_set)
        Adds a new sufficient set to the cache.

        Parameters
        ----------
        suff_set : list
            A list of tuples representing the new sufficient set to be added.

    __str__()
        Returns a string representation of the sufficient sets.
    """

    def __init__(self):
        self._cache = []

    def add(self, suff_set):
        """
        Adds a new sufficient set to the cache.

        Parameters
        ----------
        suff_set : list
            A list of tuples representing the new sufficient set to be added.
        """
        for x, y in suff_set:
            if ((x, y) not in self._cache) and ((y, x) not in self._cache):
                self._cache.append((x, y))

    def __str__(self):
        """
        Returns a string representation of the sufficient sets.

        Returns
        -------
        str
            A string representation of the sufficient sets.
        """
        s = "Sufficient sets:\n"
        if self._cache:
            for sufficient_set in self._cache:
                s += f"  {sufficient_set}\n"
        else:
            s = "No sufficient sets found"

        return s

    def __repr__(self):
        s = "["
        last = False
        if self._cache:
            for sufficient_set in self._cache:
                s += f"{sufficient_set}"
                if sufficient_set == self._cache[-1]:
                    last = True
                if not last:
                    s += ", "
        s += "]"
        return s


def get_backdoor_paths(dag: nx.DiGraph, x: str, y: str):
    """
    Returns all backdoor paths between two nodes in a graph. A backdoor path
    is a path that starts with an edge towards 'x' and ends with an edge
    towards 'y'.

    Parameters:
    -----------
    dag: nx.DiGraph
        A directed graph
    x: str
        A node in the graph
    y: str
        A node in the graph

    Returns:
    --------
    paths: list
        A list of paths between x and y
    """
    # Check if x or y are not in the graph
    if x not in dag.nodes() or y not in dag.nodes():
        return []
    undirected_graph = dag.to_undirected()
    # list all paths between 'x' and 'y'
    paths = (p for p in nx.all_simple_paths(
        undirected_graph, source=x, target=y) if dag.has_edge(p[1], x))
    return list(paths)


def get_paths(graph: nx.DiGraph, x: str, y: str):
    """
    Returns all simple paths between two nodes in a directed graph.

    Parameters
    ----------
        - graph (nx.DiGraph): A directed graph.
        - x (str): The starting node.
        - y (str): The ending node.

    Returns
    -------
        - list: A list of all simple paths between x and y.
    """
    # Check if x or y are not in the graph
    if x not in graph.nodes() or y not in graph.nodes():
        return []
    undirected_graph = graph.to_undirected()
    # list all paths between 'x' and 'y'
    paths = list(nx.all_simple_paths(undirected_graph, source=x, target=y))
    return paths


def find_colliders_in_path(dag: nx.DiGraph, path: List[str]):
    """
    Returns all colliders in a path.

    Parameters:
    -----------
    G: nx.DiGraph
        A directed graph
    path: list
        A path formed by nodes in the graph

    Returns:
    --------
    colliders: set
        A set of colliders in the path
    """
    colliders = []
    for i in range(1, len(path)-1):
        if dag.has_edge(path[i-1], path[i]) and dag.has_edge(path[i+1], path[i]):
            colliders.append(path[i])

    return set(colliders)


def get_sufficient_sets_for_pair(dag, x, y, verbose=False):
    """
    Compute the sufficient sets for a pair of nodes in a graph. A sufficient set
    is a set of nodes that blocks all backdoor paths between x and y.

    Parameters:
    -----------
    G: nx.DiGraph
        A directed graph
    x: str
        A node in the graph
    y: str
        A node in the graph
    verbose: bool
        If True, print additional information

    Returns:
    --------
    sufficient_sets: list
        A list of sufficient sets for the pair of nodes (x, y)
    """
    backdoor_paths = get_backdoor_paths(dag, x, y)
    if verbose:
        print("Backdoor paths:")
        for path in backdoor_paths:
            print(path)
    sufficient_sets = []
    for path in backdoor_paths:
        # get all nodes in the path except the first and last
        sufficient_set = path[1:-1]
        # check that no node in sufficient_set is descendant of x
        descendants = nx.descendants(dag, x)
        if any([d in descendants for d in sufficient_set]):
            if verbose:
                print(
                    f"Path {path} discarded because it contains a descendant of x")
            continue

        sufficient_sets.append(sufficient_set)

    # Now I must check that the nodes in the sufficient set block every backdoor path
    # between x and y
    final_suff_set = []
    for sufficient_set in sufficient_sets:
        if verbose:
            print(
                f"Checking that {sufficient_set} blocks all backdoor paths "
                f"between x and y")
        # Check if any of the nodes in the sufficient set is in a collider in the path
        colliders = find_colliders_in_path(dag, [x] + sufficient_set + [y])
        # If any of the nodes in the sufficient set is a collider, then continue
        if len(colliders) > 0:
            if verbose:
                print(
                    f"  🚫 {sufficient_set} contains a collider: {colliders}")
            continue
        all_conditions = True
        for path in backdoor_paths:
            if verbose:
                print(f"  Checking path {path}")
            # Check that this path can be blocked by any node in the sufficient set
            # The path is blocked if any of the nodes in the sufficient set
            # is in the path
            colliders = find_colliders_in_path(dag, path)
            if verbose:
                print(f"    ! Colliders in path: {colliders}")
            # Find what nodes from the sufficient set are in the path
            nodes_in_path = set(sufficient_set).intersection(set(path))
            if verbose:
                print(
                    f"    + Nodes from sufficient set in path: {nodes_in_path}")
            if len(nodes_in_path) > 0:
                # Check that at least one of the nodes in the path is NOT a collider
                if nodes_in_path.intersection(colliders) == set():
                    if verbose:
                        print(
                            f"      Path {path} blocked by nodes in {sufficient_set} ✅")
                elif len(nodes_in_path) == len(colliders):
                    if verbose:
                        print(f"      ALL nodes in {sufficient_set} are colliders "
                              f"in {path} ❌\n"
                              f"      => {nodes_in_path.intersection(colliders)} == "
                              f"{colliders}")
                    all_conditions = False
                else:
                    if verbose:
                        print(f"      Some nodes in {sufficient_set} are NOT colliders "
                              f"in {path} ✅")
            else:
                if verbose:
                    print(f"      No nodes in {sufficient_set} are in {path}, "
                          f"so they do not block this path ❌")
                all_conditions = False
        if all_conditions:
            if verbose:
                print(
                    f"  👍 {sufficient_set} blocks all backdoor paths "
                    f"between x and y")
            final_suff_set.append(sufficient_set)

    return final_suff_set


def get_sufficient_sets(dag, verbose=False):
    """
    Get the sufficient sets (admissible sets) for all pairs of nodes in a graph.

    Parameters:
    -----------
    G: nx.DiGraph
        A directed graph
    verbose: bool
        If True, print additional information

    Returns:
    --------
    suff_sets: Suff_Sets
        A list of sufficient sets for all pairs of nodes in the graph
    """
    suff_sets = SufficientSets()
    for x, y in itertools.combinations(dag.nodes(), 2):
        sufficient_set = get_sufficient_sets_for_pair(dag, x, y, verbose)
        if sufficient_set:
            suff_sets.add(sufficient_set)

    return suff_sets


def get_conditional_independencies(dag, verbose=False):
    """
    Computes the set of conditional independencies implied by the graph G.

    Parameters:
    -----------
    dag : networkx.DiGraph
        The directed acyclic graph representing the causal relationships
        between the variables.
    verbose : bool, optional
        If True, prints additional information about the computation.

    Returns:
    --------
    cond_indeps : Cond_Indep
        The object containing the set of conditional independencies implied
        by the graph G.
    """
    cond_indeps = ConditionalIndependencies()
    # Enumerate all pairs of nodes in G that are not d_separated
    for x, y in itertools.combinations(dag.nodes(), 2):
        # Check if x and y are connected by an edge
        if dag.has_edge(x, y) or dag.has_edge(y, x):
            continue
        if not nx.d_separated(dag, {x}, {y}, set()):
            if verbose:
                print(f"Pair ({x}, {y})")
            paths = get_paths(dag, x, y)
            # Check if any of the paths contains a collider
            for path in paths:
                if verbose:
                    print("  Path:", path)
                colliders = find_colliders_in_path(dag, path)
                if len(colliders) == 0:
                    cond_indeps.add(x, y, set(path[1:-1]))
                    if verbose:
                        print(f"   (no colliders on path {path})\n"
                              f"   (len(colliders) == {len(colliders)})\n"
                              f"    ✅ {x} ⊥ {y} | {set(path[1:-1])}")
                else:
                    if verbose:
                        print(f"    Colliders in path: {colliders}")
                    for blocker in path[1:-1]:
                        if verbose:
                            print(f"    Blocking on {blocker}")
                        if blocker not in colliders:
                            cond_indeps.add(x, y, {blocker})
                            if verbose:
                                print(f"      ✅ {x} ⊥ {y} | {blocker}")
                        else:
                            if verbose:
                                print(
                                    f"    🚫 Blocking on {blocker} is Collider: "
                                    f"{colliders}")
        else:
            cond_indeps.add(x, y, set())
            if verbose:
                print(f"Pair ({x}, {y})\n"
                      f"  ✅ {x} ⊥ {y} | ∅")

    return cond_indeps


if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_edges_from([('z1', 'x'), ('z1', 'z3'), ('z3', 'x'),
                     ('z3', 'y'), ('x', 'y'), ('z2', 'z3'), ('z2', 'y')])

    ss = get_sufficient_sets(G, verbose=False)
    print(ss)

    cond_independencies = get_conditional_independencies(G, verbose=False)
    print(cond_independencies)
