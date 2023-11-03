#
# Module to compute conditional independencencies in a DAG, and
# sufficient tests.
#
# Author: Jesús Renero
#
import itertools
import networkx as nx


class Cond_Indep:
    def __init__(self):
        self._cache = {}

    def __str__(self):
        s = "Conditional Independences:\n"
        for (x, y), z in self._cache.items():
            if z == set():
                s += f"  {x} ⊥ {y}\n"
            else:
                s += f"  {x} ⊥ {y} | {z}\n"
        return s

    def __repr__(self):
        s = "{"
        last = False
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
        if (x, y) in self._cache:
            self._cache[(x, y)] = self._cache[(x, y)].union(z)
        else:
            self._cache[(x, y)] = z


class Suff_Sets:
    def __init__(self):
        self._cache = []

    def add(self, suff_set):
        for x, y in suff_set:
            if (not (x, y) in self._cache) and (not (y, x) in self._cache):
                self._cache.append((x, y))

    def __str__(self):
        s = ""
        if self._cache:
            s += "Sufficient sets:\n"
            for ss in self._cache:
                s += f"  {ss}\n"
        else:
            s = "No sufficient sets found"

        return s


def get_backdoor_paths(G, x, y):
    """
    Returns all backdoor paths between two nodes in a graph.
    """
    uG = G.to_undirected()
    # list all paths between 'x' and 'y'
    paths = list(nx.all_simple_paths(uG, source=x, target=y))
    # filter those starting by an edge towards 'x' in the directed graph
    paths = [p for p in paths if G.has_edge(p[1], x)]
    return paths


def get_paths(G, x, y):
    """
    Returns all paths between two nodes in a graph.
    """
    uG = G.to_undirected()
    # list all paths between 'x' and 'y'
    paths = list(nx.all_simple_paths(uG, source=x, target=y))
    return paths


def find_colliders_in_path(G, path):
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
        if G.has_edge(path[i-1], path[i]) and G.has_edge(path[i+1], path[i]):
            colliders.append(path[i])

    return set(colliders)


def get_sufficient_sets_for_pair(G, x, y, verbose=False):
    backdoor_paths = get_backdoor_paths(G, x, y)
    if verbose:
        print("Backdoor paths:")
        for path in backdoor_paths:
            print(path)
    sufficient_sets = []
    for path in backdoor_paths:
        # get all nodes in the path except the first and last
        sufficient_set = path[1:-1]
        # check that no node in sufficient_set is descendant of x
        descendants = nx.descendants(G, x)
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
        colliders = find_colliders_in_path(G, [x] + sufficient_set + [y])
        # If any of the nodes in the sufficient set is a collider, then continue
        if len(colliders) > 0:
            if verbose:
                print(
                    f"  🚫 {sufficient_set} contains a collider: {colliders}")
            continue
        all_conditions = True
        for path in backdoor_paths:
            print(f"  Checking path {path}") if verbose else None
            # Check that this path can be blocked by any node in the sufficient set
            # The path is blocked if any of the nodes in the sufficient set
            # is in the path
            colliders = find_colliders_in_path(G, path)
            print(f"    ! Colliders in path: {colliders}") if verbose else None
            # Find what nodes from the sufficient set are in the path
            nodes_in_path = set(sufficient_set).intersection(set(path))
            print(f"    + Nodes from sufficient set in path:"
                  f" {nodes_in_path}") if verbose else None
            if len(nodes_in_path) > 0:
                # Check that at least one of the nodes in the path is NOT a collider
                if nodes_in_path.intersection(colliders) == set():
                    print(
                        f"      Path {path} blocked by nodes in "
                        f"{sufficient_set} ✅") if verbose else None
                elif len(nodes_in_path) == len(colliders):
                    print(f"      ALL nodes in {sufficient_set} are colliders "
                          f"in {path} ❌\n"
                          f"      => {nodes_in_path.intersection(colliders)} == "
                          f"{colliders}") if verbose else None
                    all_conditions = False
                else:
                    print(f"      Some nodes in {sufficient_set} are NOT colliders "
                          f"in {path} ✅") if verbose else None
            else:
                print(f"      No nodes in {sufficient_set} are in {path}, "
                      f"so they do not block this path ❌") if verbose else None
                all_conditions = False
        if all_conditions:
            print(
                f"  👍 {sufficient_set} blocks all backdoor paths "
                f"between x and y") if verbose else None
            final_suff_set.append(sufficient_set)

    return final_suff_set


def get_sufficient_sets(G, verbose=False):
    suff_sets = Suff_Sets()
    for x, y in itertools.combinations(G.nodes(), 2):
        sufficient_set = get_sufficient_sets_for_pair(G, x, y, verbose)
        if sufficient_set:
            suff_sets.add(sufficient_set)

    return suff_sets


def get_conditional_independencies(G, verbose=False):
    cond_indeps = Cond_Indep()
    # Enumerate all pairs of nodes in G that are not d_separated
    for x, y in itertools.combinations(G.nodes(), 2):
        # Check if x and y are connected by an edge
        if G.has_edge(x, y) or G.has_edge(y, x):
            continue
        if not nx.d_separated(G, {x}, {y}, set()):
            print(f"Pair ({x}, {y})") if verbose else None
            paths = get_paths(G, x, y)
            # Check if any of the paths contains a collider
            for path in paths:
                print("  Path:", path) if verbose else None
                colliders = find_colliders_in_path(G, path)
                if len(colliders) == 0:
                    cond_indeps.add(x, y, set(path[1:-1]))
                    print(f"   (no colliders on path {path})\n"
                          f"   (len(colliders) == {len(colliders)})\n"
                          f"    ✅ {x} ⊥ {y} | {set(path[1:-1])}") if verbose else None
                else:
                    print(
                        f"    Colliders in path: {colliders}") if verbose else None
                    for blocker in path[1:-1]:
                        print(
                            f"    Blocking on {blocker}") if verbose else None
                        if blocker not in colliders:
                            cond_indeps.add(x, y, {blocker})
                            print(
                                f"      ✅ {x} ⊥ {y} | {blocker}") if verbose else None
                        else:
                            print(
                                f"    🚫 Blocking on {blocker} is Collider: {colliders}"
                            ) if verbose else None
        else:
            cond_indeps.add(x, y, set())
            print(f"Pair ({x}, {y})\n"
                  f"  ✅ {x} ⊥ {y} | ∅") if verbose else None

    return cond_indeps


if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_edges_from([('z1', 'x'), ('z1', 'z3'), ('z3', 'x'),
                     ('z3', 'y'), ('x', 'y'), ('z2', 'z3'), ('z2', 'y')])
    ss = get_sufficient_sets(G, verbose=False)
    print(ss)
