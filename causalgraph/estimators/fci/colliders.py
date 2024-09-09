"""
This file contains all the methods related to colliders and d-separation sets.

"""
# pylint: disable=E1101:no-member
# pylint: disable=W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=W0106:expression-not-assigned
# pylint: disable=C0103:invalid-name, C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, R0902:too-many-instance-attributes
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=R1702:too-many-branches

import itertools

import networkx as nx
import timeout_decorator
from mlforge.progbar import ProgBar

from causalgraph.estimators.fci import rules
from causalgraph.estimators.fci.debug import Debug
from causalgraph.estimators.fci.initialization import save_graph
from causalgraph.estimators.fci.pag import PAG


def init_pag(skeleton, sepSet, verbose, debug):
    """
    Initializes the PAG (Partially Ancestral Graph) from the base skeleton.

    Args:
        skeleton (networkx.Graph): The base skeleton graph.
        sepSet (dict): The dictionary containing the separation sets.
        verbose (bool): Flag indicating whether to print verbose output.
        debug (Debug): The debug object for debugging purposes.

    Returns:
        tuple: A tuple containing the initialized PAG and the possible d-separation sets.
    """
    if verbose:
        print("2nd Stage Skeleton")
    debug.dbg("Initializing PAG from base skeleton")
    pag = PAG()
    pag.add_nodes_from(skeleton.nodes())
    pag.add_edges_from(skeleton.edges())
    if verbose:
        print("Orienting PAG edges...")
    orient_V(pag, sepSet, debug)
    if verbose:
        print("Finding possible d-separation sets...")
    dseps = possible_d_seps(pag, debug)
    return pag, dseps


def orientEdges(skeleton, sepSet, log, verbose, debug, data_file, output_path,
                save_intermediate: bool = False, prog_bar=False, silent=False):
    """
    A function to orient the edges of a skeleton using the orientation
    rules of the FCI algorithm

    Parameters
    ----------
        skeleton: networkx.Graph(), skeleton estimation
        sepSet: dict Dicitonary containg separation sets of all pairs of nodes
        log: Logger The logger to be used (optional)
        debug: Debug Debugger
        save_intermediate: ditto

    Returns
    -------
        PAG containing estimated causal relationships in data

    """
    pag = PAG()
    pag.add_nodes_from(skeleton)
    pag.add_edges_from(skeleton.edges)
    orient_V(pag, sepSet, debug)
    if save_intermediate:
        save_graph(pag, "adjmat_FCI_orientV", data_file, output_path, log)
    changed_pag = True
    while changed_pag:
        changed_pag = False
        three_tuples = list(itertools.permutations(pag, 3))
        if verbose:
            print("Applying rules...")

        # pbar = tqdm(
        #     total=len(three_tuples),
        #     **tqdm_params("App.Rules", prog_bar, silent=silent))
        pbar = ProgBar().start_subtask("Colliders", len(three_tuples))

        for idx, (i, j, k) in enumerate(three_tuples):
            changed_pag = rules.rule1(pag, i, j, k)
            changed_pag |= rules.rule2(pag, i, j, k)
            for node in (node for node in pag if node not in [i, j, k]):
                changed_pag |= rules.rule3(pag, i, j, k, node)
                changed_pag |= rules.rule4(pag, i, j, k, node, sepSet)
                changed_pag |= rules.rule5(pag, i, j, k, node)
                changed_pag |= rules.rule67(pag, i, j, k)
                changed_pag |= rules.rule8(pag, i, j, k)
                changed_pag |= rules.rule9(pag, i, j, k, node)
                changed_pag |= rules.rule10(pag, i, j, k, node)
            pbar.update_subtask("Colliders", idx+1)

    pbar.remove("Colliders")
    if save_intermediate:
        save_graph(pag, "adjmat_FCI_ruled", data_file, output_path, log)
    return pag


def get_dsep_combs(dseps, x, y, i):
    """
    Get combinations of d-separating variables for a given variable pair.

    Parameters:
    dseps (dict): A dictionary containing d-separating variables for each variable.
    x (str): The first variable.
    y (str): The second variable.
    i (int): The number of variables to combine.

    Returns:
    list: A list of combinations of d-separating variables where "y" is not present.
    """
    dsep_combinations = list(
        itertools.combinations(dseps[x], i))
    # Keep only those combinations where "y" is not present
    dsep_combinations = [tuple(filter(lambda e: e != y, t))
                         for t in dsep_combinations]
    return list(filter(lambda comb: len(comb) != 0, dsep_combinations))


def orient_V(pag, sepSet, debug):
    """
    A function to orient the colliders in a PAG

    Parameters
    ----------
        pag: PAG
            PAG to be oriented
        sepSet: dict
            separation d#sets of all pairs of nodes in PAG
        debug: Debug
            Debugger.
    Returns
    -------
        PAG
            PAG with v-structures oriented
    """
    three_tuples = itertools.permutations(pag, 3)
    for i, j, k in three_tuples:
        if pag.has_edge(i, j) and pag.has_edge(k, j) and not pag.has_edge(i, k):
            if j not in sepSet[(i, k)]:
                pag.direct_edge(i, j)
                pag.direct_edge(k, j)
                debug.dbg(f"  - Orienting collider {i}, {j}, {k}")


@timeout_decorator.timeout(seconds=1)
def is_possible_d_sep(X, Y, pag, debug):
    """
    A function to test if one node is in the possibled sep set of another

    Parameters
    ----------
        X: (str) one node tested for d seperation
        Y: (str) other node to be tested
        pag: (PAG) PAG containing the nodes
        debug: (Debug) debugger
    Returns
    -------
        (bool) true if nodes possibly d-sep eachother
    """
    all_paths = nx.all_simple_paths(pag, X, Y, cutoff=len(pag)-3)
    for path in all_paths:
        path_sep = True
        debug.dbg(f" ({len(path[:-1])})", end="")
        for i in range(1, len(path[:-1])):
            collider = pag.has_directed_edge(
                path[i - 1], path[i]
            ) and pag.has_directed_edge(path[i + 1], path[i])
            triangle = (
                pag.has_edge(path[i - 1], path[i])
                and pag.has_edge(path[i + 1], path[i])
                and pag.has_edge(path[i - 1], path[i + 1])
            )
            if not (collider or triangle):
                path_sep = False
        if path_sep:
            debug.dbg(" ... True")
            return True
    debug.dbg(" ... False")
    return False


def possible_d_seps(pag, debug: Debug):
    """
    Method to construct the possible d-sep-set of a pag

    Parameters
    ----------
        pag: (PAG) PAG to find dseps for
        debug: (Debug) Debugger

    Returns
    -------
        dict
            keys: nodes
            values: nodes which could d-seperate other nodes from key node

    Raises
    ------
        TimeoutError
            If the timeout is reached.
    """
    dseps = dict((i, []) for i in pag)
    tuples = list(itertools.permutations(pag, 2))
    debug.dbg(f"Searching for possible d-seps ({len(tuples)})")
    for idx, (i, j) in enumerate(tuples):
        debug.dbg(f'  - Analyzing [{idx:>5}/{len(tuples):<5d}] ({i:<12s} - {j:>12s})',
                  end="")
        try:
            potential_dsep = is_possible_d_sep(i, j, pag, debug)
        except TimeoutError:
            debug.dbg(" ** Timeout **")
        else:
            if potential_dsep:
                dseps[i].append(j)
    return dseps


def get_neighbors(x, pag):
    neighbors = list(pag.neighbors(x))
    return neighbors
