"""
This file contains all the methods related to colliders and d-separation sets.


"""
import itertools

import networkx as nx
import timeout_decorator
from tqdm.auto import tqdm

from causalgraph.estimators.fci import rules
from causalgraph.estimators.fci.debug import Debug
from causalgraph.estimators.fci.initialization import save_graph
from causalgraph.estimators.fci.pag import PAG


def init_pag(skeleton, sepSet, verbose, debug):
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
                save_intermediate: bool = False):
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
        pbar = tqdm(total=len(three_tuples),
                    disable=debug.verbose, leave=False)
        pbar.set_description("App.Rules")
        pbar.reset()
        for i, j, k in three_tuples:
            pbar.update(1)
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
            pbar.refresh()
        pbar.close()
    if save_intermediate:
        save_graph(pag, "adjmat_FCI_ruled", data_file, output_path, log)
    return pag


def get_dsep_combs(dseps, x, y, i):
    dsep_combinations = list(
        itertools.combinations(dseps[x], i))
    # Keep only those combinations where "y" is not present
    dsep_combinations = [tuple(filter(lambda e: e != y, t))
                         for t in dsep_combinations]
    return list(filter(lambda ℷ: len(ℷ) != 0, dsep_combinations))


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

    """
    dseps = dict((i, []) for i in pag)
    tuples = list(itertools.permutations(pag, 2))
    debug.dbg(f"Searching for possible d-seps ({len(tuples)})")
    for idx, (i, j) in enumerate(tuples):
        debug.dbg(f'  - Analyzing [{idx:>5}/{len(tuples):<5d}] ({i:<12s} - {j:>12s})',
                  end="")
        try:
            potential_dsep = is_possible_d_sep(i, j, pag, debug)
        except:
            debug.dbg(" ** Timeout **")
            pass
        else:
            if potential_dsep:
                dseps[i].append(j)
    return dseps


def get_neighbors(x, pag):
    neighbors = list(pag.neighbors(x))
    return neighbors
