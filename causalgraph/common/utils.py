"""
Utility functions for causalgraph
(C) J. Renero, 2022, 2023
"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

import glob
import os
import pickle
import types
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import pydot
import pydotplus
import torch

from causalgraph.independence.edge_orientation import get_edge_orientation
from typing import List, Optional
import networkx as nx

AnyGraph = Union[nx.Graph, nx.DiGraph]


def save_experiment(
        obj_name: str,
        folder: str,
        results: object,
        overwrite: bool = False
):
    """
    Creates a folder for the experiment and saves results. Results is a
    dictionary that will be saved as an opaque pickle. When the experiment will
    require to be loaded, the only parameter needed are the folder name.

    Args:
        obj_name (str): the name to be given to the pickle file to be saved. If
            a file already exists with that name, a file with same name and a
            extension will be generated.
        folder (str): a full path to the folder where the experiment is to be saved.
            If the folder does not exist it will be created.
        results (obj): the object to be saved as experiment. This is typically a
            dictionary with different items representing different parts of the
            experiment.
        overwrite (bool): If True, then the experiment is saved even if a file
            with the same name already exists. If False, then the experiment is
            saved only if a file with the same name does not exist.

    Return:
        (str) The name under which the experiment has been saved
    """
    if not os.path.exists(folder):
        Path(folder).mkdir(parents=False, exist_ok=True)

    if not overwrite:
        output = valid_output_name(obj_name, folder, extension="pickle")
    else:
        output = join(folder, obj_name + ".pickle")

    with open(output, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output


def load_experiment(obj_name: str, folder: str):
    """
    Loads a pickle from a given folder name. It is not necessary to add the "pickle"
    extension to the experiment name.

    Parameters:
    -----------
    obj_name (str): The name of the object saved in pickle format that is to be loaded.
    folder (str): A full path where looking for the experiment object.

    Returns:
    --------
    An obj loaded from a pickle file.
    """
    if Path(obj_name).suffix == "" or Path(obj_name).suffix != "pickle":
        ext = ".pickle"
    else:
        ext = ''

    experiment = f"{str(Path(folder, obj_name))}{ext}"
    with open(experiment, 'rb') as h:
        obj = pickle.load(h)
    return obj


def valid_output_name(filename: str, path: str, extension=None) -> str:
    """
    Builds a valid name. In case there's another file which already exists
    adds a number (1, 2, ...) until finds a valid filename which does not
    exist.

    Returns
    -------
    The filename if the name is valid and file does not exists,
            None otherwise.

    Params
    ------
    filename: str
        The base filename to be set.
    path: str
        The path where trying to set the filename
    extension:str
        The extension of the file, without the dot '.' If no extension is
        specified, any extension is searched to avoid returning a filepath
        of an existing file, no matter what extension it has.
    """
    if extension:
        if extension[0] == '.':
            extension = extension[1:]
        base_filepath = join(path, filename) + '.{}'.format(extension)
    else:
        base_filepath = join(path, filename)
    output_filepath = base_filepath
    idx = 1
    while len(glob.glob(f"{output_filepath}*")) > 0:
        if extension:
            output_filepath = join(
                path, filename) + '_{:d}.{}'.format(
                idx, extension)
        else:
            output_filepath = join(path, filename + '_{}'.format(idx))
        idx += 1

    return output_filepath


def graph_from_dot_file(dot_file: Union[str, Path]) -> nx.DiGraph:
    """ Returns a NetworkX DiGraph from a DOT FILE. """
    dot_object = pydot.graph_from_dot_file(dot_file)
    dot_string = dot_object[0].to_string()
    dot_string = dot_string.replace('\"\\n\"', '')
    dot_string = dot_string.replace("\n;\n", "\n")
    dotplus = pydotplus.graph_from_dot_data(dot_string)
    dotplus.set_strict(True)
    final_graph = nx.nx_pydot.from_pydot(dotplus)
    if '\\n' in final_graph.nodes:
        final_graph.remove_node('\\n')

    return final_graph


def graph_from_dictionary(d: Dict[str, List[Union[str, Tuple[str, float]]]]) -> AnyGraph:
    """
    Builds a graph from a dictionary like {'u': ['v', 'w'], 'x': ['y']}.
    The elements of the list can be tuples including weight

    Args:
        d (dict): A dictionary of the form {'u': ['v', 'w'], 'x': ['y']}, where
            an edge between 'u' goes towards 'v' and 'w', and also an edge from
            'x' goes towards 'y'. The format can also be like:
            {'u': [('v', 0.2), ('w', 0.7)], 'x': [('y', 0.5)]}, where the values
            in the tuple are interpreted as weights.

    Returns:
        networkx.DiGraph with the nodes and edges specified.
    """
    g = nx.DiGraph()
    for node, parents in d.items():
        if len(parents) > 0:
            if type(parents[0]) == tuple:
                for parent, weight in parents:
                    g.add_edge(parent, node, weight=weight)
            else:
                for parent in parents:
                    g.add_edge(parent, node)
    return g


def select_device(force: str = None) -> str:
    """
    Selects the device to be used for training. If force is not None, then
    the device is forced to be the one specified. If force is None, then
    the device is selected based on the availability of GPUs. If no GPUs are
    available, then the CPU is selected.

    Args:
        force (str): If not None, then the device is forced to be the one
            specified. If None, then the device is selected based on the
            availability of GPUs. If no GPUs are available, then the CPU is
            selected.

    Returns:
        (str) The device to be used for training.

    Raises:
        ValueError: If the forced device is not available or not a valid device.
    """
    if force is not None:
        if force in ['cuda', 'mps', 'cpu']:
            if force == 'cuda' and torch.cuda.is_available():
                device = force
            elif force == 'mps' and torch.backends.mps.is_available():
                device = force
            elif force == 'mps' and not torch.backends.mps.is_available():
                device = 'cpu'
            elif force == 'cpu':
                device = force
            else:
                raise ValueError(f"Invalid device: {force}")
        else:
            raise ValueError(f"Invalid device: {force}")
    else:
        if torch.cuda.is_available() and torch.backends.cuda.is_built():
            device = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"
    return device


def graph_intersection(g1: AnyGraph, g2: AnyGraph) -> AnyGraph:
    """
    Returns the intersection of two graphs. The intersection is defined as the
    set of nodes and edges that are common to both graphs. The intersection is
    performed on the nodes and edges, not on the attributes of the nodes and
    edges.

    Args:
        g1 (networkx.DiGraph): The first graph.
        g2 (networkx.DiGraph): The second graph.

    Returns:
        (networkx.DiGraph) The intersection of the two graphs.
    """
    # Get the nodes from g1 and g2, as the union of both
    nodes = set(g1.nodes).intersection(set(g2.nodes))

    # Check if any of the graphs is empty
    if len(nodes) == 0:
        return nx.DiGraph()

    # Take the data from the nodes in g1 and g2, as the minimum value of both
    # first_node = list(g1.nodes)[0]
    # data_fields = g1.nodes[first_node].keys()
    # nodes_data = {}
    # for key in data_fields:
    #     for n in nodes:
    #         nodes_data[n] = {key: min(g1.nodes[n][key], g2.nodes[n][key])}

    # Get the edges from g1 and g2, and take only those that match completely
    # edges = set(g1.edges).intersection(set(g2.edges))
    edges = g1.edges & g2.edges

    # Create a new graph with the nodes, nodes_data and edges
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    # nx.set_node_attributes(g, nodes_data)
    g.add_edges_from(edges)

    return g


def graph_union(g1: AnyGraph, g2: AnyGraph) -> AnyGraph:
    """
    Returns the union of two graphs. The union is defined as the set of nodes and
    edges that are in both graphs. The union is performed on the nodes and edges,
    not on the attributes of the nodes and edges.
    """
    nodes = set(g1.nodes).union(set(g2.nodes))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(g1.edges)
    g.add_edges_from(g2.edges)

    # Take the data from the nodes in g1 and g2, as the minimum value of both
    # first_node = list(g1.nodes)[0]
    # data_fields = g1.nodes[first_node].keys()
    # nodes_data = {}
    # for key in data_fields:
    #     for n in nodes:
    #         nodes_data[n] = {key: min(g1.nodes[n][key], g2.nodes[n][key])}

    return g


def digraph_from_connected_features(
        X,
        feature_names,
        models,
        connections,
        root_causes,
        prior: Optional[List[List[str]]] = None,
        reciprocity=True,
        anm_iterations=10,
        max_anm_samples=400,
        verbose=False):
    """
    Builds a directed graph from a set of features and their connections. The
    connections are determined by the SHAP values of the features. The
    orientation of the edges is determined by the causal direction of the
    features. The orientation is determined by the method of edge orientation
    defined in the independence module.

    Parameters:
    -----------
    X (numpy.ndarray): The input data.
    feature_names (list): The list of feature names.
    models (obj): The model used to explain the data.
    connections (dict): The dictionary of connections between features. The
        dictionary is of the form {'feature1': ['feature2', 'feature3'], ...}
    root_causes (list): The list of root causes. The root causes are the
        features that are not caused by any other feature.
    reciprocity (bool): If True, then the edges are oriented only if the
        direction is reciprocal. If False, then the edges are oriented
        regardless of the reciprocity.
    anm_iterations=10 (int): The number of iterations to be used by the independence
        module to determine the orientation of the edges.
    max_anm_samples=400 (int): The maximum number of samples to be used by the
        independence module to determine the orientation of the edges. If the
        number of samples is larger than the number of samples in the data, then
        the number of samples is set to the number of samples in the data.
    verbose (bool): If True, then the function prints information about the
        orientation of the edges.

    Returns:
    --------
    networkx.DiGraph: The directed graph with the nodes and edges determined
        by the connections and the orientation determined by the independence
        module.

    """
    if verbose:
        print("-----\ndigraph_from_connected_features()")
    unoriented_graph = nx.Graph()
    for target in feature_names:
        for peer in connections[target]:
            # Add edges ONLY between nodes where SHAP recognizes both directions
            if reciprocity:
                if target in connections[peer]:
                    unoriented_graph.add_edge(target, peer)
            else:
                unoriented_graph.add_edge(target, peer)

    dag = nx.DiGraph()
    # Add regression mean score to each node
    for i, feature in enumerate(feature_names):
        dag.add_node(feature)
        dag.nodes[feature]['regr_score'] = models.scoring[i]

    # Determine edge orientation for each edge
    if X.shape[0] > max_anm_samples:
        X = X.sample(max_anm_samples)
        if verbose:
            print(f"Reduced number of samples to {max_anm_samples}")
    for u, v in unoriented_graph.edges():
        # Set orientation to 0 ~ unknown
        orientation = 0

        # Check if we can set it from prior knowledge
        if prior:
            print(f"Checking edge {u} -> {v}...") if verbose else None
            idx_u = [i for i, l in enumerate(prior) if u in l][0]
            idx_v = [i for i, l in enumerate(prior) if v in l][0]
            both_in_top_list = idx_u == 0 and idx_v == 0
            u_is_before_v = idx_u - idx_v < 0
            v_is_before_u = idx_v - idx_u < 0
            if both_in_top_list:
                print(
                    f"Edge {u} -x- {v} removed: both top list") if verbose else None
                orientation = +1
                continue
            elif u_is_before_v:
                print(
                    f"Edge {u} -> {v} added: {u} before {v}") if verbose else None
                dag.add_edge(u, v)
                continue
            elif v_is_before_u:
                print(
                    f"Edge {v} -> {u} added: {v} before {u}") if verbose else None
                dag.add_edge(v, u)
                continue
            else:
                pass

        if orientation == 0:
            print(f"Checking edge {u} -> {v}...") if verbose else None
            orientation = get_edge_orientation(
                X, u, v, iters=anm_iterations, method="gpr", verbose=verbose)
            if orientation == +1:
                print(f"  Edge {u} -> {v} added from ANM") if verbose else None
                dag.add_edge(u, v)
            elif orientation == -1:
                print(f"  Edge {v} -> {u} added from ANM") if verbose else None
                dag.add_edge(v, u)
            else:
                pass

    # Apply a simple fix: if quality of regression for a feature is poor, then
    # that feature can be considered a root or parent node. Therefore, we cannot
    # have an edge pointing to that node.
    changes = []
    for parent_node in root_causes:
        print(f"Checking root cause {parent_node}...") if verbose else None
        for cause, effect in dag.edges():
            if effect == parent_node:
                print(
                    f"Reverting edge {cause} -> {effect}") if verbose else None
                changes.append((cause, effect))
    for cause, effect in changes:
        dag.remove_edge(cause, effect)
        dag.add_edge(effect, cause)

    return dag


def valid_candidates_from_prior(feature_names, effect, prior):
    """
    This method returns the valid candidates for a given effect, based on the
    prior information. The prior information is a list of lists, where each list
    contains the features that are known to be in the same level of the hierarchy.

    Parameters:
    -----------
    feature_names: List[str]
        The list of feature names.
    effect: str
        The effect for which to find the valid candidates.
    prior: List[List[str]]
        The prior information about the hierarchy of the features.

    Returns:
    --------
    List[str]
        The valid candidates for the given effect.
    """
    if prior is None:
        return [c for c in feature_names if c != effect]

    # identify in what list is the effect, from the list of lists defined in prior
    idx = [i for i, sublist in enumerate(prior) if effect in sublist]
    if not idx:
        raise ValueError(f"Effect '{effect}' not found in prior")

    # candidates are elements in the list ' idx' and all the lists before it
    candidates = [item for sublist in prior[:idx[0] + 1]
                  for item in sublist]

    # return the candidates, excluding the effect itself
    return [c for c in candidates if c != effect]


def break_cycles_using_prior(
        original_dag: nx.DiGraph,
        prior: List[List[str]],
        verbose: bool = False) -> nx.DiGraph:
    """
    This method remove potential edges in the DAG by removing edges that are
    incompatible with the temporal relationship defined in the prior knowledge.
    Any edge pointing backwards, according to the order established in the prior
    knowledge, is removed.

    Parameters:
    -----------
    original_dag (nx.DiGraph): The original directed acyclic graph.
    prior (List[List[str]]): A list of lists containing the prior knowledge
        about the edges in the DAG. The lists define a hierarchy of edges that
        represent a temporal relation in cause and effect. If a node is in the first
        list, then it is a root cause. If a node is in the second list, then it is
        caused by the nodes in the first list or the second list, and so on.
    verbose (bool): If True, then the method prints information about the
        edges that are removed.

    Returns:
    --------
    nx.DiGraph: The directed acyclic graph with the edges removed according to
        the prior knowledge.
    """
    dag = nx.DiGraph()
    dag.add_edges_from(original_dag.edges)

    if verbose:
        print("Prior knowledge:", prior)
    cycles = list(nx.simple_cycles(dag))
    while len(cycles) > 0:
        cycle = cycles.pop(0)
        if verbose:
            print("Checking cycle", cycle)
        for i, node in enumerate(cycle):
            if node == cycle[-1]:
                neighbour = cycle[0]
            else:
                neighbour = cycle[i+1]
            # Check if the edge between the two nodes is in the prior knowledge
            if [node, neighbour] not in prior:
                if verbose:
                    print(
                        f"↳ Checking '{node} -> {neighbour}' in the prior knowledge")
                idx_node = [i for i, l in enumerate(prior) if node in l][0]
                idx_neighbour = [i for i, l in enumerate(
                    prior) if neighbour in l][0]
                if idx_neighbour - idx_node < 0:
                    if verbose:
                        print(
                            f"  ↳ Breaking cycle, removing edge {node} -> {neighbour}")
                        print("** Recomputing cycles **")
                    dag.remove_edge(node, neighbour)
                    cycles = list(nx.simple_cycles(dag))
                    break
    return dag


def potential_misoriented_edges(
        loop: List[str],
        discrepancies: Dict,
        verbose: bool = False):
    """
    Find potential misoriented edges in a loop based on discrepancies.

    Args:
        loop (List[str]): The list of nodes in the loop.
        discrepancies (Dict): A dictionary containing discrepancies between nodes.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        List[Tuple[str, str, float]]: A list of potential misoriented edges,
            sorted by the difference in goodness-of-fit scores.

    """
    potential_misoriented_edges = []
    for i, node in enumerate(loop):
        if i == len(loop) - 1:
            neighbor = loop[0]
        else:
            neighbor = loop[i + 1]
        fwd_gof = 1. - discrepancies[neighbor][node].shap_gof
        bwd_gof = 1. - discrepancies[node][neighbor].shap_gof
        orientation = +1 if fwd_gof < bwd_gof else -1
        if orientation == -1:
            potential_misoriented_edges.append((node, neighbor, fwd_gof - bwd_gof))
        if verbose:
            direction = "-->" if fwd_gof < bwd_gof else "<--"
            print(
                f"Edge: {node}{direction}{neighbor} FWD:{fwd_gof:.3f} "
                f"BWD:{bwd_gof:.3f}")

    return sorted(potential_misoriented_edges, key=lambda x: x[2], reverse=True)


def break_cycles_if_present(
        dag: nx.DiGraph,
        discrepancies: Dict,
        prior: Optional[List[List[str]]] = None,
        verbose: bool = False):
    """
    Breaks cycles in a directed acyclic graph (DAG) by removing the edge with
    the lowest goodness of fit (R2). If there are multiple cycles, they are
    all traversed and fixed.

    Parameters:
    - dag (nx.DiGraph): the DAG to break cycles in.
    - knowledge (pd.DataFrame): a DataFrame containing the permutation importances
        for each edge in the DAG.
    - prior (List[List[str]]): a list of lists containing the prior knowledge
        about the edges in the DAG. The lists define a hierarchy of edges that
        represent a temporal relation in cause and effect. If a node is in the first
        list, then it is a root cause. If a node is in the second list, then it is
        caused by the nodes in the first list or the second list, and so on.

    Returns:
    - dag (nx.DiGraph): the DAG with cycles broken.
    """
    if verbose:
        print("-----\nbreak_cycles_if_present()")

    # If prior is set, then break cycles using the prior knowledge
    if prior:
        new_dag = break_cycles_using_prior(dag, prior, verbose)
    else:
        new_dag = dag.copy()

    # Find all cycles in the DAG
    cycles = list(nx.simple_cycles(new_dag))

    # This might be important, to remove first double edges
    cycles.sort(key=len)
    if len(cycles) == 0:
        if verbose:
            print("No cycles found")
        return new_dag

    # Traverse all cycles, fixing them
    cycles_info = []
    while len(cycles) > 0:
        cycle = cycles.pop(0)
        # For every pair of consecutive nodes in the cycle, store its SHAP discrepancy
        cycle_edges = {}
        for node in cycle:
            if node == cycle[-1]:
                neighbour = cycle[0]
            else:
                neighbour = cycle[cycle.index(node)+1]
            cycle_edges[(node, neighbour)
                        ] = discrepancies[neighbour][node].shap_gof
        cycles_info.append((cycle, cycle_edges))

        # Find the edge with the lowest SHAP discrepancy
        min_gof = min(cycle_edges.values())
        min_edge = [edge for edge, gof in cycle_edges.items() if gof ==
                    min_gof][0]

        # Check if any of the edges in the loop is potentially misoriented.
        # If so, change the orientation of the edge with the lowest SHAP discrepancy
        potential_misoriented = potential_misoriented_edges(
            cycle, discrepancies, verbose)
        if len(potential_misoriented) > 0:
            if verbose:
                print("Potential misoriented edges in the cycle:")
                for edge in potential_misoriented:
                    print(f"  {edge[0]} --> {edge[1]} ({edge[2]:.3f})")
            # Check if changing the orientation of the edge would break the cycle
            test_dag = new_dag.copy()
            test_dag.remove_edge(*min_edge)
            test_dag.add_edge(*potential_misoriented[0][1::-1])
            test_cycles = list(nx.simple_cycles(test_dag))
            if len(test_cycles) == len(cycles):
                if verbose:
                    print(
                        f"Breaking cycle {cycle} by changing orientation of edge {min_edge}")
                new_dag.remove_edge(*min_edge)
                new_dag.add_edge(*potential_misoriented[0][1::-1])
                continue

        # Remove the edge with the lowest SHAP discrepancy, checking that
        # the edge still exists
        if min_edge in new_dag.edges:
            if verbose:
                print(f"Breaking cycle {cycle} by removing edge {min_edge}")
            new_dag.remove_edge(*min_edge)

        # Recompute whether there're cycles in the DAG after this change
        cycles = list(nx.simple_cycles(new_dag))
        cycles.sort(key=len)

    return new_dag


def graph_from_adjacency(
        adjacency: np.ndarray,
        node_labels=None,
        th=0.0,
        inverse: bool = False,
        absolute_values: bool = False
) -> nx.DiGraph:
    """
    Manually parse the adj matrix to shape a dot graph

    Args:
        adjacency: a numpy adjacency matrix
        node_labels: an array of same length as nr of columns in the adjacency
            matrix containing the labels to use with every node.
        th: (float) weight threshold to be considered a valid edge.
        inverse (bool): Set to true if rows in adjacency reflects where edges are
            comming from, instead of where are they going to.
        absolute_values: Take absolute value of weight label to check if its greater
            than the threshold.

    Returns:
         The Graph (DiGraph)
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(adjacency.shape[1]))

    # What to do with absolute values?
    def not_abs(x):
        return x

    w_val = np.abs if absolute_values else not_abs

    def weight_gt(w, thresh):
        return w != 0.0 if thresh is None else w_val(w) > thresh

    # Do I have a threshold to consider?
    for i, row in enumerate(adjacency):
        for j, value in enumerate(row):
            if inverse:
                if weight_gt(adjacency[j][i], th):
                    G.add_edge(i, j, weight=w_val(adjacency[j][i]))
            else:
                if weight_gt(value, th):
                    # , arrowhead="normal")
                    G.add_edge(i, j, weight=w_val(value))
    # Map the current column numbers to the letters used in toy dataset
    if node_labels is not None and len(node_labels) == adjacency.shape[1]:
        mapping = dict(zip(sorted(G), node_labels))
        G = nx.relabel_nodes(G, mapping)

    return G


def graph_from_adjacency_file(file: Union[Path, str], th=0.0) -> Tuple[
        nx.DiGraph, pd.DataFrame]:
    """
    Read Adjacency matrix from a file and return a Graph

    Args:
        file: (str) the full path of the file to read
        th: (float) weight threshold to be considered a valid edge.
    Returns:
        DiGraph, DataFrame
    """
    df = pd.read_csv(file, dtype="str")
    df = df.astype("float64")
    labels = list(df)
    G = graph_from_adjacency(df.values, node_labels=labels, th=th)

    return G, df


def graph_to_adjacency(graph: AnyGraph,
                       node_names: Optional[List[str]] = None,
                       weight_label: str = "weight") -> np.ndarray:
    """
    A method to generate the adjacency matrix of the graph. Labels are
    sorted for better readability.

    Args:
        graph: (Union[Graph, DiGraph]) the graph to be converted.
        weight_label: the label used to identify the weights.

    Return:
        graph: (numpy.ndarray) A 2d array containing the adjacency matrix of
            the graph.
    """
    symbol_map = {"o": 1, ">": 2, "-": 3}
    labels = sorted(list(graph.nodes))  # [node for node in self]
    # Double check if all nodes are in the graph
    if node_names is not None:
        for n in list(node_names):
            if n not in set(labels):
                labels.append(n)
        labels = sorted(labels)
    # Fix for the case where an empty node is parsed from the .dot file
    if '\\n' in labels:
        labels.remove('\\n')
    mat = np.zeros((len(labels), (len(labels))))
    for x in labels:
        for y in labels:
            if graph.has_edge(x, y):
                if bool(graph.get_edge_data(x, y)):
                    if y in graph.get_edge_data(x, y).keys():
                        mat[labels.index(x)][labels.index(y)] = symbol_map[
                            graph.get_edge_data(x, y)[y]
                        ]
                    else:
                        mat[labels.index(x)][labels.index(y)] = graph.get_edge_data(
                            x, y
                        )[weight_label]
                else:
                    mat[labels.index(x)][labels.index(y)] = 1
    mat[np.isnan(mat)] = 0
    return mat


def graph_to_adjacency_file(graph: AnyGraph, output_file: Union[Path, str]):
    """
    A method to write the adjacency matrix of the graph to a file. If graph has
    weights, these are the values stored in the adjacency matrix.

    Args:
        graph: (Union[Graph, DiGraph] the graph to be saved
        output_file: (str) The full path where graph is to be saved
    """
    mat = graph_to_adjacency(graph)
    labels = sorted(list(graph.nodes))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(",".join([f"{label}" for label in labels]))
        f.write("\n")
        for i, label in enumerate(labels):
            f.write(f"{label}")
            f.write(",")
            f.write(",".join([str(point) for point in mat[i]]))
            f.write("\n")


def stringfy_object(object_: object) -> str:
    """
    Convert an object into a string representation, including its attributes.

    Args:
        object_ (object): The object to be converted.

    Returns:
        str: A string representation of the object and its attributes.
    """
    forbidden_attrs = [
        'fit', 'predict', 'fit_predict', 'score', 'get_params', 'set_params']
    ret = "Object attributes\n"
    ret += f"{'-'*80}\n"
    for attr in dir(object_):
        if attr.startswith('_') or \
            attr in forbidden_attrs or \
                isinstance(getattr(object_, attr), types.MethodType):
            continue
        if isinstance(getattr(object_, attr), (pd.DataFrame, np.ndarray)):
            ret += f"{attr:25} {type(getattr(object_, attr)).__name__} "
            ret += f"{getattr(object_, attr).shape}\n"
            continue
        if isinstance(getattr(object_, attr), nx.DiGraph):
            n_nodes = getattr(object_, attr).number_of_nodes()
            n_edges = getattr(object_, attr).number_of_edges()
            ret += f"{attr:25} {n_nodes} nodes, {n_edges} edges\n"
            continue
        if isinstance(getattr(object_, attr), dict):
            keys_list = [
                f"{k}:{type(getattr(object_, attr)[k])}"
                for k in getattr(object_, attr).keys()
            ]
            ret += f"{attr:25} dict {str(keys_list)[:60]}...\n"
            continue
        if isinstance(getattr(object_, attr), list):
            list_ = '[' + ', '.join(str(e)
                                    for e in getattr(object_, attr)) + ']'
            ret += f"{attr:25} list {list_[:60]}...\n"
            continue
        if isinstance(getattr(object_, attr), (int, float, str, bool)):
            ret += f"{attr:25} {getattr(object_, attr)}\n"
            continue
        ret += f"{attr:25} <{getattr(object_, attr).__class__.__name__}>\n"

    return ret
