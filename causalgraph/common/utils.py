#
# Utility functions for causalgraph
#
# (C) J. Renero, 2022, 2023
#

import glob
import os
import pickle
from os.path import join
from pathlib import Path
from typing import Dict, List, Tuple, Union

import networkx as nx
import pydot as pydot
import pydotplus
import torch


AnyGraph = Union[nx.Graph, nx.DiGraph]


def save_experiment(obj_name: str, folder: str, results: dict):
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

    Return:
        (str) The name under which the experiment has been saved
    """
    if not os.path.exists(folder):
        Path(folder).mkdir(parents=False, exist_ok=True)
    output = valid_output_name(obj_name, folder, extension="pickle")
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
    first_node = list(g1.nodes)[0]
    data_fields = g1.nodes[first_node].keys()
    nodes_data = {}
    for key in data_fields:
        for n in nodes:
            nodes_data[n] = {key: min(g1.nodes[n][key], g2.nodes[n][key])}
        
    # Get the edges from g1 and g2, and take only those that match completely
    # edges = set(g1.edges).intersection(set(g2.edges))
    edges = g1.edges & g2.edges
    
    # Create a new graph with the nodes, nodes_data and edges
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    nx.set_node_attributes(g, nodes_data)
    g.add_edges_from(edges)
    
    return g
