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
from typing import Union

import networkx as nx
import pydot as pydot
import pydotplus


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
