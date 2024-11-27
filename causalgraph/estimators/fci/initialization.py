"""
This module contains the functions for saving the output of the FCI algorithm.
"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

import csv
from pathlib import Path
from typing import Dict, Union

from networkx import DiGraph, Graph

# from causalgraph.common.utils import graph_to_adjacency_file, valid_output_name
from ...common import utils
from causalgraph.estimators.fci.pag import PAG


def get_output_name(
        name: str,
        data_file: str,
        output_path: str) -> str:
    """
    Generates the output file name based on the input parameters.

    Args:
        name (str): The name of the output file.
        data_file (str): The path to the data file.
        output_path (str): The path to the output directory.

    Returns:
        str: The generated output file name.
    """
    output_name = f"{name}_{Path(data_file).stem}"
    output_file = f"{str(Path(output_path) / output_name)}.csv"
    if Path(output_file).is_file():
        output_file = utils.valid_output_name(
            output_name, str(Path(output_path)), "csv")
    return output_file


def save_graph(
    graph: Union[Graph, DiGraph, PAG],
    prefix: str,
    data_file: str,
    output_path: str,
    log=None,
):
    """
    Save the graph to file

    Args:
        graph: (Union[Graph, DiGraph]), the graph to be saved
        prefix: (str), the prefix to be used for the file where output is to be saved.
        cfg: (Configuration), internal cfg
        log: the logger object to use
    """
    output_file = get_output_name(prefix, data_file, output_path)
    utils.graph_to_adjacency_file(graph, output_file)
    if log:
        log.info(output_file)


def save_sepset(dsep_set: Dict, prefix: str, data_file, output_path, log=None):
    """
    Save the separation sets to file

    Args:
        dsep_set: (Dict), the separation sets
        prefix: (str), the prefix to be used for the file where output is to be saved.
        cfg: (Configuration), internal cfg
        log: the log object to use
    """
    sepset_file = get_output_name(prefix, data_file, output_path)
    dsep_set_to_csv(dsep_set, sepset_file)
    if log:
        log.info(sepset_file)


def dsep_set_from_csv(ss_file):
    """
    Reads a d-separation set from a CSV file, with the format:
    "from","to",comma-separated list of nodes
    :param ss_file: str
        the file to be used for importing the dsep-set
    :return: dictionary
        the dict with the sep set, where the key is the tuple of
        nodes for which there exists a d-sep set.
    """
    sepSet = dict()
    with open(ss_file, 'r', encoding="utf-8") as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if rows[1] != '':
                sepSet[(rows[0], rows[1])] = tuple(rows[2])
            else:
                sepSet[(rows[0], rows[1])] = None
    return sepSet


def dsep_set_to_csv(sepSet_c, ss_file):
    """
    Saves a dsep set to a CSV file.
    :param sepSet_c: dict
    :param ss_file: str with the name of the output file
    :return: None
    """
    with open(ss_file, 'w', encoding="utf-8") as outfile:
        for k in sepSet_c.keys():
            outfile.write(f"{k[0]},{k[1]},{''.join(sepSet_c[k])}\n")
