import numpy as np
import pytest
from typing import Dict
import networkx as nx
from gadjid import sid

from causalgraph.metrics.SID import SID


def test_manual_SID():
    G = np.array([[0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    H1 = np.array([[0, 1, 1, 1, 1],
                   [0, 0, 1, 1, 1],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])
    H2 = np.array([[0, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])
    H3 = np.array([[0, 1, 1, 1, 1],
                   [1, 0, 1, 1, 1],
                   [1, 1, 0, 1, 0],
                   [1, 1, 1, 0, 0],
                   [1, 1, 0, 0, 0]])
    H4 = np.array([[0, 0, 1, 1, 1],
                   [0, 0, 0, 1, 0],
                   [1, 1, 0, 1, 0],
                   [0, 1, 0, 0, 1],
                   [1, 0, 1, 0, 0]])
    H5 = np.array([[0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])

    sid1 = SID(G, H1)
    assert sid1['sid'] == 0
    sid2 = SID(G, H2)
    assert sid2['sid'] == 8
    sid3 = SID(G, H3)
    assert sid3['sidLowerBound'] == 0
    assert sid3['sidUpperBound'] == 15
    sid4 = SID(G, H4)
    assert sid4['sidLowerBound'] == 8
    assert sid4['sidUpperBound'] == 16
    sid5 = SID(G, H5)
    assert sid5['sid'] == 12


def is_dag(adj_matrix):
    """
    Checks if the given adjacency matrix corresponds to a DAG using NetworkX.

    Args:
        adj_matrix (numpy.ndarray): The adjacency matrix of the graph.

    Returns:
        bool: True if the graph is a DAG, False otherwise.
    """
    # Create a directed graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Check if the graph is a DAG
    return nx.is_directed_acyclic_graph(G)


def compared_SID(trueGraph: np.ndarray, estGraph: np.ndarray) -> Dict[str, float]:
    """
    Compute the Structural Intervention Distance (SID) between two graphs.
    https://github.com/CausalDisco/gadjid

    This function calculates the SID between a true graph and an estimated graph.
    It first checks if the estimated graph is a DAG, and if not, returns a default
    result. It then ensures both input graphs are numpy arrays before proceeding
    with the SID calculation.

    Args:
        trueGraph (np.ndarray): The adjacency matrix of the true graph.
        estGraph (np.ndarray): The adjacency matrix of the estimated graph.

    Returns:
        Dict[str, float]: A dictionary containing the SID metrics:
            - 'sid': The computed SID value.
            - 'sidLowerBound': The lower bound of the SID.
            - 'sidUpperBound': The upper bound of the SID.
    """
    if not is_dag(estGraph):
        return {
            'sid': 0,
            'sidLowerBound': 0.0,
            'sidUpperBound': 0.0
        }
    # Check if trueGraph and estGraph are np.ndarrays, if not, convert them
    if not isinstance(trueGraph, np.ndarray):
        trueGraph = np.array(trueGraph)
    if not isinstance(estGraph, np.ndarray):
        estGraph = np.array(estGraph)

    # Convert to numpy matrices before calling sid
    trueGraph = np.matrix(trueGraph)
    estGraph = np.matrix(estGraph)

    s = sid(trueGraph, estGraph, edge_direction="from row to column")
    return {
        'sid': s[1],
        'sidLowerBound': s[0],
        'sidUpperBound': s[0]
    }


def random_dag(size, probability):
    """
    Generate a random Directed Acyclic Graph (DAG) as an adjacency matrix.

    This function creates a random DAG by following these steps:
    1. Generate a random binary matrix using binomial distribution.
    2. Make it upper triangular to ensure acyclicity.
    3. Randomly permute the rows and columns to distribute the edges.

    Args:
        size (int): The number of nodes in the DAG.
        probability (float): The probability of an edge between any two nodes.

    Returns:
        numpy.ndarray: An adjacency matrix representing the random DAG.
    """
    rng = np.random.default_rng(0)
    adj = rng.binomial(1, probability, size=(size, size)).astype(np.int8)
    adj = np.triu(adj, 1)
    perm = rng.permutation(size)

    return adj[perm, :][:, perm]


def test_compared_sid():
    """
    Test the consistency between SID and compared_SID functions.

    This test generates random DAGs, compares them using both SID and compared_SID
    functions, and checks if the results are consistent. It runs the comparison
    multiple times and reports the percentage of times the results match.

    The test process:
    1. Generate a random DAG (G)
    2. Create a modified version of G (H) by randomly removing some edges
    3. Compute SID between G and H using both SID() and compared_SID()
    4. Compare the results and count matches
    5. Repeat steps 1-4 multiple times
    6. Report the percentage of matching results

    No assertions are made; this test is primarily for observing consistency.
    """
    n = 20  # Number of test iterations
    threshold = 0.2
    count = 0
    for i in range(n):
        # Choose a random dimension "p" for the matrix
        p = np.random.randint(3, 20)

        G = random_dag(p, 0.2)
        H = G.copy()

        indices = np.where(G == 1)
        for i in range(len(indices[0])):
            if np.random.random() > threshold:
                H[indices[0][i]][indices[1][i]] = 0

        # Compute the SID between G and H
        sid1 = SID(G, H, output=False)
        sid2 = compared_SID(G, H)
        if sid1['sid'] == sid2['sid']:
            count += 1

    assert count/n == 1.0
