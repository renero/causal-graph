"""
Directed Edges (A → B):
    This edge type indicates a direct causal effect from node A to node B.
    It means that A is a direct cause of B.
Bidirected Edges (A ↔ B):
    This edge suggests that there is a common cause (confounder) affecting both A and B.
    It is usually used to indicate that A and B have a correlation due to a latent 
    variable.
Partially Directed Edges (A o-> B or A <-o B):
    These edges represent uncertainty about the direction of causation.
    A o-> B indicates that there might be a causal effect from A to B, but it's not 
    certain.
    A <-o B indicates that there might be a causal effect from B to A, but it's not 
    certain.
Nondirected Edges (A -- B):
    This edge type represents an uncertain relationship between A and B.
    It's unclear whether there is a direct cause, a common cause, or some other type 
    of relationship.
Partially Bidirected Edges (A o-o B):
    This edge suggests that there is uncertainty about whether there is a direct causal 
    effect, a common cause, or both.
    It represents a high level of uncertainty about the nature of the relationship 
    between A and B.

"""

# pylint: disable=E1101:no-member
# pylint: disable=W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=W0106:expression-not-assigned
# pylint: disable=C0103:invalid-name, C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, R0902:too-many-instance-attributes
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=R1702:too-many-branches

from typing import List

import networkx as nx
import numpy as np


class PAG(nx.Graph):
    """
    A class implementaing a graph which can have
    all edge types of a partial ancestral graph
    """

    def has_partially_directed_edge(self, u, v):
        """
        A method to check the graph for a partially directed edge
        (an edge with a o tag on the to node)

        Parameters
        ----------
        u : str
            the from node of the edge
        v: str
            the to node of the edge

        Returns
        -------
        bool
            True if has partially directed edge, false if not

        """
        if super().has_edge(u, v):
            tags = super().get_edge_data(u, v)
            if tags[u] == 'o' and tags[v] == ">":
                return True
            return False
        return False

    def has_bidirected_edge(self, u, v):
        """
        A method to check the graph for a bidirectional edge
        (an edge with a > tag on the to node and
        a > tag on the from node)

        Parameters
        ----------
        u : str
            the from node of the edge
        v: str
            the to node of the edge

        Returns
        -------
        bool
            True if has bidirectional edge, false if not

        """
        if super().has_edge(u, v):
            tags = super().get_edge_data(u, v)
            if tags[u] == ">" and tags[v] == ">":
                return True
            return False
        return False

    def has_partially_bidirected_edge(self, u, v):
        """
        A method to check the graph for a partially bidirectional edge
        (an edge with a o tag on the to node and
        a > tag on the from node)

        Parameters
        ----------
        u : str
            the from node of the edge
        v: str
            the to node of the edge

        Returns
        -------
        bool
            True if has partially bidirectional edge, false if not

        """
        if super().has_edge(u, v):
            tags = super().get_edge_data(u, v)
            if tags[u] == "o" and tags[v] == "o":
                return True
            return False
        return False

    def has_fully_directed_edge(self, u, v):
        """
        A method to check the graph for a fully directed edge
        (an edge with a > tag on the to node and
        a - tag on the from node)

        Parameters
        ----------
        u : str
            the from node of the edge
        v: str
            the to node of the edge

        Returns
        -------
        bool
            True if has fully directed edge, false if not

        """
        if super().has_edge(u, v):
            tags = super().get_edge_data(u, v)
            if tags[u] == "-" and tags[v] == ">":
                return True
            else:
                return False
        else:
            return False

    def has_fully_undirected_edge(self, u, v):
        """
        A method to check the graph for a fully undirected edge
        (an edge with a - tag on the to node and
        a - tag on the from node)

        Parameters
        ----------
        u : str
            the from node of the edge
        v: str
            the to node of the edge

        Returns
        -------
        bool
            True if has undirected edge, false if not
        """
        if super().has_edge(u, v):
            tags = super().get_edge_data(u, v)
            if tags[u] == "-" and tags[v] == "-":
                return True
            else:
                return False
        else:
            return False

    def has_directed_edge(self, u, v):
        """
        A method to check the graph for a directed edge
        (an edge with a > tag on the to node)

        Parameters
        ----------
        u : str
            the from node of the edge
        v: str
            the to node of the edge

        Returns
        -------
        bool
            True if has directed edge, false if not

        """
        if super().has_edge(u, v):
            tags = super().get_edge_data(u, v)
            if tags[v] == ">":
                return True
            return False
        return False

    def add_edge(self, u, v, utag="o", vtag="o"):
        """
        A method to add an edge to a graph

        Parameters
        ----------
        u : str
            the from node of an edge
        v: str
            the to node of an edge
        utag: str
            the tag of the from node of an edge
        vtag: str
            the tag of the to node of an edge
        """
        super().add_edges_from([(u, v, {u: utag, v: vtag})])

    def add_edges_from(self, bunch, **attr):
        """
        A method to add a list of edges with o tags to a graph

        Parameters
        ----------
        bunch: iterable
            a list of pairs of nodes which are edges
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.
        """
        for edge in bunch:
            super().add_edges_from(
                [(*edge, {edge[0]: "o", edge[1]: "o"})], **attr)

    def setTag(self, edge, node, tag):
        """
        A method to set the tag of one side of an edge

        Parameters
        ----------
        edge : [str,str]
            the two nodes of the edge to modify
        node: str
            the side of the edge to modify
        tag: str
            the new tag of the edge, in [>,-,o]
        """
        if super().has_edge(*edge):
            self.edges[edge][node] = tag

    def direct_edge(self, u, v):
        """
        A method to direct an edge in a graph (set the tag of the to node to >)

        Parameters
        ----------
        u : str
            the from node of the edge
        v: str
            the to node of the edge

        """
        if super().has_edge(u, v):
            self.edges[(u, v)][v] = ">"

    def fully_direct_edge(self, u, v):
        """
        A method to fully direct an edge in a graph
        (set the tag of the to node to > and the from node to -)

        Parameters
        ----------
        u : str
            the from node of the edge
        v: str
            the to node of the edge

        """
        if super().has_edge(u, v):
            self.edges[(u, v)][v] = ">"
            self.edges[(u, v)][u] = "-"

    def undirect_edge(self, u, v):
        """
        A method to fully undirect an edge in a graph
        (set the tag of the to node to - and the from node to -)
        Parameters
        ----------
        u : str
            the from node of the edge
        v: str
            the to node of the edge

        """
        if super().has_edge(u, v):
            self.edges[(u, v)][v] = "-"
            self.edges[(u, v)][u] = "-"

    def hasDiscPath(self, u, v, b):
        """
        A method to see if the pag has a discriminating path
        between two nodes
        Parameters
        ----------
        u : str
            the from node of the path
        v: str
            the to node of the path
        b: str
            the penultimate node of the disc path

        Returns
        -------
        bool
            True if there is a disc path, false if not

        """
        all_paths = nx.all_simple_paths(self, u, v)
        for path in all_paths:
            if b in path:
                b_pred = (path.index(v) - path.index(b)) == 1 and self.has_edge(
                    b, v)
                all_colliders = True
                for node in path[1:-1]:
                    prev = path[path.index(node) - 1]
                    suc = path[path.index(node) + 1]
                    if (node != b) and not (
                            (self.has_directed_edge(prev, node))
                            and (self.has_directed_edge(suc, node))
                    ):
                        all_colliders = False
                all_pred = True
                for node in path[1:-2]:
                    if not (self.has_directed_edge(node, v)):
                        all_pred = False
                nonadj = not self.has_edge(u, v)
                if b_pred and all_colliders and all_pred and nonadj:
                    return True
        return False

    def findDiscPath(self, u, v, b):
        """
        A method find all discriminating paths
        between two nodes in the pag
        Parameters
        ----------
        u : str
            the from node of the path
        v: str
            the to node of the path
        b: str
            the penultimate node of the disc path

        Returns
        -------
        str[][]
            List of discriminating paths in pag between the two nodes on the
            third node
        """
        all_paths = nx.all_simple_paths(self, u, v)
        discpaths = []
        for path in all_paths:
            if b in path:
                b_pred = (path.index(v) - path.index(b)) == 1 and self.has_edge(
                    b, v)
                all_colliders = True
                for node in path[1:-1]:
                    prev = path[path.index(node) - 1]
                    suc = path[path.index(node) + 1]
                    if (node != b) and not (
                            (self.has_directed_edge(prev, node))
                            and (self.has_directed_edge(suc, node))
                    ):
                        all_colliders = False
                all_pred = True
                for node in path[1:-2]:
                    if not (self.has_directed_edge(node, v)):
                        all_pred = False
                nonadj = not self.has_edge(u, v)
                if b_pred and all_colliders and all_pred and nonadj:
                    discpaths.append(path)
        return discpaths

    def has_o(self, u, v, side):
        """
        A method to check the graph for a o tag on one side of an edge

        Parameters
        ----------
        u : str
            the from node of the edge
        v: str
            the to node of the edge
        side: str
            the side to test for the tag

        Returns
        -------
        bool
            True if has undirected edge, false if not
        """
        cond = False
        if self.has_edge(u, v):
            if self[u][v][side] == "o":
                cond = True
        return cond

    def isUncovered(self, path):
        """
        A method to see if the path is uncovered in the pag
        Parameters
        ----------
        path: str[]
            list of nodes

        Returns
        -------
        bool
            True if path is uncovered path, false if not

        """
        for x in range(1, len(path) - 1):
            pred = path[x - 1]
            suc = path[x + 1]
            if self.has_edge(pred, suc):
                return False
        return True

    def isPD(self, path):
        """
        A method to see if the path is potentially directed in the pag
        Parameters
        ----------
        path: str[]
            list of nodes

        Returns
        -------
        bool
            True if path is potentially directed path, false if not

        """
        for x in range(len(path) - 1):
            node = path[x]
            suc = path[x + 1]
            edge = self.get_edge_data(node, suc)
            if edge[suc] == "-" or edge[node] == ">":
                return False
        return True

    def isCirclePath(self, path):
        """
        A method to see if the every tag in the path is an o
        Parameters
        ----------
        path: str[]
            list of nodes

        Returns
        -------
        bool
            True if every tag in the path is an o, false if not
        """
        for i in range(len(path[:-1])):
            node = path[i]
            suc = path[i + 1]
            if not (self.has_o(node, suc, node) and self.has_o(suc, node, suc)):
                return False
        return True

    def findUncoveredCirclePaths(self, u, v):
        """
        A method find all circle paths
        between two nodes in the pag
        Parameters
        ----------
        u : str
            the from node of the path
        v: str
            the to node of the path

        Returns
        -------
        str[][]
            List of circle paths in pag between the two nodes
        """
        paths = []
        for path in nx.all_simple_paths(self, u, v):
            if self.isUncovered(path) and self.isCirclePath(path):
                paths.append(path)
        return paths

    def to_matrix(self):
        """
        A method to generate the adjacency matrix of the graph. Labels are
        sorted for better readability.

        Returns
        ----------
        numpy.ndarray
            a 2d array containing the adjacency matrix of the graph

        """
        symbol_map = {"o": 1, ">": 2, "-": 3}

        labels = sorted(list(self.nodes))  # [node for node in self]
        mat = np.zeros((len(labels), (len(labels))))
        for x in labels:
            for y in labels:
                if self.has_edge(x, y):
                    if bool(self.get_edge_data(x, y)):
                        mat[labels.index(x)][labels.index(y)] = symbol_map[
                            self.get_edge_data(x, y)[y]
                        ]
                    else:
                        mat[labels.index(x)][labels.index(y)] = 1
        return mat

    def write_to_file(self, path):
        """
        A method to write the adjacency matrix of the graph to a file

        Parameters
        ----------
        path: str
            path to file to write adjacency matrix to

        """
        mat = self.to_matrix()
        labels = sorted(list(self.nodes))
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join([f"{label}" for label in labels]))
            f.write("\n")
            for i, label in enumerate(labels):
                f.write(f"{label}")
                f.write(",")
                f.write(",".join([str(int(point)) for point in mat[i]]))
                f.write("\n")

    def to_dag(self, node_names: List[str] = None):
        dag = nx.DiGraph()
        if node_names is None:
            dag.add_nodes_from(self.nodes)
        else:
            dag.add_nodes_from(node_names)
        for u, v in self.edges:
            # Easy case - fully directed edge: u -> v
            if self.has_fully_directed_edge(u, v):
                dag.add_edge(u, v)
                continue
            # Bi-directed edge: u <-> v
            if self.has_bidirected_edge(u, v):
                dag.add_edge(u, v)
                dag.add_edge(v, u)
                continue
            # Partially directed edge: u o-> v
            if self.has_partially_directed_edge(u, v):
                dag.add_edge(u, v)
                continue
            # Partially bi-directed edge: u o-o v
            if self.has_partially_bidirected_edge(u, v):
                continue
            # Non-directed edge: u - v
            if self.has_fully_undirected_edge(u, v):
                continue

        return dag

    def to_dag_old(self):
        """
        Converts a Partially Directed Acyclic Graph (PDAG) to a Directed Acyclic Graph (DAG).
        Args:
        pdag (networkx.DiGraph): A PDAG represented as a NetworkX DiGraph.
        Returns:
        networkx.DiGraph: A DAG if conversion is possible, otherwise returns None.
        """
        # Create a copy of the PDAG to work on
        dag = self.copy()

        # Check if the original PDAG is acyclic
        if not nx.is_directed_acyclic_graph(dag):
            print("The provided graph is not acyclic.")
            return None

        # Identify undirected edges (i.e., edges that appear in both directions)
        undirected_edges = [(u, v)
                            for u, v in nx.edges(dag) if (v, u) in nx.edges(dag)]

        # Remove the undirected edges from the graph
        dag.remove_edges_from(undirected_edges)

        # Attempt to orient each undirected edge without creating a cycle
        for u, v in undirected_edges:
            # Temporarily add edge in one direction
            dag.add_edge(u, v)

            # Check if adding this edge creates a cycle
            if nx.is_directed_acyclic_graph(dag):
                continue  # Edge is okay, move to next one
            else:
                # Edge created a cycle, try the other direction
                dag.remove_edge(u, v)
                dag.add_edge(v, u)

                # Check if reversing the edge creates a cycle
                if not nx.is_directed_acyclic_graph(dag):
                    print(
                        "Unable to orient graph into a DAG without creating a cycle.")
                    return None

        return dag
