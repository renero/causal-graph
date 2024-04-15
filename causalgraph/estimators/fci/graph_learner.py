# pylint: disable=E1101:no-member
# pylint: disable=W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=W0106:expression-not-assigned
# pylint: disable=C0103:invalid-name, C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, R0902:too-many-instance-attributes
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=R1702:too-many-branches

import itertools
import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Union, List

import networkx as nx
import numpy as np
import pandas as pd
from hyppo.independence import Hsic
from networkx import DiGraph, Graph
from pandas import DataFrame
from pygam import pygam

from mlforge import ProgBar

from causalgraph.common.utils import graph_from_adjacency_file
from causalgraph.estimators.fci.initialization import (dsep_set_from_csv,
                                                       save_graph, save_sepset)
from causalgraph.estimators.fci.debug import *


class GraphLearner():
    """
    Base Class for all graph learning algorithms
    implementing functionality used across algorithms
    """

    data: pd.DataFrame = None
    labels: List[str] = None
    samples: np.ndarray = None

    def __init__(
            self,
            logger,
            data_file,
            indep_test,
            alpha=0.05,
            njobs=1,
            parallel=False,
            load_base_skeleton: bool = False,
            output_path=None,
            base_skeleton: str = None,
            base_sepset: str = None,
            save_intermediate: bool = False,
            verbose: bool = False
    ):
        """
        Initialise graph learner object

        Parameters
        ----------
            # :param data: pandas.DataFrame,
            #     DataFrame with rows as datapoints and columns as variables
            :param log: A logger
            :param alpha: float, optional
                The minimum p-value returned by the indepence test
                for the data to be considered independent
            :param parallel: boolean indicating whether I'm running in
                parallel mode. Default is False
        """
        self.alpha = alpha
        self.indep_test = indep_test
        self.parallel = parallel
        self.log = logger
        self.verbose = verbose
        self.njobs = njobs
        self.output_path = output_path
        self.data_file = data_file
        self.load_base_skeleton = load_base_skeleton
        self.base_skeleton = base_skeleton
        self.base_sepset = base_sepset
        self.save_intermediate = save_intermediate

        # Debugging stuff
        self.debug = DebugFCI(self.verbose)
        self.oo = self.debug.dbg
        self.verbose = self.verbose

    def _init_data(self, data):
        """
        Initialise data, labels and samples attributes. This method is
        called from the `fit()` method of the subclasses.
        """
        self.data = data
        self.labels = list(self.data.columns)
        self.samples = self.data.to_numpy(dtype='float')

    def learn_or_load_base_skeleton(self):
        """
        Learn a new Skeleton or load an existing one if the files specified
        as parameters in the YAML file for "skeleton" and "sepset" exist.

        :return: graph, dict containing the skeleton and separation sets.
        """
        if self.load_base_skeleton:
            sk_file = Path(self.output_path, self.base_skeleton)
            ss_file = Path(self.output_path, self.base_sepset)
            if os.path.exists(sk_file) and os.path.exists(ss_file):
                skeleton = graph_from_adjacency_file(sk_file)
                sepset = dsep_set_from_csv(ss_file)
                if self.log:
                    self.log.info(f"Read {sk_file}")
                    self.log.info(f"Read {ss_file}")
            else:
                raise FileNotFoundError("Cannot find BASE skeleton or sepset")
        else:
            now = time.time()
            if self.njobs > 1:
                skeleton, sepset = self.parallel_learn_base_skeleton()
            else:
                skeleton, sepset = self.learn_base_skeleton()
            self.debug.bm(
                f"1st Stage Skeleton time: {time.time() - now:.1f} secs.")
            if self.save_intermediate:
                save_graph(skeleton, "base_skeleton_FCI",
                           self.data_file, self.output_path, self.log)
                save_sepset(sepset, "base_sepset_FCI",
                            self.data_file, self.output_path, self.log)
        return skeleton, sepset

    def learn_base_skeleton(self) -> Tuple[Union[Graph, DiGraph], Dict]:
        """
        A function to build the skeleton of a causal graph from data

        Returns
        -------
            networkx.Graph
                The skeleton of the causal network
            dict
                Dicitonary containg separation sets of all pairs of nodes
        """
        actions, condlen, condsize, graph, sepset = self.init_learning()

        # Iterate over each pair of adjacent nodes
        # pbar = tqdm(
        #     self.labels,
        #     **tqdm_params("Base Skeleton", self.prog_bar, silent=self.silent))
        pbar = ProgBar().start_subtask(len(self.labels))

        while condlen != 0:
            # condlen controls the amount of potential dependencies to explore
            # at each iteration the nr of cond. sets are added to this variable
            # if it happens to be zero, then exploration stops.
            # pbar.reset()
            condlen = 0
            self.oo(f"> Iterating over labels: {','.join(self.labels)}")
            for lx, x in enumerate(self.labels):
                pbar.refresh()
                self.oo(f" + x = {x}; {lx + 1}/{len(self.all_labels_but(x))}")
                condlen, actions = self.check_independence(
                    x, graph, condlen, condsize, sepset, actions)
                pbar.update_subtask(1)
            condsize += 1

        self.debug.stack(actions)
        return graph, sepset

    def init_learning(self):
        if self.verbose:
            print("1st Stage Skeleton")
        actions = defaultdict(list)
        condsize = 0
        condlen = 1
        graph, sepset = self.init_graph()
        return actions, condlen, condsize, graph, sepset

    @staticmethod
    def update_actions(actions, tup):
        if tup is not None:
            if tup[2]:
                actions[tup[0]].append((tup[1], tup[2]))
            else:
                actions[tup[0]].append((tup[1], ()))

    def parallel_learn_base_skeleton(self) -> Tuple[Union[Graph, DiGraph], Dict]:
        """
        A function to parallel build the skeleton of a causal graph from data

        Returns
        -------
            networkx.Graph
                The skeleton of the causal network
            dict
                Dicitonary containg separation sets of all pairs of nodes
        """
        actions, condlen, condsize, graph, sepset = self.init_learning()
        pbar = None
        pool = mp.Pool(mp.cpu_count())

        def collect_result(tup):
            nonlocal condlen, graph, sepset, actions, pbar
            condlen += tup[0]
            for orig_node in tup[1]:
                for dest_node, cs in tup[1][orig_node]:
                    self.update_actions(actions, (orig_node, dest_node, cs))
                    if graph.has_edge(orig_node, dest_node):
                        graph.remove_edge(orig_node, dest_node)
                    sepset[(orig_node, dest_node)] = cs
                    sepset[(dest_node, orig_node)] = cs
            if not self.verbose:
                pbar.update(1)

        # Iterate over each pair of adjacent nodes
        # pbar = tqdm(
        #     len(self.labels),
        #     **tqdm_params("Base Skeleton", self.prog_bar, silent=self.silent))
        pbar = ProgBar().start_subtask(len(self.labels))

        pbar.set_description("Base skeleton")
        while condlen != 0:
            # condlen controls the amount of potential dependencies to explore
            # at each iteration the nr of cond. sets are added to this variable
            # if it happens to be zero, then exploration stops.
            # pbar.reset()
            condlen = 0
            results = []
            self.oo(f"> Iterating over labels: {','.join(self.labels)}")
            for x in self.labels:
                result = pool.apply_async(
                    self.check_independence,
                    args=(x, graph, condlen, condsize, sepset),
                    callback=collect_result)
                results.append(result)
                pbar.update_subtask()
                # pbar.refresh()
            [result.wait() for result in results]
            condsize += 1

        pool.close()
        self.debug.stack(actions)

        return graph, sepset

    def check_independence(
            self,
            x,
            graph,
            condlen,
            condsize,
            sepset,
            actions=None):
        """
        This method checks for independence between X and all its neighbors
        in the graph. If X and Y are independent, then the edge between X and Y
        is removed from the graph.

        Parameters
        ----------
            x: str
                The node being tested for independence
            graph: nx.Graph
                The graph being built
            condlen: int
                The number of conditioning sets being tested
            condsize: int
                The size of the conditioning sets being tested
            sepset: dict
                A dictionary containing the separation sets of all pairs of
                nodes
            actions: dict, optional
                A dictionary containing the actions taken by the algorithm
                Default is None
        Returns
        -------
            int
                The number of conditioning sets being tested
        """
        if actions is None:
            actions = defaultdict(list)

        neighbors = list(graph.neighbors(x))
        neighbors_str = ",".join(list(neighbors))
        self.oo(f"{A0}> Iterating over neighbors: {neighbors_str}")
        for ny, y in enumerate(neighbors):
            self.oo(f"{A0}+ y = {y}; {ny + 1}/{len(neighbors)}")
            # Generate the conditioning sets needed for
            # independence tests
            condSets = self.gen_cond_sets(x, y, graph, condsize)
            if len(condSets) == 0:
                continue
            condlen += len(condSets)
            # Test for independence with each conditioning set
            cs = ",".join([f"({','.join(cs)})" for cs in condSets])
            self.oo(f"{A2}> Check {x}⫫{y} over {len(condSets)} CS: [{cs}]")
            tup = self.test_cond_independence(x, y, condSets, graph, sepset)
            if tup is not None:
                self.update_actions(actions, tup)

        return condlen, actions

    def test_cond_independence(self, x, y, cond_sets, graph, sep_sets):
        """
        Test if X and Y are independent, using all possible conditioning sets
        in the graph from nodes adjacent to X. If it is found that X and Y are
        independent, exploration finish, and the edge between X and Y is
        removed from the graph.
        :param x: str, name of node x
        :param y: str, name of node y
        :param cond_sets: dict with tuples of nodes adjacent to X
        :param sep_sets: list of pairs of nodes, with all possible d-seps
        :param graph: Graph or DiGraph.
        :return: bool, indicating whether X and Y are independent.
        """
        L = len(cond_sets)
        for idx, cond_set in enumerate(cond_sets):
            if self.is_independent(x, y, cond_set, idx, L):
                self.update_graph(x, y, cond_set, graph, sep_sets)
                return x, y, cond_set
        return None

    def init_graph(self) -> Tuple[Union[Graph, DiGraph], Dict]:
        # Generate completed graph
        graph = nx.complete_graph(len(self.labels))
        # Rename nodes in graph with labels
        name_map = {i: label for i, label in enumerate(self.labels)}
        graph = nx.relabel_nodes(graph, name_map)
        # Generate an empty list of separation sets with all possible pair
        # combinations of initially set of nodes.
        sepset = {}
        for p in itertools.permutations(self.labels, 2):
            sepset[p] = ()
        return graph, sepset

    def update_graph(self, x, y, cs, graph, sepset):
        self.oo(f"{A3}! {x}⫫{y}|({','.join(cs)}) ❌ Independent!")
        graph.remove_edge(x, y)
        sepset[(x, y)] = cs
        sepset[(y, x)] = cs

    def all_labels_but(self, x):
        return sorted(list(set(self.labels) - {x}))

    def run_test(self, x_label, y_label, cond_set):
        """
        Runs the conditional independence test between X and Y, conditionally
        based on cond_set
        :param x: str, name of node X
        :param y: str, name of node Y
        :param cond_set: list of nodes forming the conditioning set for the set
        :return: (bool, t_stat, threshold) indicating whether X is independent
            of Y, conditioned to cond_set, the value of the t_stat, and the
            threshold used/computed to determine independence.
        """
        x_idx = self.labels.index(x_label)
        y_idx = self.labels.index(y_label)
        condset_idx = [self.labels.index(e) for e in cond_set]

        x, y = self.samples[:, x_idx], self.samples[:, y_idx]
        Z = self.samples[:, condset_idx] if len(condset_idx) != 0 else None

        indep, t_stat, th = self.cond_indep_test(x, y, Z)
        return indep, t_stat, th

    def is_independent(self, x, y, cond_set, idx, L):
        """
        Run the independence test between X and Y, using the conditioning set
        passed as argument.
        :param x: str, name of node X
        :param y: str, name of node Y
        :param cond_set: Tuple containing a list of nodes
        :param idx: int, with the index of this cond-set among all possible
        :param L: int, total nr. of conditioning sets.
        :return: bool, indicating whether X and Y are independent.
        """
        indep, t_stat, th = self.run_test(x, y, cond_set)
        # stop testing if X indep Y, and remove edge
        if not indep:
            msg = "{}- [{:02d}/{:02d}] {}⫫{}|({}) - Not independent"
            self.oo(msg.format(A3, idx + 1, L, x, y, ','.join(cond_set)))
        return indep

    @staticmethod
    def gen_cond_sets(x, y, g, size):
        """ A function to build the set of conditioning sets to be for
        variables x and y on graph g of a certain size when generating a
        skeleton

        Arguments:
            - X: str,
                One variable being tested for independence,
            - Y: str
                The other variable being tested for independence
            - graph: float, optional
                The minimum p-value returned by the indepence test
                for the data to be considered independent
            - size: int
                the size of each conditioning set to be returned
        Returns:
            list of lists of strings
                a list of conditioning sets to be tested
        """
        # Handle size 0 case
        if size == 0:
            return [()]
        # Get all neighbors of x in g
        adjy = g.neighbors(x)
        # Remove y from neighbour list
        adj = [node for node in adjy if node != y]
        # Generate all unique combinations of len size
        combos = list(itertools.combinations(adj, size))
        return combos

    def cond_indep_test(self, x: np.array, y: np.array, Z: np.array = None):
        """
        Perform the conditional independence test configured in the
        main class attribute

        Arguments:
        ----------
            - x (np.array): The first variable being tested for independence
            - y (np.array): The second variable being tested for independence
            - Z (np.array): optional (default=None) The conditioning set, if any

        Returns:
        --------
            - bool: True if x is independent of y, False otherwise
            - float: the t-statistic of the HSIC independence test
            - float: the p-value of the independence test

        """
        if Z is None:
            stat, pval = Hsic().test(x, y)
        else:
            r_x, r_y = self.residuals(x, y, Z)
            stat, pval = Hsic().test(r_x, r_y)

        indep = pval > 0.05
        return indep, stat, pval

    def residuals(self, x, y, Z):
        """
        Train a GAM with cond_set to predict "i", and then "j". For both cases
        get the residuals.

        Arguments:
            - x (np.array): x-values of the data
            - y (np.array): y-values of the data
            - Z (np.array): Z-values of the data
        Returns:
            - r_x (np.array): residuals of GAM fitted to predict x from Z
            - r_y (np.array): residuals of GAM fitted to predict y from Z
        """
        g = pygam.GAM()
        g.fit(Z, x)
        residuals_x = g.deviance_residuals(Z, x)
        g.fit(Z, y)
        residuals_y = g.deviance_residuals(Z, y)

        return residuals_x, residuals_y

    @staticmethod
    def prepare_data(data_file, delim=' ', isLabeled=False, ):
        """
        A function which reads data from a file into a pandas dataframe
        the file should consist of rows of datapoints with each variable
        separated by some delimination string

        Parameters
        ----------
            data_file : str,
                The path to the file containing data
            delim : str, optional
                the deliminating string, ',' for csv files
            isLabeled : bool, optional
                True if the first line in the file is the lit of variabe names
        Returns
        -------
            pandas.DataFrame
                data frame containing data from file
        """

        with open(data_file, 'r') as f:
            if isLabeled:
                labels = f.readline().replace('\n', '').split(delim)
                line1 = f.readline().replace('\n', '').split(delim)
            else:
                line1 = f.readline().replace('\n', '').split(delim)
                labels = [str(i) for i in range(len(line1))]
            data = [line1]
            for line in f.readlines():
                line = line.replace('\n', '').split(delim)
                data.append(line)
            data = DataFrame(data, columns=labels)
        return data

    @staticmethod
    def findPath(x, y, directed, explored):
        """
        A method to check if there is a path between two nodes in  graph
        ----------
        x: str
            from node of a path
        y: str
            to node of a path
        directed: nx.DiGraph
            Directed graph
        explored: str[]
            list of nodes explored by previous recursive calls left as [] in
            calling

        Returns
        -------
        bool
            True if there is a path between x and y in directed
        """

        explored.append(x)
        neigh = []
        for n in directed.successors(x):
            neigh.append(n)
        Z = []
        for n in neigh:
            if n not in explored:
                Z.append(n)
        if y in Z:
            return True
        if len(Z) == 0:
            return False
        isPath = False
        for z in Z:
            zcont = GraphLearner.findPath(z, y, directed, explored)
            isPath = isPath or zcont
        return isPath
