"""
FCI algorithm.

This is a modified version of the FCI algorithm to work in parallel and improving
the performance of the algorithm, as well as the conditional independence tests.
"""

# pylint: disable=E1101:no-member
# pylint: disable=W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=W0106:expression-not-assigned
# pylint: disable=C0103:invalid-name, C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, R0902:too-many-instance-attributes
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=R1702:too-many-branches

import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import networkx as nx
from networkx import Graph
from sklearn.discriminant_analysis import StandardScaler
from mlforge.progbar import ProgBar                         # type: ignore

from causalgraph.common.utils import (graph_from_adjacency_file,
                                      graph_from_dot_file)
from causalgraph.estimators.fci.colliders import (get_dsep_combs,
                                                  get_neighbors, init_pag,
                                                  orientEdges)
from causalgraph.estimators.fci.debug import DebugFCI
from causalgraph.estimators.fci.graph_learner import GraphLearner
from causalgraph.estimators.fci.initialization import (dsep_set_from_csv,
                                                       save_graph, save_sepset)
from causalgraph.estimators.fci.pag import PAG
from causalgraph.independence.hsic import HSIC
from causalgraph.metrics.compare_graphs import evaluate_graph

#
# TODO: No control over max_samples to be used in HSIC `check_independence`
#


class FCI(GraphLearner):
    """
    A graph learner which implements the FCI algorithm
    """

    data = None
    pag = None
    dag = None
    is_fitted_ = False
    metrics = None
    feature_names = None

    def __init__(
            self,
            name: str,
            output_path: Union[Path, str] = "./",
            **kwargs):
        """
        Initialize the FCI algorithm creating an FCI learner.

        Parameters
        ----------
        data (pd.DataFrame): data to be used for learning the causal graph
        data_file (str): name of the file containing the data
        output_path (Path): path to the directory where the output files will be saved
        Keyword Arguments:
            logger (logging.Logger): logger object
            indep_test (Callable): independence test to be used for checking
                independence
            load_base_skeleton (bool): if True, load the base skeleton from the files
                specified
                in the YAML file for "base_skeleton" and "base_sepset"
            load_final_skeleton (bool): if True, load the final skeleton from the
                files specified in the YAML file for "final_skeleton" and "final_sepset"
            save_intermediate (bool): if True, save the intermediate results
            base_skeleton (str): name of the file containing the base skeleton
            base_sepset (str): name of the file containing the base sepset
            final_skeleton (str): name of the file containing the final skeleton
            final_sepset (str): name of the file containing the final sepset
            njobs (int): number of jobs to be used for parallel processing. If 1, run in
                sequential mode.
            verbose (bool): if True, print the progress of the algorithm

        Returns
        -------
        FCI object.
        """

        self.data_file = name
        self.name = name
        self.output_path = output_path
        self.indep_test = kwargs.get("indep_test", HSIC)
        self.load_final_skeleton = kwargs.get("load_final_skeleton", False)
        self.save_intermediate = kwargs.get("save_intermediate", False)
        self.final_skeleton = kwargs.get("final_skeleton", None)
        self.final_sepset = kwargs.get("final_sepset", None)
        self.njobs = kwargs.get("njobs", 1)
        self.verbose = kwargs.get("verbose", False)
        self.load_base_skeleton = kwargs.get("load_base_skeleton", False)
        self.base_skeleton = kwargs.get("base_skeleton", None)
        self.base_sepset = kwargs.get("base_sepset", None)
        self.log = kwargs.get("logger", None)
        self.prog_bar = kwargs.get("prog_bar", True)
        self.silent = kwargs.get("silent", False)

        super().__init__(logger=self.log,
                         data_file=self.data_file,
                         indep_test=self.indep_test,
                         parallel=self.njobs > 1,
                         verbose=self.verbose,
                         **kwargs)

        if self.verbose:
            self.debug = DebugFCI(self.verbose)

    def fit(self, X: pd.DataFrame):
        """
        function to learn a causal network from data

        Returns
        -------
        PAG
            causal network learned from data
        """
        assert isinstance(X, pd.DataFrame), "Data must be a pandas DataFrame"
        self.data = X
        self.feature_names = list(X.columns)

        super()._init_data(self.data)

        if self.verbose:
            print("Getting Skeleton of graph...")

        start_time = time.time()
        skeleton, sepset = self.learn_or_load_final_skeleton()
        if self.verbose:
            print("Orienting Edges...")

        self.pag = orientEdges(skeleton, sepset, data_file=self.data_file,
                               output_path=self.output_path, log=self.log,
                               verbose=self.verbose, debug=self.debug,
                               prog_bar=self.prog_bar, silent=self.silent)

        self.dag = self.pag.to_dag(self.feature_names)

        if self.verbose:
            print("Learning complete")
        self.debug.bm(f"Total time: {time.time() - start_time:.1f} secs.")

        return self

    def fit_predict(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            ref_graph: nx.DiGraph = None):
        """
        Method to fit the data and evaluate the learned graph against a reference graph.

        Parameters
        ----------
        X (pd.DataFrame): data to be used for learning the causal graph
        ref_graph (nx.DiGraph): reference graph to be used for evaluation

        Returns
        -------
        nx.DiGraph
            learned causal graph
        """
        pbar = ProgBar("FCI", 2, verbose=False) if self.prog_bar else None

        self.fit(train)

        # For compatibility with the other methods, the PAG is copied to a variable
        # called `dag`.
        if (len(self.dag.edges()) == 0) and (len(self.pag.edges()) != 0):
            # Copy pag nodes and edges to dag, one by one.
            for node in self.pag.nodes:
                self.dag.add_node(node, **self.pag.nodes[node])
            for edge in self.pag.edges:
                self.dag.add_edge(edge[0], edge[1], **self.pag.edges[edge])

        if ref_graph and self.dag:
            self.metrics = evaluate_graph(
                ref_graph, self.dag, self.feature_names)

        pbar.remove("FCI")

        return self.dag

    def learn_or_load_final_skeleton(self) -> Tuple[Graph, Dict]:
        """
        Learn a new Skeleton (2nd stage) or load an existing one if the files specified
        as parameters in the YAML file for "final_skeleton" and "final_sepset" exist.

        :return: graph, dict containing the skeleton and separation sets.
        """
        if self.load_final_skeleton:
            sk_file = Path(self.output_path, self.final_skeleton)
            ss_file = Path(self.output_path, self.final_sepset)
            if os.path.exists(sk_file) and os.path.exists(ss_file):
                skeleton = graph_from_adjacency_file(sk_file)
                sepset = dsep_set_from_csv(ss_file)
                if self.log:
                    self.log.info(f"Read {sk_file}")
                    self.log.info(f"Read {ss_file}")
            else:
                raise FileNotFoundError("Cannot find FINAL skeleton or sepset")
        else:
            now = time.time()
            if self.njobs > 1:
                skeleton, sepset = self.parallel_learn_skeleton()
            else:
                skeleton, sepset = self.learn_skeleton()
            self.debug.bm(
                f"2nd Stage Skeleton time: {time.time() - now:.1f} secs.")
            if self.save_intermediate:
                save_graph(skeleton, prefix="final_skeleton_FCI",
                           data_file=self.data_file,
                           output_path=self.output_path, log=self.log)
                save_sepset(sepset, prefix="final_sepset_FCI",
                            data_file=self.data_file, output_path=self.output_path,
                            log=self.log)
        return skeleton, sepset

    def learn_skeleton(self):
        """
        A  function to build the skeleton of a causal graph from data
        :returns: PDAG(The skeleton of the causal network) and a dict containg
        separation sets of all pairs of nodes
        """
        if self.log:
            self.log.info("Running in sequential mode")

        # Learn the base skeleton first.
        skeleton, sepSet = self.learn_or_load_base_skeleton()
        if self.verbose:
            for edge in skeleton.edges():
                print(edge)

        # Initialization
        pag_actions = defaultdict(list)
        pag, dseps = init_pag(skeleton, sepSet, self.verbose, self.debug)
        if self.verbose:
            print("Finding colliders...", flush=True)

        pbar = ProgBar().start_subtask(
            "Learn Skeleton", len(pag)) if self.prog_bar else None

        for lx, x in enumerate(pag):
            neighbors = get_neighbors(x, pag)
            self.debug.neighbors(x, lx, neighbors, pag)
            if len(neighbors) == 0:
                pbar.update_subtask(
                    "Learn Skeleton", 1) if self.prog_bar else None
                continue
            for ny, y in enumerate(neighbors):
                tup = self.find_colliders(x, y, pag, ny, dseps, sepSet,
                                          neighbors)
                if tup:
                    self.update_actions(pag_actions, tup)
            self.debug.stack(pag_actions)
            pbar.update_subtask("Learn Skeleton", 1) if self.prog_bar else None

        pbar.remove("Learn Skeleton") if self.prog_bar else None

        return pag, sepSet

    def parallel_learn_skeleton(self):
        """
        PARALLEL version of Learn Skeleton
        :return: tuple(PAG, sepSet)
        """
        if self.log:
            self.log.info(f"Running in parallel {self.njobs}")
        # Learn the base skeleton first.
        skeleton, sepSet = super().learn_or_load_base_skeleton()

        # Initialization
        now = time.time()
        pag_actions = defaultdict(list)
        pag, dseps = init_pag(skeleton, sepSet, self.verbose, self.debug)
        results = []
        pbar = None

        def update_PAG(tup):
            nonlocal results, pag, dseps, pag_actions, pbar
            if tup is None:
                results.append(False)
                pbar.update(1)
                return
            orig_node, dest_node, sep_set = tup[0], tup[1], tup[2]
            self.update_actions(pag_actions, (orig_node, dest_node, sep_set))
            if pag.has_edge(orig_node, dest_node):
                pag.remove_edge(orig_node, dest_node)
            dseps[(orig_node, dest_node)] = sep_set
            dseps[(dest_node, orig_node)] = sep_set
            results.append(True)
            pbar.update(1)

        if self.verbose:
            print("Finding colliders...", flush=True)
        pool = mp.Pool(self.njobs)

        pbar = ProgBar().start_subtask("Learn skeleton", len(pag))

        results = []
        for lx, x in enumerate(pag):
            pool.apply_async(
                self.explore_neighbours,
                args=(x, lx, pag, dseps, sepSet),
                callback=update_PAG)
            pbar.update_subtask("Learn skeleton", 1)

        # Wait until everyone is finished...
        pool.join()
        self.debug.stack(pag_actions)
        self.debug.bm(f"2nd stage skeleton time: {time.time() - now:.1f} secs")
        return pag, sepSet

    def explore_neighbours(self, x, id_x, pag, dseps, sepSet):
        # self.oo(f"–––––– {mp.current_process().name} ––––––")
        neighbors = get_neighbors(x, pag)
        self.debug.neighbors(x, id_x, neighbors, pag)
        if len(neighbors) == 0:
            return None
        for ny, y in enumerate(neighbors):
            tup = self.find_colliders(x, y, pag, ny, dseps, sepSet, neighbors)
            if tup:
                return tup
        return None

    def find_colliders(self, x, y, pag, ny, dseps, sepSet, neighbors):
        self.debug.y(x, y, ny, dseps, neighbors)
        i = 0
        while i < len(dseps[x]) + 1:
            tup = self.test_independence(x, y, i, dseps, sepSet, pag)
            if tup:
                self.debug.interrupt()
                return tup
            i += 1
        return None

    def test_independence(self, x: str, y: str, comb_len: int, dseps, sepSet,
                          pag: PAG) -> Optional[Tuple[str, str, list]]:
        dsep_combinations = get_dsep_combs(dseps, x, y, comb_len)
        if len(dsep_combinations) == 0:
            # self.debug.empty_set(comb_len, dseps, x, y)
            return None

        # Walk through all combinations formed for this d-sep.
        self.debug.d_seps(comb_len, dseps, dsep_combinations, x, y)
        for idx, dsep in enumerate(dsep_combinations):
            dsep = list(dsep)
            if len(dsep) < 1:
                continue
            if self.is_independent(x, y, dsep, idx, len(dsep_combinations)):
                self.update_graph(x, y, dsep, pag, sepSet)
                self.debug.interrupt()
                return x, y, dsep
        return None


def main(dataset_name,
         input_path="/Users/renero/phd/data/sachs/",
         output_path="/Users/renero/phd/output/RC4/sachs/compared/",
         save=False,
         **kwargs):
    """
    Create a call to FCI with a sample dataset.
    """
    ref_graph = graph_from_dot_file(f"{input_path}{dataset_name}.dot")
    data = pd.read_csv(f"{input_path}{dataset_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    train = data.sample(frac=0.8, random_state=42)
    test = data.drop(train.index)

    fci = FCI(name=dataset_name, output_path=output_path, **kwargs)
    fci.fit_predict(train=data, test=None, ref_graph=ref_graph)

    if fci.dag:
        for edge in fci.dag.edges():
            print(edge)
    else:
        for edge in fci.pag.edges():
            print(edge)
    if fci.metrics:
        print(fci.metrics)
    else:
        print("No metrics available")

    # if save:
    #     where_to = utils.save_experiment(rex.name, output_path, rex)
    #     print(f"Saved '{rex.name}' to '{where_to}'")


# Create a call to FCI with a sample dataset.
if __name__ == "__main__":
    main("sachs", njobs=1)
    # main("rex_generated_linear_1", njobs=1)
