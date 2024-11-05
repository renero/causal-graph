import os
import numpy as np
import pandas as pd
import networkx as nx

from causalgraph.estimators.notears.loss import least_squares_loss, least_squares_loss_grad
from causalgraph.estimators.notears.main import notears_standard, run

from causalgraph.common import utils
from causalgraph.metrics.compare_graphs import evaluate_graph


class NOTEARS:
    def __init__(
            self,
            name: str,
            variant=notears_standard,
            loss=least_squares_loss,
            loss_grad=least_squares_loss_grad,
            e=1e-8,
            verbose=False):

        self.name = name
        self.variant = variant
        self.loss = loss
        self.loss_grad = loss_grad
        self.e = e
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None):
        self.labels = list(X.columns)
        self.model = run(
            self.variant,
            X,
            self.loss,
            self.loss_grad,
            e=self.e,
            verbose=self.verbose)

        if self.verbose:
            print('Acyclicity loss...: {:.4g}'.format(self.model['h']))
            print('Least squares loss: {:.4g}'.format(self.model['loss']))

        return self

    def predict(self, ref_graph: nx.DiGraph=None, threshold:float=0.1):
        if ref_graph:
            true_adj_mat = utils.graph_to_adjacency(
                ref_graph, labels=self.labels)
            num_nodes = true_adj_mat.shape[0]
            num_edges = len(ref_graph.edges())

        acyclic_W = np.where(self.model['W'] > threshold, 1, 0).T

        self.dag = utils.graph_from_adjacency(
            adjacency=acyclic_W, node_labels=self.labels)

        self.metrics = evaluate_graph(ref_graph, self.dag)

        return self

    def fit_predict(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            ref_graph: nx.DiGraph = None,
            threshold=0.1):
        self.fit(train)
        self.predict(ref_graph, threshold)

        return self.dag


def main(dataset_name,
         input_path="/Users/renero/phd/data/",
         output_path="/Users/renero/phd/output/",
         save=False):

    data = pd.read_csv(f"{input_path}{dataset_name}.csv")
    ref_graph = utils.graph_from_dot_file(f"{input_path}{dataset_name}.dot")

    notears = NOTEARS(name="main_run")
    notears.fit_predict(data, test=None, ref_graph=ref_graph)
    print(notears.metrics)


if __name__ == "__main__":
    main("toy_dataset")
