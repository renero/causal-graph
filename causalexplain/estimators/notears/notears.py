"""
NOTEARS

(C) Original code from https://github.com/xunzheng/notears
"""


import networkx as nx
import numpy as np
import pandas as pd
import scipy

from causalexplain.common import utils
from causalexplain.estimators.notears.loss import (least_squares_loss,
                                                 least_squares_loss_grad)
from causalexplain.metrics.compare_graphs import evaluate_graph


class NOTEARS:
    def __init__(
            self,
            name: str,
            loss=least_squares_loss,
            loss_grad=least_squares_loss_grad,
            c=0.25,
            r=10.0,
            e=1e-8,
            rnd_W_init=False,
            verbose=False):

        self.name = name
        self.loss = loss
        self.loss_grad = loss_grad
        self.c = c
        self.r = r
        self.e = e
        self.rnd_W_init = rnd_W_init
        self.verbose = verbose

    def notears_standard(self, data, return_all_progress=False):
        """
        Runs NOTEARS algorithm.

        Args:
            data (np.array): n x d data matrix with n samples, d variables
            c (float): minimum rate of progress, c \in (0,1)
            r (float): penalty growth rate, r > 1
            e (float): optimation accuracy, e > 0 (acyclicity stopping criteria)
            loss (function): loss function
            loss_grad (function): gradient of the loss function
            rnd_W_init (bool): initialize W to std. normal random matrix, rather than
                zero matrix
            output_all_progress (bool): return all intermediate values of W, rather than
            just the final value
        Returns:
            dict: { 'h': acyclicity of output,
                    'loss': loss of output,
                    'W': resulting optimized adjacency matrix}
        """
        n = np.shape(data)[0]
        d = np.shape(data)[1]

        data = np.array(data).astype(dtype=np.float64)
        cov = np.cov(data.T)

        if self.rnd_W_init:
            W = np.random.randn(d, d)
        else:
            W = np.zeros([d, d])  # initial guess
        W = W.astype(dtype=np.float64)
        a = 0.0    # initial guess
        p = 1.0    # initial penalty

        if return_all_progress:
            ret = []

        def h(W):
            # tr exp(W ◦ W) − d
            return np.trace(scipy.linalg.expm(np.multiply(W, W))) - d

        def h_grad(W):
            # ∇h(W) = [exp(W ◦ W)]^T ◦ 2W
            return np.multiply(
                np.transpose(scipy.linalg.expm(np.multiply(W, W))), 2.0 * W)

        def L(W, p, a):
            W = np.reshape(W, [d, d]).astype(dtype=np.float64)
            return self.loss(W, data, cov, d, n) + (p/2.0)*(h(W)**2) + a*(h(W))

        def L_grad(W, p, a):
            W = np.reshape(W, [d, d]).astype(dtype=np.float64)
            return np.reshape(
                self.loss_grad(W, data, cov, d, n) + h_grad(W)*(a + (p*h(W))),
                [d**2]).astype(dtype=np.float64)

        def get_W_star(p, W, a):
            W_flat = W.flatten()
            W_star = scipy.optimize.minimize(L, W_flat, args=(
                p, a), jac=L_grad, method='L-BFGS-B', options={'disp': False})
            W_star.x = W_star.x.reshape(W.shape).astype(dtype=np.float64)
            return W_star

        while True:
            W_star = get_W_star(p, W, a)
            W_star = W_star['x']
            # W_star = W_star.reshape(get_W_star(p, W, a)['x'],
            #   [d, d]).astype(dtype=np.float64)
            h_W_star = h(W_star)
            if h(W) != 0.0:
                while h_W_star >= max(self.c * h(W), self.e):
                    p = self.r*p
                    W_star = np.reshape(get_W_star(p, W, a)['x'], [
                                        d, d]).astype(dtype=np.float64)
                    h_W_star = h(W_star)
                    if self.verbose:
                        print("Increasing p:\t p = {: .2e}\n\t\t h_W_star = {}".format(
                            p, h_W_star))
            if return_all_progress:
                ret.append({'h': h_W_star, 'loss': self.loss(
                    W_star, data, cov, d, n), 'a': a, 'W': W_star})
            if h_W_star < self.e:
                if self.verbose:
                    print("Done:\t\t h = {}\n\t\t loss = {}\nt\t\t a = {}".format(
                        h_W_star, self.loss(W_star, data, cov, d, n), a))
                if return_all_progress:
                    return ret
                return {
                    'h': h_W_star,
                    'loss': self.loss(W_star, data, cov, d, n),
                    'W': W_star}
            if self.verbose:
                print("Progress:\t h = {}\n\t\t loss = {}\n\t\t a = {}". format(
                    h_W_star, self.loss(W_star, data, cov, d, n), a))
            a = a + p*h_W_star
            W = W_star

    def fit(self, X: pd.DataFrame, **kwargs):
        self.labels = list(X.columns)
        self.model = self.notears_standard(X)

        if self.verbose:
            print('Acyclicity loss...: {:.4g}'.format(self.model['h']))
            print('Least squares loss: {:.4g}'.format(self.model['loss']))

        return self

    def predict(self, ref_graph: nx.DiGraph = None, threshold: float = 0.1):
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
            threshold=0.1,
            **kwargs):
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
