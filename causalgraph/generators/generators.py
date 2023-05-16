"""Acyclic Graph Generator.

Generates a dataset out of an acyclic FCM.
Author : Olivier Goudet and Diviyan Kalainathan

Minor update: J. Renero (due to pandas changes to as_matrix())
"""

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import scale

from causalgraph.generators.mechanisms import *
from causalgraph.common.plots import dag2dot


class AcyclicGraphGenerator(object):
    """Generates a cross-sectional dataset out of a cyclic FCM."""

    def __init__(self, causal_mechanism,
                 initial_variable_generator=gmm_cause,
                 points=500, nodes=20, timesteps=0, parents_max=5):
        """
        :params:
        @param:initial_variable_generator(points): init variables of the graph
        @param:causal_mechanism(causes): generating causes in the graph to
            choose between: ['linear', 'polynomial', 'sigmoid_add',
            'sigmoid_mix', 'gp_add', 'gp_mix']
        """
        super(AcyclicGraphGenerator, self).__init__()
        self.mechanism = {'linear': LinearMechanism,
                          'polynomial': Polynomial_Mechanism,
                          'sigmoid_add': SigmoidAM_Mechanism,
                          'sigmoid_mix': SigmoidMix_Mechanism,
                          'gp_add': GaussianProcessAdd_Mechanism,
                          'gp_mix': GaussianProcessMix_Mechanism}[causal_mechanism]
        self.data = pd.DataFrame(
            None, columns=["V{}".format(i) for i in range(nodes)])
        self.nodes = nodes
        if timesteps == 0:
            self.timesteps = np.inf
        else:
            self.timesteps = timesteps
        self.points = points
        self.adjacency_matrix = np.zeros((nodes, nodes))
        self.parents_max = parents_max
        self.initial_generator = initial_variable_generator
        self.cfunctions = None
        self.g = None

    def init_variables(self, verbose=False):
        """Redefine the causes of the graph."""
        # Resetting adjacency matrix
        for i in range(self.nodes-1):
            for j in np.random.choice(range(i+1, self.nodes),
                                      np.random.randint(0, min([self.parents_max,
                                                                self.nodes-i])),
                                      replace=False):
                if i != j:
                    self.adjacency_matrix[i, j] = 1

        try:
            assert any([sum(self.adjacency_matrix[:, i]) ==
                        self.parents_max for i in range(self.nodes)])
            self.g = nx.DiGraph(self.adjacency_matrix)
            assert not list(nx.simple_cycles(self.g))

        except AssertionError:
            if verbose:
                print("Regenerating, graph non valid...")
            self.init_variables()

        # Mechanisms
        self.cfunctions = [self.mechanism(int(sum(self.adjacency_matrix[:, i])),
                                          self.points)
                           if sum(self.adjacency_matrix[:, i])
                           else self.initial_generator for i in range(self.nodes)]

    # def generate(self, nb_steps=100, averaging=50, rescale=True):
    def generate(self, rescale=True):
        """Generate data from an FCM containing cycles."""
        if self.cfunctions is None:
            self.init_variables()

        for i in nx.topological_sort(self.g):
            # Root cause
            if not sum(self.adjacency_matrix[:, i]):
                self.data[f'V{i}'] = self.cfunctions[i](self.points)
            # Generating causes
            else:
                self.data[f'V{i}'] = self.cfunctions[i](
                    np.stack(self.data.iloc[:, self.adjacency_matrix[:, i].nonzero()[0]].values))
            if rescale:
                self.data[f'V{i}'] = scale(np.stack(self.data[f'V{i}'].values))

        return self.g, self.data

    def to_csv(self, fname_radical, **kwargs):
        """
        Save data to the csv format by default, in two separate files.

        Optional keyword arguments can be passed to pandas.
        """
        if self.data is not None:
            self.data.to_csv(fname_radical+'_data.csv', **kwargs)
            pd.DataFrame(self.adjacency_matrix).to_csv(
                fname_radical+'_target.csv', **kwargs)
            # Save also the DOT format
            graph_dot_format = dag2dot(self.g).to_string()
            graph_dot_format = f"strict {graph_dot_format[:-9]}\n}}"
            with open(fname_radical+'_target.dot', "w") as f:
                f.write(graph_dot_format)

        else:
            raise ValueError("Graph has not yet been generated. \
                              Use self.generate() to do so.")


if __name__ == '__main__':
    g = AcyclicGraphGenerator("polynomial", points=500,
                              nodes=10, timesteps=0, parents_max=3)
    graph, data = g.generate()
    data.to_csv(
        "/Users/renero/phd/data/generated_polynomial_10.csv", index=False)
    graph_dot_format = dag2dot(graph, plot=False).to_string()
    graph_dot_format = f"strict {graph_dot_format[:-9]}\n}}"
    # write to file
    with open("/Users/renero/phd/data/generated_polynomial_10.dot", "w") as f:
        f.write(graph_dot_format)
