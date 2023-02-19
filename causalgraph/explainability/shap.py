"""
This is a module to be used as a reference for building other modules
"""
from typing import Dict

import numpy as np
import shap
import torch
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from tqdm.auto import tqdm
import networkx as nx

from causalgraph.common.feature_selection import select_features
from causalgraph.independence.edge_orientation import get_edge_orientation


class ShapEstimator(BaseEstimator):
    """
    """

    def __init__(
            self,
            models: Dict = None,
            method: str = 'knee',
            sensitivity: float = 1.0,
            tolerance: float = None,
            descending: bool = False,
            iters: int = 20,
            reciprocity: False = False,
            min_impact: float = 1e-06,
            progbar: bool = True,
            verbose: bool = False,
            prog_bar: bool = True,
            on_gpu: bool = False):
        """
        """
        self.models = models
        self.method = method
        self.sensitivity = sensitivity
        self.tolerance = tolerance
        self.descending = descending
        self.iters = iters
        self.reciprocity = reciprocity
        self.min_impact = min_impact
        self.progbar = progbar
        self.verbose = verbose
        self.prog_bar = prog_bar
        self.on_gpu = on_gpu

    def fit(self, X, y=None):
        """
        """
        #X, y = check_X_y(X, y, accept_sparse=True)

        columns = list(self.models.keys())
        self.shap_values = dict()
        pbar = tqdm(total=len(columns),
                    desc="Computing SHAPLEY values", disable=not self.prog_bar, leave=False)
        for target_name in columns:
            pbar.update()
            model = self.models[target_name]
            features_tensor = model.train_loader.dataset.features
            if self.on_gpu:
                tensorData = torch.autograd.Variable(features_tensor).cuda()
                explainer = shap.DeepExplainer(model.model.cuda(), tensorData)
            else:
                tensorData = torch.autograd.Variable(features_tensor)
                explainer = shap.DeepExplainer(model.model, tensorData)
            self.shap_values[target_name] = explainer.shap_values(tensorData)
            pbar.refresh()
        pbar.close()

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Builds a causal graph from the shap values using a selection mechanism based
        on the knee or abrupt methods.
        """
        self.feature_names_ = list(X.columns)
        X_array = check_array(X)
        check_is_fitted(self, 'is_fitted_')

        self.parents = dict()
        for target in self.feature_names_:
            candidate_causes = [f for f in self.feature_names_ if f != target]
            self.parents[target] = select_features(
                values=self.shap_values[target],
                feature_names=candidate_causes,
                method=self.method,
                tolerance=self.tolerance,
                sensitivity=self.sensitivity,
                descending=self.descending,
                min_impact=self.min_impact)

        # Add edges ONLY between nodes where SHAP recognizes both directions
        G_shap_unoriented = nx.Graph()
        for target in self.feature_names_:
            for parent in self.parents[target]:
                if self.reciprocity:
                    if target in self.parents[parent]:
                        G_shap_unoriented.add_edge(target, parent)
                else:
                    G_shap_unoriented.add_edge(target, parent)

        G_shap = nx.DiGraph()
        pbar = tqdm(total=len(G_shap_unoriented.edges()), disable=not self.progbar,
                    leave=False, desc="Orienting causal graph")
        for u, v in G_shap_unoriented.edges():
            pbar.update(1)
            orientation = get_edge_orientation(
                X, u, v, iters=self.iters, method="gpr", verbose=self.verbose)
            if orientation == +1:
                G_shap.add_edge(u, v)
            elif orientation == -1:
                G_shap.add_edge(v, u)
            else:
                pass
                # G_shap.add_edge(u, v)
                # G_shap.add_edge(v, u)
            pbar.refresh()
        pbar.close()

        return G_shap
