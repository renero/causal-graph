"""
This is a module to be used as a reference for building other modules
"""
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import torch

from scipy.stats import spearmanr
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

from causalgraph.common import *
from causalgraph.independence.feature_selection import select_features
from causalgraph.models.dnn import NNRegressor
from causalgraph.independence.edge_orientation import get_edge_orientation
from causalgraph.common.plots import plot_shap_summary


class ShapEstimator(BaseEstimator):
    """
    """

    def __init__(
            self,
            models: NNRegressor = None,
            method: str = 'knee',
            sensitivity: float = 1.0,
            tolerance: float = None,
            descending: bool = False,
            iters: int = 20,
            reciprocity: False = False,
            min_impact: float = 1e-06,
            verbose: bool = False,
            enable_progress_bar: bool = True,
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
        self.verbose = verbose
        self.prog_bar = enable_progress_bar
        self.on_gpu = on_gpu

        self._fit_desc = "Computing SHAPLEY values"
        self._pred_desc = "Building graph skeleton"

    def fit(self, X, y=None):
        """
        """
        # X, y = check_X_y(X, y, accept_sparse=True)

        self.all_feature_names_ = list(self.models.regressor.keys())
        self.shap_values = dict()

        pbar = tqdm(total=len(self.all_feature_names_),
                    desc=f"{self._fit_desc:<25}", disable=not self.prog_bar,
                    leave=False)

        for target_name in self.all_feature_names_:
            pbar.update(1)
            model = self.models.regressor[target_name]
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
        # X_array = check_array(X)
        check_is_fitted(self, 'is_fitted_')

        pbar = tqdm(total=3,
                    desc=f"{self._fit_desc:<25}", disable=not self.prog_bar,
                    leave=False)

        self.parents = dict()
        for target in self.all_feature_names_:
            candidate_causes = [
                f for f in self.all_feature_names_ if f != target]
            self.parents[target] = select_features(
                values=self.shap_values[target],
                feature_names=candidate_causes,
                method=self.method,
                tolerance=self.tolerance,
                sensitivity=self.sensitivity,
                descending=self.descending,
                min_impact=self.min_impact)
        pbar.update(1)
        pbar.refresh()
        # Add edges ONLY between nodes where SHAP recognizes both directions
        G_shap_unoriented = nx.Graph()
        for target in self.all_feature_names_:
            for parent in self.parents[target]:
                if self.reciprocity:
                    if target in self.parents[parent]:
                        G_shap_unoriented.add_edge(target, parent)
                else:
                    G_shap_unoriented.add_edge(target, parent)
        pbar.update(1)
        pbar.refresh()
        pbar.close()

        G_shap = nx.DiGraph()
        desc = "Orienting causal graph"
        pbar = tqdm(total=len(G_shap_unoriented.edges()),
                    disable=not self.prog_bar,
                    leave=False, desc=f"{desc:<25}")
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

    def compute_shap_discrepancies(self):
        """
        Compute the discrepancies between the SHAP values and the target values
        for all features and all targets.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the discrepancies for all features and all targets.
        """
        self.discrepancies = pd.DataFrame(columns=self.all_feature_names_)
        for target_name in tqdm(self.all_feature_names_, leave=False):
            X, y = self.models.get_input_tensors(target_name)
            feature_names = list(X.columns)

            # Add a new row to dataframe with the value of the target_name as index
            self.discrepancies.loc[target_name] = 0

            # Loop through all features and compute the discrepancy
            for feature in feature_names:
                discrepancy = self.compute_shap_alignment(
                    X[feature].values,
                    self.shap_values[target_name],
                    y,
                    feature,
                    feature_names)
                self.discrepancies.loc[target_name,
                                       feature] = discrepancy['discrepancy']

        return self.discrepancies

    def compute_shap_alignment(
            self,
            x,
            shap_values,
            y,
            feature,
            feature_names,
            ax=None,
            plot: bool = False):
        """ 
        Compute the alignment of the shap values for a given feature with the target 
        variable. This is done by computing the correlation between the shap values and 
        the target variable.

        The discrepancy value is computed as: $1 - \\textrm{corr}(y, shap_values)$

        I'm experimenting with the SHAP dependency plots. Given a target variable, 
        I plot the values of each feature against its SHAP values, and against the 
        target variable. In theory, both plots should regress to the same point, and 
        present the same slope. How similar are these slopes is what I called the 
        Discrepancy Ratio. For some predictions, it is extremely low, showing that 
        SHAP tendency actually reflects the actual influence of that feature in the 
        prediction of the target variable. For other features, though, it is high, 
        showing that the SHAP values obtained for that specific feature do not 
        correspond to the actual relationship with the target.

        In this way, you can see which features are actually playing a role in the 
        prediction, and which are not. This can help you better understand the causal 
        relationships between features and the target variable.

        Parameters
        ----------
        x : pd.DataFrame
            The input data. This is a numpy array with the values of the feature 
            used to predict the target variable in the main model from which 
            the SHAP values were computed.
        shap_values : pd.DataFrame 
            The SHAP values for the feature used to predict the target variable in 
            the main model from which the SHAP values were computed.
        y : pd.DataFrame
            The target variable. This is a numpy array with the values of the target
            variable in the main model from which the SHAP values were computed.
        feature : str
            The name of the feature used to predict the target variable in the 
            main model from which the SHAP values were computed.
        feature_names : list
            The names of the features used to predict the target variable in the 
            main model from which the SHAP values were computed.
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            The axis to plot the alignment on, by default None
        plot : bool, optional
            Whether to plot the alignment, by default False

        Returns
        -------
        dict{float, float, float}
            A dictionary with the discrepancy, the slope of the SHAP values 
            regression and the slope of the regression of the target variable 
            on the feature.

        """
        # Compute a regression to the shap values
        feature_pos = feature_names.index(feature)
        b_shap, m_shap = self._regress(x, shap_values[:, feature_pos])
        b_target, m_target = self._regress(x, y)
        corr = spearmanr(shap_values[:, feature_pos], y)

        if plot:
            # Plot the SHAP values and the regression
            ax.scatter(x, shap_values[:, feature_pos],
                       alpha=0.3, marker='.', label="$\phi$")
            ax.plot(x, m_shap*x+b_shap, color='blue', linewidth=.5)

            # Plot the target and the regression
            ax.scatter(x, y, alpha=0.3, marker='+', label="target")
            ax.plot(x, m_target*x+b_target, color='orange', linewidth=.5)

            ax.set_title(
                f"corr:{corr.correlation:.2f}, $m_S:${m_shap:.2f}, $m_y:${m_target:.2f}")

            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel(feature)
            ax.legend()

        return {'discrepancy': 1. - np.abs(corr.correlation),
                'm_shap': m_shap,
                'm_target': m_target}

    def _regress(self, x, y):
        """Fit a linear regression on x and y.

        Parameters
        ----------
        x : array
            x-coordinates of the data points.
        y : array
            y-coordinates of the data points.

        Returns
        -------
        slope : float
            The slope of the fitted line.
        intercept : float
            The y-intercept of the fitted line.
        """
        reg = sm.OLS(y, sm.add_constant(x)).fit()
        b, m = reg.params[0], reg.params[1]
        return b, m

    def adjust_edges_from_shap_discrepancies(
            self,
            graph: nx.DiGraph,
            feature_names: List[str],
            discrepancies: pd.DataFrame,
            increase_tolerance: float = 0.0,
            sd_upper: float = 0.1):
        """
        Adjust the edges of the graph based on the discrepancy index. This method
        removes edges that have a discrepancy index larger than the given standard
        deviation tolerance. The method also removes edges that have a discrepancy
        index larger than the discrepancy index of the target.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to adjust the edges of.
        feature_names : List[str]
            The names of the features in the graph.
        discrepancies : pd.DataFrame
            The discrepancy index for each pair of features.
        increase_tolerance : float, optional
            The tolerance for the increase in the discrepancy index, by default 0.0.
        sd_tol : float, optional
            The Shap Discrepancy tolerance, by default 0.02. This is the max difference
            between the SHAP Discrepancy in both causal directions.
        sd_upper : float, optional
            The upper bound for the Shap Discrepancy, by default 0.1. This is the max
            difference between the SHAP Discrepancy in both causal directions.
        verbose : bool, optional
            Whether to print the edges that are removed, by default False

        Returns
        -------
        networkx.DiGraph
            The graph with the edges adjusted.
        """

        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(graph.nodes())
        new_graph.add_edges_from(graph.edges())
        edges_reversed = []

        # Experimental
        if self._increase_upper_tolerance(discrepancies):
            increase_upper_tolerance = True
        else:
            increase_upper_tolerance = False

        for target in feature_names:
            target_mean = np.mean(discrepancies.loc[target].values)
            # Experimental
            if increase_upper_tolerance:
                tolerance = target_mean * increase_tolerance
            else:
                tolerance = 0.0

            # Iterate over the features and check if the edge should be reversed.
            for feature in feature_names:
                # If the edge is already reversed, skip it.
                if (target, feature) in edges_reversed or \
                        feature == target or \
                        not new_graph.has_edge(feature, target):
                    continue

                forward_sd = discrepancies.loc[target, feature]
                reverse_sd = discrepancies.loc[feature, target]
                diff = np.abs(forward_sd - reverse_sd)
                vector = [forward_sd, reverse_sd, diff, 0., 0., 0., 0.]

                # If the forward standard deviation is less than the reverse
                # standard deviation and within the tolerance range, attempt
                # to reverse the edge.
                if (forward_sd < reverse_sd) and \
                        (forward_sd < (target_mean + tolerance)):
                    # If the difference between the standard deviations is within
                    # the acceptable range, reverse the edge and check for cycles
                    # in the graph.
                    if diff < sd_upper and diff > 0.002:  # diff > sd_tol and
                        new_graph.remove_edge(feature, target)
                        new_graph.add_edge(target, feature)
                        edges_reversed.append((feature, target))
                        cycles = list(nx.simple_cycles(new_graph))
                        # If reversing the edge creates a cycle that includes both
                        # the target and feature nodes, reverse the edge back to
                        # its original direction and log the decision as discarded.
                        if len(cycles) > 0 and self._nodes_in_cycles(cycles, feature, target):
                            new_graph.remove_edge(target, feature)
                            edges_reversed.remove((feature, target))
                            new_graph.add_edge(feature, target)
                            self._debugmsg("Discarded(cycles)",
                                           target, target_mean, feature, tolerance,
                                           vector, sd_upper, cycles)
                        # If reversing the edge does not create a cycle, log the
                        # decision as reversed.
                        else:
                            self._debugmsg("(*) Reversed edge",
                                           target, target_mean, feature, tolerance,
                                           vector, sd_upper, cycles)
                    # If the difference between the standard deviations is not
                    # within the acceptable range, log the decision as discarded.
                    else:
                        self._debugmsg("Outside tolerance",
                                       target, target_mean, feature, tolerance,
                                       vector, sd_upper, [])
                # If the forward standard deviation is greater than the reverse
                # standard deviation and within the tolerance range, log the
                # decision as ignored.
                else:
                    self._debugmsg("Ignored edge",
                                   target, target_mean, feature, tolerance,
                                   vector, sd_upper, [])

        return new_graph

    def _nodes_in_cycles(self, cycles, feature, target) -> bool:
        """
        Check if the given nodes are in any of the cycles.
        """
        for cycle in cycles:
            if feature in cycle and target in cycle:
                return True
        return False

    def _increase_upper_tolerance(self, discrepancies: pd.DataFrame):
        """
        Increase the upper tolerance if the discrepancy matrix properties are
        suspicious. We found these suspicious values empirically in the polymoial case.
        """
        D = discrepancies.values
        det = np.linalg.det(D)
        norm = np.linalg.norm(D)
        cond = np.linalg.cond(D)
        m1 = "(*)" if det < -0.5 else "   "
        m2 = "(*)" if norm > .7 else "   "
        m3 = "(*)" if cond > 1500 else "   "
        if self.verbose:
            print(f"    {m2}{norm=:.2f} & ({m1}{det=:.2f} | {m3}{cond=:.2f})")
        if norm > 7.0 and (det < -0.5 or cond > 2000):
            return True
        return False

    def _input_vector(self, discrepancies, target, feature, target_mean):
        """
        Builds a vector with the values computed from the discrepancy index.
        Used to feed the model in _adjust_from_model method.
        """
        source_mean = np.mean(discrepancies.loc[feature].values)
        forward_sd = discrepancies.loc[target, feature]
        reverse_sd = discrepancies.loc[feature, target]
        diff = np.abs(forward_sd - reverse_sd)
        sdiff = np.abs(forward_sd + reverse_sd)
        d1 = diff / sdiff
        d2 = diff / forward_sd

        # Build the input for the model
        input_vector = np.array(
            [forward_sd, reverse_sd, diff, d1, d2,
             target_mean, source_mean])

        return input_vector

    def _debugmsg(
            self,
            msg,
            target,
            target_threshold,
            feature,
            tolerance,
            vector,
            sd_upper,
            cycles):
        if not self.verbose:
            return
        forward_sd, reverse_sd, diff, _, _, _, _ = vector
        fwd_bwd = f"{GREEN}<{RESET}" if forward_sd < reverse_sd else f"{RED}≮{RESET}"
        fwd_tgt = f"{GREEN}<{RESET}" if forward_sd < target_threshold + \
            tolerance else f"{RED}>{RESET}"
        diff_upper = f"{GREEN}<{RESET}" if diff < sd_upper else f"{RED}>{RESET}"
        print(f" -  {msg:<17s}: {feature} -> {target}",
              f"fwd|bwd({forward_sd:.3f}{fwd_bwd}{reverse_sd:.3f});",
              f"fwd{fwd_tgt}⍴({target_threshold:.3f}+{tolerance:.2f});",
              f"𝛿({diff:.3f}){diff_upper}Up({sd_upper:.2f}); "
              # f"𝛿({diff:.3f}){diff_tol}tol({sd_tol:.2f});",
              f"µ:{forward_sd/target_threshold:.3f}"
              )
        if len(cycles) > 0 and self._nodes_in_cycles(cycles, feature, target):
            print(f"    ~~ Cycles: {cycles}")

    def summary_plot(
            self,
            target_name,
            ax=None,
            max_features_to_display=20,
            **kwargs):
        """
        This code has been extracted from the summary_plot in SHAP package
        If you want to produce the original plot, simply type:

        >>> shap.summary_plot(shap_values, plot_type='bar', 
                              feature_names=feature_names_no_target)

        """
        feature_order = np.argsort(
            np.sum(np.abs(self.shap_values[target_name]), axis=0))
        feature_inds = feature_order[:max_features_to_display]
        mean_shap_values = np.abs(self.shap_values[target_name]).mean(0)

        plot_shap_summary(self.all_feature_names_, mean_shap_values, feature_inds, 
                          selected_features=None, ax=ax, **kwargs)
