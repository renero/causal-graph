"""
This module builds the causal graph based on the informacion that we derived
from the SHAP values. The main idea is to use the SHAP values to compute the
discrepancy between the SHAP values and the target values. This discrepancy
is then used to build the graph.

"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name, R0902:too-many-instance-attributes
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, W0621:redefined-outer-name
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

import inspect
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Union

import networkx as nx
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import statsmodels.stats.api as sms
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import kstest, spearmanr
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from mlforge.progbar import ProgBar  # type: ignore

from causalgraph.common import *
from causalgraph.common import utils
from causalgraph.explainability.hierarchies import Hierarchies
from causalgraph.independence.feature_selection import select_features


AnyGraph = Union[nx.DiGraph, nx.Graph]
K = 180.0 / math.pi


@dataclass
class ShapDiscrepancy:
    """
    A class representing the discrepancy between the SHAP value and the parent
    value for a given feature.

    Attributes:
        - target (str): The name of the target feature.
        - parent (str): The name of the parent feature.
        - shap_heteroskedasticity (bool): Whether the SHAP value exhibits
            heteroskedasticity.
        - parent_heteroskedasticity (bool): Whether the parent value exhibits
            heteroskedasticity.
        - shap_p_value (float): The p-value for the SHAP value.
        - parent_p_value (float): The p-value for the parent value.
        - shap_model (sm.regression.linear_model.RegressionResultsWrapper): The
            regression model for the SHAP value.
        - parent_model (sm.regression.linear_model.RegressionResultsWrapper): The
            regression model for the parent value.
        - shap_discrepancy (float): The discrepancy between the SHAP value and the
            parent value.
        - shap_correlation (float): The correlation between the SHAP value and the
            parent value.
        - ks_pvalue (float): The p-value for the Kolmogorov-Smirnov test.
        - ks_result (str): The result of the Kolmogorov-Smirnov test.
    """
    target: str
    parent: str
    shap_heteroskedasticity: bool
    parent_heteroskedasticity: bool
    shap_p_value: float
    parent_p_value: float
    shap_model: sm.regression.linear_model.RegressionResultsWrapper
    parent_model: sm.regression.linear_model.RegressionResultsWrapper
    shap_discrepancy: float
    shap_correlation: float
    shap_gof: float
    ks_pvalue: float
    ks_result: str


class ShapEstimator(BaseEstimator):
    """
    A class for computing SHAP values and building a causal graph from them.

    Parameters
    ----------
    explainer : str, default="explainer"
        The SHAP explainer to use. Possible values are "kernel", "gradient", and
        "explainer".
    models : BaseEstimator, default=None
        The models to use for computing SHAP values. If None, a linear regression
        model is used for each feature.
    correlation_th : float, default=None
        The correlation threshold to use for removing highly correlated features.
    mean_shap_percentile : float, default=0.8
        The percentile threshold for selecting features based on their mean SHAP value.
    iters : int, default=20
        The number of iterations to use for the feature selection method.
    reciprocity : bool, default=False
        Whether to enforce reciprocity in the causal graph.
    min_impact : float, default=1e-06
        The minimum impact threshold for selecting features.
    exhaustive : bool, default=False
        Whether to use the exhaustive (recursive) method for selecting features.
        If this is True, the threshold parameter must be provided, and the
        clustering is performed until remaining values to be clustered are below
        the given threshold.
    threshold : float, default=None
        The threshold to use when exhaustive is True. If None, exception is raised.
    on_gpu : bool, default=False
        Whether to use the GPU for computing SHAP values.
    verbose : bool, default=False
        Whether to print verbose output.
    prog_bar : bool, default=True
        Whether to show a progress bar.
    silent : bool, default=False
        Whether to suppress all output.
    """

    device = utils.select_device("cpu")

    def __init__(
            self,
            explainer: str = "explainer",
            models: BaseEstimator = None,
            correlation_th: float = None,
            mean_shap_percentile: float = 0.8,
            iters: int = 20,
            reciprocity: False = False,
            min_impact: float = 1e-06,
            exhaustive: bool = False,
            on_gpu: bool = False,
            verbose: bool = False,
            prog_bar: bool = True,
            silent: bool = False):
        """
        Initialize the ShapEstimator object.

        Parameters
        ----------
        explainer : str, default="explainer"
            The SHAP explainer to use. Possible values are "kernel", "gradient", and
            "explainer".
        models : BaseEstimator, default=None
            The models to use for computing SHAP values. If None, a linear regression
            model is used for each feature.
        correlation_th : float, default=None
            The correlation threshold to use for removing highly correlated features.
        mean_shap_percentile : float, default=0.8
            The percentile threshold for selecting features based on their
            mean SHAP value.
        iters : int, default=20
            The number of iterations to use for the feature selection method.
        reciprocity : bool, default=False
            Whether to enforce reciprocity in the causal graph.
        min_impact : float, default=1e-06
            The minimum impact threshold for selecting features.
        exhaustive : bool, default=False
            Whether to use the exhaustive (recursive) method for selecting features.
            If this is True, the threshold parameter must be provided, and the
            clustering is performed until remaining values to be clustered are below
            the given threshold.
        threshold : float, default=None
            The threshold to use when exhaustive is True. If None, exception is raised.
        on_gpu : bool, default=False
            Whether to use the GPU for computing SHAP values.
        verbose : bool, default=False
            Whether to print verbose output.
        prog_bar : bool, default=True
            Whether to show a progress bar.
        silent : bool, default=False
            Whether to suppress all output.
        """
        self.explainer = explainer
        self.models = models
        self.correlation_th = correlation_th
        self.mean_shap_percentile = mean_shap_percentile
        self.iters = iters
        self.reciprocity = reciprocity
        self.min_impact = min_impact
        self.exhaustive = exhaustive
        self.verbose = verbose
        self.prog_bar = prog_bar
        self.silent = silent
        self.on_gpu = on_gpu

        self._fit_desc = f"Running SHAP explainer ({self.explainer})"
        self._pred_desc = "Building graph skeleton"

    def __str__(self):
        return utils.stringfy_object(self)

    def fit(self, X):
        """
        Fit the ShapleyExplainer model to the given dataset.

        Parameters:
        - X: The input dataset.

        Returns:
        - self: The fitted ShapleyExplainer model.
        """
        # X, y = check_X_y(X, y, accept_sparse=True)
        self.feature_names = list(self.models.regressor.keys())
        self.shap_explainer = {}
        self.shap_values = {}
        # self.shap_scaled_values = {}
        self.shap_mean_values = {}
        self.feature_order = {}
        self.all_mean_shap_values = np.empty((0, ), dtype=np.float16)

        # Who is calling me?
        try:
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            caller_name = calframe[1][3]
            if caller_name == "__call__" or caller_name == "_run_step":
                caller_name = "ReX"
        except Exception:  # pylint: disable=broad-except
            caller_name = "unknown"

        if self.prog_bar and not self.verbose:
            pbar_name = f"({caller_name}) SHAP_fit"
            pbar = ProgBar().start_subtask(pbar_name, len(self.feature_names))
        else:
            pbar = None

        self.X_train, self.X_test = train_test_split(
            X, test_size=min(0.2, 250 / len(X)), random_state=42)

        # Make a copy of the data if correlation threshold is set, since I will have
        # to drop some features at each iteration.
        if self.correlation_th:
            self.corr_matrix = Hierarchies.compute_correlation_matrix(X)
            self.correlated_features = Hierarchies.compute_correlated_features(
                self.corr_matrix, self.correlation_th, self.feature_names,
                verbose=self.verbose)
            X_train_original = self.X_train.copy()
            X_test_original = self.X_test.copy()

        for target_idx, target_name in enumerate(self.feature_names):
            # if correlation_th is not None then, remove features that are highly
            # correlated with the target, at each step of the loop
            if self.correlation_th is not None:
                self.X_train = X_train_original.copy()
                self.X_test = X_test_original.copy()
                if len(self.correlated_features[target_name]) > 0:
                    self.X_train = self.X_train.drop(
                        self.correlated_features[target_name], axis=1)
                    self.X_test = self.X_test.drop(
                        self.correlated_features[target_name], axis=1)
                    if self.verbose:
                        print("REMOVED CORRELATED FEATURES: ",
                              self.correlated_features[target_name])

            # Get the model and the data (tensor form)
            if hasattr(self.models.regressor[target_name], "model"):
                model = self.models.regressor[target_name].model
                model = model.cuda() if self.on_gpu else model.cpu()
            else:
                model = self.models.regressor[target_name]

            X_train = self.X_train.drop(target_name, axis=1).values
            X_test = self.X_test.drop(target_name, axis=1).values

            # Run the selected SHAP explainer
            self.shap_values[target_name] = self._run_selected_shap_explainer(
                target_name, model, X_train, X_test)

            # self.shap_scaled_values[target_name] = self.shap_values[target_name]

            # Create the order list of features, in decreasing mean SHAP value
            self.feature_order[target_name] = np.argsort(
                np.sum(np.abs(self.shap_values[target_name]), axis=0))
            self.shap_mean_values[target_name] = np.abs(
                self.shap_values[target_name]).mean(0)
            self.all_mean_shap_values = np.concatenate(
                (self.all_mean_shap_values,
                 self.shap_mean_values[target_name]))
            if self.verbose:
                print(f"  Feature order for '{target_name}' "
                      f"{self.feature_order[target_name]}")
                print(f"  Target({target_name}) -> ", end="")
                srcs = [src for src in self.feature_names if src != target_name]
                for i in range(len(self.shap_mean_values[target_name])):
                    print(
                        f"{srcs[i]}:{self.shap_mean_values[target_name][i]:.3f};", end="")
                print()

            # Add zeroes to positions of correlated features
            if self.correlation_th is not None:
                self._add_zeroes(
                    target_name, self.correlated_features[target_name])

            pbar.update_subtask(pbar_name, target_idx + 1) if pbar else None

        pbar.remove(pbar_name) if pbar else None
        self.all_mean_shap_values = self.all_mean_shap_values.flatten()
        self._compute_scaled_shap_threshold()

        # Leave X_train and X_test as they originally were
        if self.correlation_th is not None:
            self.X_train = X_train_original
            self.X_test = X_test_original

        self.is_fitted_ = True

        return self

    def _compute_scaled_shap_threshold(self):
        """
        Compute the scaled SHAP threshold based on the given percentile.
        If the percentile is 0.0 or None, then the threshold is set to 0.0.
        """
        if self.mean_shap_percentile:
            self.mean_shap_threshold = np.quantile(
                self.all_mean_shap_values, self.mean_shap_percentile)
        else:
            self.mean_shap_threshold = 0.0

    def _run_selected_shap_explainer(self, target_name, model, X_train, X_test):
        """
        Run the selected SHAP explainer, according to the given parameters.

        Parameters
        ----------
        target_name : str
            The name of the target feature.
        model : torch.nn.Module
            The model for the given target.
        X_train : PyTorch.Tensor object
            The training data.
        X_test : PyTorch.Tensor object
            The testing data.

        Returns
        -------
        shap_values : np.ndarray
            The SHAP values for the given target.
        """
        if self.explainer == "kernel":
            self.shap_explainer[target_name] = shap.KernelExplainer(
                model.predict, X_train)
            shap_values = self.shap_explainer[target_name].\
                shap_values(X_test)[0]
        elif self.explainer == "gradient":
            X_train_tensor = torch.from_numpy(X_train).float()
            X_test_tensor = torch.from_numpy(X_test).float()
            self.shap_explainer[target_name] = shap.GradientExplainer(
                model.to(self.device), X_train_tensor.to(self.device))
            shap_values = self.shap_explainer[target_name](
                X_test_tensor.to(self.device)).values
        elif self.explainer == "explainer":
            self.shap_explainer[target_name] = shap.Explainer(
                model.predict, X_train)
            explanation = self.shap_explainer[target_name](X_test, silent=True)
            shap_values = explanation.values
        else:
            raise ValueError(
                f"Unknown explainer: {self.explainer}. "
                f"Please select one of: kernel, gradient, explainer.")

        return shap_values

    def _add_zeroes(self, target, correlated_features):
        features = [f for f in self.feature_names if f != target]
        for correlated_feature in correlated_features:
            correlated_feature_position = features.index(correlated_feature)
            self.all_mean_shap_values[-1] = np.insert(
                self.all_mean_shap_values[-1], correlated_feature_position, 0.)

    def predict(
            self,
            X,
            root_causes=None,
            prior: List[List[str]] = None) -> nx.DiGraph:
        """
        Builds a causal graph from the shap values using a selection mechanism based
        on clustering, knee or abrupt methods.

        Parameters
        ----------
        X : pd.DataFrame
            The input data. Consists of all the features in a pandas DataFrame.
        root_causes : List[str], optional
            The root causes of the graph. If None, all features are considered as
            root causes, by default None.
        prior : List[List[str]], optional
            The prior knowledge about the connections between the features. If None,
            all features are considered as valid candidates for the connections, by
            default None.

        Returns
        -------
        nx.DiGraph
            The causal graph.
        """
        if self.verbose:
            print("-----\nshap.predict()")

        check_is_fitted(self, 'is_fitted_')
        self.prior = prior

        # Recompute mean_shap_percentile here, in case it was changed
        self._compute_scaled_shap_threshold()

        # Who is calling me?
        try:
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            caller_name = calframe[1][3]
            if caller_name == "__call__" or caller_name == "_run_step":
                caller_name = "ReX"
        except Exception:  # pylint: disable=broad-except
            caller_name = "unknown"

        if self.prog_bar and (not self.verbose):
            pbar_name = f"({caller_name}) SHAP_predict"
            pbar = ProgBar().start_subtask(pbar_name, 4 + len(self.feature_names))
        else:
            pbar = None

        # Compute error contribution at this stage, since it needs the individual
        # SHAP values
        self.compute_error_contribution()
        pbar.update_subtask(pbar_name, 1) if pbar else None

        self._compute_discrepancies(self.X_test)
        pbar.update_subtask(pbar_name, 2) if pbar else None

        self.connections = {}
        for target_idx, target in enumerate(self.feature_names):
            # The valid parents are the features that are in the same level of the
            # hierarchy as the target, or in previous levels. In case prior is not
            # provided, all features are valid candidates.
            candidate_causes = utils.valid_candidates_from_prior(
                self.feature_names, target, self.prior)

            # Filter out features that are highly correlated with the target
            if self.correlation_th is not None:
                candidate_causes = [f for f in candidate_causes
                                    if f not in self.correlated_features[target]]

            print(
                f"Selecting features for target {target}...") if self.verbose else None

            feature_names_wo_target = [
                f for f in self.feature_names if f != target]

            # Select the features that are connected to the target
            self.connections[target] = select_features(
                values=self.shap_values[target],
                feature_names=feature_names_wo_target,
                min_impact=self.min_impact,
                exhaustive=self.exhaustive,
                threshold=self.mean_shap_threshold,
                verbose=self.verbose)
            pbar.update_subtask(
                pbar_name, target_idx + 3) if pbar else None

        dag = utils.digraph_from_connected_features(
            X, self.feature_names, self.models, self.connections, root_causes, prior,
            reciprocity=self.reciprocity, anm_iterations=self.iters,
            verbose=self.verbose)
        pbar.update_subtask(pbar_name, len(
            self.feature_names) + 3) if pbar else None

        dag = utils.break_cycles_if_present(
            dag, self.shap_discrepancies, self.prior, verbose=self.verbose)
        pbar.update_subtask(pbar_name, len(
            self.feature_names) + 4) if pbar else None

        pbar.remove(pbar_name) if pbar else None

        return dag

    def adjust(
            self,
            graph: AnyGraph,
            increase_tolerance: float = 0.0,
            sd_upper: float = 0.1):

        # self._compute_shap_discrepancies(X)
        new_graph = self._adjust_edges_from_shap_discrepancies(
            graph, increase_tolerance, sd_upper)
        return new_graph

    def _compute_discrepancies(self, X: pd.DataFrame):
        """
        Compute the discrepancies between the SHAP values and the target values
        for all features and all targets.

        Parameters
        ----------
        X : pd.DataFrame
            The input data. Consists of all the features in a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the discrepancies for all features and all targets.
        """
        check_is_fitted(self, 'is_fitted_')
        self.discrepancies = pd.DataFrame(columns=self.feature_names)
        self.shap_discrepancies = defaultdict(dict)
        X_original = X.copy() if self.correlation_th else None
        for target_name in self.feature_names:
            # Check if we must remove correlated features
            if self.correlation_th is not None:
                X = X_original.copy()
                if len(self.correlated_features[target_name]) > 0:
                    X = X.drop(self.correlated_features[target_name], axis=1)
                    if self.verbose:
                        print(
                            f"REMOVED CORRELATED FEATURES ({target_name}): "
                            f"{self.correlated_features[target_name]}")

            X_features = X.drop(target_name, axis=1)
            y = X[target_name].values

            if self.correlation_th is not None:
                feature_names = [
                    f for f in self.feature_names if (f != target_name) &
                    (f not in self.correlated_features[target_name])]
            else:
                feature_names = [
                    f for f in self.feature_names if f != target_name]

            self.discrepancies.loc[target_name] = 0
            self.shap_discrepancies[target_name] = defaultdict(ShapDiscrepancy)

            # Loop through all features and compute the discrepancy
            for parent_name in feature_names:
                # Take the data that is needed at this iteration
                parent_data = X_features[parent_name].values
                parent_pos = feature_names.index(parent_name)
                shap_data = self.shap_values[target_name][:, parent_pos]

                # Form three vectors to compute the discrepancy
                x = parent_data.reshape(-1, 1)
                s = shap_data.reshape(-1, 1)

                # Compute the discrepancy
                self.shap_discrepancies[target_name][parent_name] = \
                    self._compute_discrepancy(
                        x, y, s,
                        target_name,
                        parent_name)
                SD = self.shap_discrepancies[target_name][parent_name]
                self.discrepancies.loc[target_name,
                                       parent_name] = SD.shap_discrepancy

        return self.discrepancies

    def _compute_discrepancy(self, x, y, s, target_name, parent_name) -> ShapDiscrepancy:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x = x.values
        elif not isinstance(x, np.ndarray):
            x = np.array(x)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        elif not isinstance(y, np.ndarray):
            y = np.array(y)

        X = sm.add_constant(x)
        model_y = sm.OLS(y, X).fit()
        model_s = sm.OLS(s, X).fit()

        # Heteroskedasticity tests:
        # The null hypothesis (H0): Signifies that Homoscedasticity is present
        # The alternative hypothesis (H1): Signifies that Heteroscedasticity is present
        # If the p-value is less than the significance level (0.05), we reject the
        # null hypothesis and conclude that heteroscedasticity is present.
        test_shap = sms.het_breuschpagan(model_s.resid, X)
        shap_heteroskedasticity = test_shap[1] < 0.05

        test_parent = sms.het_breuschpagan(model_y.resid, X)
        parent_heteroskedasticity = test_parent[1] < 0.05

        corr = spearmanr(s, y)
        discrepancy = 1 - np.abs(corr.correlation)
        # The p-value is below 5%: we reject the null hypothesis that the two
        # distributions are the same, with 95% confidence.
        _, ks_pvalue = kstest(s[:, 0], y)

        return ShapDiscrepancy(
            target=target_name,
            parent=parent_name,
            shap_heteroskedasticity=shap_heteroskedasticity,
            parent_heteroskedasticity=parent_heteroskedasticity,
            shap_p_value=test_shap[1],
            parent_p_value=test_parent[1],
            shap_model=model_s,
            parent_model=model_y,
            shap_discrepancy=discrepancy,
            shap_correlation=corr.correlation,
            shap_gof=r2_score(y, s),
            ks_pvalue=ks_pvalue,
            ks_result="Equal" if ks_pvalue > 0.05 else "Different"
        )

    def _adjust_edges_from_shap_discrepancies(
            self,
            graph: nx.DiGraph,
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
        increase_tolerance : float, optional
            The tolerance for the increase in the discrepancy index, by default 0.0.
        sd_upper : float, optional
            The upper bound for the Shap Discrepancy, by default 0.1. This is the max
            difference between the SHAP Discrepancy in both causal directions.

        Returns
        -------
        networkx.DiGraph
            The graph with the edges adjusted.
        """

        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(graph.nodes(data=True))
        new_graph.add_edges_from(graph.edges())
        edges_reversed = []

        # Experimental
        if self._increase_upper_tolerance(self.discrepancies):
            increase_upper_tolerance = True
        else:
            increase_upper_tolerance = False

        for target in self.feature_names:
            target_mean = np.mean(self.discrepancies.loc[target].values)
            # Experimental
            if increase_upper_tolerance:
                tolerance = target_mean * increase_tolerance
            else:
                tolerance = 0.0

            # Iterate over the features and check if the edge should be reversed.
            for feature in self.feature_names:
                # If the edge is already reversed, skip it.
                if (target, feature) in edges_reversed or \
                        feature == target or \
                        not new_graph.has_edge(feature, target):
                    continue

                forward_sd = self.discrepancies.loc[target, feature]
                reverse_sd = self.discrepancies.loc[feature, target]
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
                    if diff < sd_upper and diff > 0.002:
                        new_graph.remove_edge(feature, target)
                        new_graph.add_edge(target, feature)
                        edges_reversed.append((feature, target))
                        cycles = list(nx.simple_cycles(new_graph))
                        # If reversing the edge creates a cycle that includes both
                        # the target and feature nodes, reverse the edge back to
                        # its original direction and log the decision as discarded.
                        if len(cycles) > 0 and \
                                self._nodes_in_cycles(cycles, feature, target):
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
        D = np.nan_to_num(D)
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

    def _adjust_predictions_shape(self, predictions, target_shape):
        # Concatenate if predictions is a list
        if isinstance(predictions, list):
            predictions = np.concatenate(predictions)
        else:
            predictions = np.array(predictions)

        # Reshape if necessary
        if predictions.shape != target_shape:
            # Flatten
            predictions = predictions.reshape(target_shape)

        return predictions

    def compute_error_contribution(self):
        """
        Computes the error contribution of each feature for each target.
        If this value is positive, then it means that, on average, the presence of
        the feature in the model leads to a higher error. Thus, without that feature,
        the prediction would have been generally better. In other words, the feature
        is making more harm than good!
        On the contrary, the more negative this value, the more beneficial
        the feature is for the predictions since its presence leads to smaller errors.

        Returns:
        --------
        err_contrib: pd.DataFrame
            Error contribution of each feature for each target.
        """
        error_contribution = dict()
        predictions = self.models.predict(self.X_test)

        predictions = self._adjust_predictions_shape(
            predictions, (self.X_test.shape[0], self.X_test.shape[1]))

        y_hat = pd.DataFrame(predictions, columns=self.feature_names)
        y_true = self.X_test
        for target in self.feature_names:
            shap_values = pd.DataFrame(
                self.shap_values[target],
                columns=[c for c in self.feature_names if c != target])
            error_contribution[target] = self._individual_error_contribution(
                shap_values, y_true[target], y_hat[target])
            #  Add a 0.0 value at index target in the error contribution series
            error_contribution[target] = error_contribution[target].append(
                pd.Series(0.0, index=[target]))
            # Sort the series by index
            error_contribution[target] = error_contribution[target].sort_index()

        self.error_contribution = pd.DataFrame(error_contribution)
        return self.error_contribution

    def _individual_error_contribution(self, shap_values, y_true, y_pred):
        """
        Compute the error contribution of each feature.
        If this value is positive, then it means that, on average, the presence of
        the feature in the model leads to a higher error. Thus, without that feature,
        the prediction would have been generally better. In other words, the feature
        is making more harm than good!
        On the contrary, the more negative this value, the more beneficial the
        feature is for the predictions since its presence leads to smaller errors.

        Parameters:
        -----------
        shap_values: pd.DataFrame
            Shap values for a given target.
        y_true: pd.Series
            Ground truth values for a given target.
        y_pred: pd.Series
            Predicted values for a given target.

        Returns:
        --------
        error_contribution: pd.Series
            Error contribution of each feature.
        """
        abs_error = (y_true - y_pred).abs()
        y_pred_wo_feature = shap_values.apply(lambda feature: y_pred - feature)
        abs_error_wo_feature = y_pred_wo_feature.apply(
            lambda feature: (y_true-feature).abs())
        error_diff = abs_error_wo_feature.apply(
            lambda feature: abs_error - feature)
        ind_error_contribution = error_diff.mean()

        return ind_error_contribution

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

    def _plot_shap_summary(
            self,
            target_name: str,
            ax,
            max_features_to_display: int = 20,
            **kwargs):
        """
        Plots the summary of the SHAP values for a given target.

        Arguments:
        ----------
            feature_names: List[str]
                The names of the features.
            mean_shap_values: np.array
                The mean SHAP values for each feature.
            feature_inds: np.array
                The indices of the features to be plotted.
            selected_features: List[str]
                The names of the selected features.
            ax: Axis
                The axis in which to plot the summary. If None, a new figure is created.
            **kwargs: Dict
                Additional arguments to be passed to the plot.
        """

        figsize_ = kwargs.get('figsize', (6, 3))
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize_)

        feature_inds = self.feature_order[target_name][:max_features_to_display]
        if self.correlation_th is not None:
            feature_names = [f for f in self.feature_names if (f != target_name) & (
                f not in self.correlated_features[target_name])]
        else:
            feature_names = [
                f for f in self.feature_names if f != target_name]
        selected_features = list(self.connections[target_name])

        y_pos = np.arange(len(feature_inds))
        ax.grid(True, axis='x')
        ax.barh(y_pos, self.shap_mean_values[target_name][feature_inds],
                0.7, align='center', color="#0e73fa", alpha=0.8)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        ax.set_yticks(y_pos, [feature_names[i] for i in feature_inds])
        ax.set_xlabel("Avg. SHAP value")
        ax.set_title(
            target_name + r' $\leftarrow$ ' +
            (','.join(selected_features) if selected_features else 'ø'))

        # Recompute mean_shap_percentile here, in case it was changed
        self._compute_scaled_shap_threshold()

        xlims = ax.get_xlim()
        if xlims[1] < self.mean_shap_threshold:
            ax.set_xlim(right=self.mean_shap_threshold +
                        ((xlims[1] - xlims[0]) / 20))

        ax.axvline(x=self.mean_shap_threshold, color='red', linestyle='--',
                   linewidth=0.5)

        fig = ax.figure if fig is None else fig
        return fig

    def _plot_discrepancies(self, target_name: str, threshold: float = 10.0, **kwargs):
        """
        Plot the discrepancies between the target variable and each feature.

        Parameters
        ----------
        - target_name (str)
            The name of the target variable.
        - threshold (float)
            The threshold to use for selecting what features to plot. Only
            features with a discrepancy index below this threshold will be plotted.

        Returns:
            None
        """
        pass


def custom_main(exp_name,
                path="/Users/renero/phd/data/RC4/",
                output_path="/Users/renero/phd/output/RC4/",
                scale=False):
    """
    Runs a custom main function for the given experiment name.

    Args:
        experiment_name (str): The name of the experiment to run.
        path (str): The path to the data files.
        output_path (str): The path to the output files.

    Returns:
        None
    """

    ref_graph = utils.graph_from_dot_file(f"{path}{exp_name}.dot")
    data = pd.read_csv(f"{path}{exp_name}.csv")
    if scale:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    # Split the dataframe into train and test
    train = data.sample(frac=0.9, random_state=42)
    test = data.drop(train.index)

    rex = utils.load_experiment(f"{exp_name}_nn", output_path)
    rex.is_fitted_ = True
    print(f"Loaded experiment {exp_name}")

    rex.shaps = ShapEstimator(
        explainer="gradient",
        models=rex.models,
        correlation_th=0.5,
        mean_shap_percentile=0.8,
        iters=20,
        reciprocity=True,
        min_impact=1e-06,
        on_gpu=False,
        verbose=False,
        prog_bar=True,
        silent=False)
    rex.shaps.fit(train)
    rex.shaps.predict(test, rex.root_causes)


def sachs_main():
    experiment_name = "sachs_long"
    path = "/Users/renero/phd/data/RC3/"
    output_path = "/Users/renero/phd/output/RC3/"

    ref_graph = utils.graph_from_dot_file(f"{path}{experiment_name}.dot")
    data = pd.read_csv(f"{path}{experiment_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Split the dataframe into train and test
    train = data.sample(frac=0.9, random_state=42)
    test = data.drop(train.index)

    rex = utils.load_experiment(f"{experiment_name}_gbt", output_path)
    rex.is_fitted_ = True
    rex.shaps.is_fitted_ = True
    print(f"Loaded experiment {experiment_name}")

    rex.shaps.prog_bar = False
    rex.shaps.verbose = True
    rex.shaps.iters = 100
    rex.shaps.predict(test, rex.root_causes)
    print("fininshed")


if __name__ == "__main__":
    custom_main('toy_dataset')
