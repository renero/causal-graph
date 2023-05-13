"""
This is a module to be used as a reference for building other modules
"""
from dataclasses import dataclass
from typing import List, Union
import warnings
from matplotlib import pyplot as plt
import matplotlib

import networkx as nx
import numpy as np
import pandas as pd
import shap
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import HuberRegressor
import statsmodels.api as sm
import torch
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

from causalgraph.common import *
from causalgraph.independence.edge_orientation import get_edge_orientation
from causalgraph.independence.feature_selection import select_features
from causalgraph.models.dnn import NNRegressor
from causalgraph.common.utils import graph_from_dot_file, load_experiment

AnyGraph = Union[nx.DiGraph, nx.Graph]


@dataclass
class ShapDiscrepancy:
    target: str
    parent: str
    shap_discrepancy: float
    intercept_shap: float
    slope_shap: float
    intercept_parent: float
    slope_parent: float
    correlation: float


class ShapEstimator(BaseEstimator):
    """
    """

    def __init__(
            self,
            models: NNRegressor = None,
            method: str = 'cluster',
            sensitivity: float = 1.0,
            tolerance: float = None,
            descending: bool = False,
            iters: int = 20,
            reciprocity: False = False,
            min_impact: float = 1e-06,
            on_gpu: bool = False,
            verbose: bool = False,
            prog_bar: bool = True):
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
        self.prog_bar = prog_bar
        self.on_gpu = on_gpu

        self._fit_desc = "Running SHAP explainer"
        self._pred_desc = "Building graph skeleton"

    def fit(self, X, y=None):
        """
        """
        # X, y = check_X_y(X, y, accept_sparse=True)

        self.all_feature_names_ = list(self.models.regressor.keys())
        self.shap_values = dict()

        pbar = tqdm(total=len(self.all_feature_names_),
                    **tqdm_params(self._fit_desc, self.prog_bar))

        for target_name in self.all_feature_names_:
            pbar.update(1)
            model = self.models.regressor[target_name].model
            tensor_data = X.drop(target_name, axis=1).values
            tensor_data = torch.from_numpy(tensor_data).float()

            if self.on_gpu:
                model = model.cuda()
                tensor_data = tensor_data.cuda()

            explainer = shap.GradientExplainer(model, tensor_data)

            self.shap_values[target_name] = explainer.shap_values(tensor_data)
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

        pbar = tqdm(total=4,
                    **tqdm_params("Building graph from SHAPs", self.prog_bar))

        self._compute_shap_discrepancies(X)

        pbar.update(1)
        pbar.refresh()

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
                min_impact=self.min_impact, verbose=self.verbose)

        pbar.update(1)
        pbar.refresh()

        G_shap_unoriented = nx.Graph()
        for target in self.all_feature_names_:
            for parent in self.parents[target]:
                # Add edges ONLY between nodes where SHAP recognizes both directions
                if self.reciprocity:
                    if target in self.parents[parent]:
                        G_shap_unoriented.add_edge(target, parent)
                else:
                    G_shap_unoriented.add_edge(target, parent)

        pbar.update(1)
        pbar.refresh()

        G_shap = nx.DiGraph()
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

        pbar.update(1)
        pbar.refresh()
        pbar.close()

        return G_shap

    def adjust(
            self,
            X: np.ndarray,
            graph: AnyGraph,
            increase_tolerance: float = 0.0,
            sd_upper: float = 0.1):

        # self._compute_shap_discrepancies(X)
        new_graph = self._adjust_edges_from_shap_discrepancies(
            graph, increase_tolerance, sd_upper)
        return new_graph

    def _compute_shap_discrepancies(self, X: pd.DataFrame):
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
        self.discrepancies = pd.DataFrame(columns=self.all_feature_names_)
        self.shap_discrepancies = dict()
        for target_name in tqdm(
                self.all_feature_names_,
                **tqdm_params("Computing Shap discrepancies", self.prog_bar)):
            y = X[target_name]
            X_features = X.drop(target_name, axis=1)
            feature_names = [
                f for f in self.all_feature_names_ if f != target_name]
            self.discrepancies.loc[target_name] = 0
            self.shap_discrepancies[target_name] = dict()

            # Loop through all features and compute the discrepancy
            for feature in feature_names:
                discrepancy = self._compute_shap_alignment(
                    X_features[feature].values,
                    y.values,
                    self.shap_values[target_name],
                    target_name,
                    feature,
                    feature_names)
                self.discrepancies.loc[target_name,
                                       feature] = discrepancy.shap_discrepancy
                self.shap_discrepancies[target_name][feature] = discrepancy

        return self.discrepancies

    def _compute_shap_alignment(
            self,
            x,
            y,
            shap_values,
            target_name,
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
        target_name : str
            The name of the target variable in the main model from which the SHAP
            values were computed.
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
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            x = x.values
        elif not isinstance(x, np.ndarray):
            x = np.array(x)
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values
        elif not isinstance(y, np.ndarray):
            y = np.array(y)

        feature_pos = feature_names.index(feature)

        # Normalize the data
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        phi = StandardScaler().fit_transform(
            shap_values[:, feature_pos].reshape(-1, 1))

        # Compute the robust regression to the shap values
        b_shap, m_shap = self._regress(x, phi)
        b_parent, m_parent = self._regress(x, y)
        corr = spearmanr(phi, y)
        discrepancy = 1 - corr.correlation

        result = ShapDiscrepancy(
            target=target_name,
            parent=feature,
            shap_discrepancy=discrepancy,
            intercept_shap=b_shap,
            slope_shap=m_shap,
            intercept_parent=b_parent,
            slope_parent=m_parent,
            correlation=corr.correlation
        )

        return result

    def _regress(self, x: np.array, y: np.array):
        """Fit a linear regression on x and y.

        Parameters
        ----------
        x : array
            x-coordinates of the data points.
        y : array
            y-coordinates of the data points.

        Returns
        -------
        intercept : float
            The y-intercept of the fitted line.
        slope : float
            The slope of the fitted line.
        """
        # reg = sm.OLS(y, sm.add_constant(x)).fit()
        # b, m = reg.params[0], reg.params[1]
        # return b, m
        reg = HuberRegressor().fit(x, y)
        b, m = reg.intercept_, reg.coef_[0]
        return b, m

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
        new_graph.add_nodes_from(graph.nodes())
        new_graph.add_edges_from(graph.edges())
        edges_reversed = []

        # Experimental
        if self._increase_upper_tolerance(self.discrepancies):
            increase_upper_tolerance = True
        else:
            increase_upper_tolerance = False

        for target in self.all_feature_names_:
            target_mean = np.mean(self.discrepancies.loc[target].values)
            # Experimental
            if increase_upper_tolerance:
                tolerance = target_mean * increase_tolerance
            else:
                tolerance = 0.0

            # Iterate over the features and check if the edge should be reversed.
            for feature in self.all_feature_names_:
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

    def _get_data_from_model(self, target_name: str):
        model = self.models.regressor[target_name]
        tensor_features = model.train_loader.dataset.features
        tensor_target = model.train_loader.dataset.target
        # Convert tensor to numpy array
        tensor_data = tensor_features.data.cpu().numpy()
        tensor_target = tensor_target.data.cpu().numpy()
        return tensor_data, tensor_target

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
            target_name: str,
            ax=None,
            max_features_to_display: int = 20,
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

        return self.plot_shap_summary(
            [feat for feat in self.all_feature_names_ if feat != target_name],
            target_name,
            mean_shap_values,
            feature_inds,
            selected_features=[parent for parent in self.parents[target_name]],
            ax=ax,
            **kwargs)

    def plot_shap_summary(
            self,
            feature_names: List[str],
            target_name: str,
            mean_shap_values,
            feature_inds,
            selected_features,
            ax,
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

        y_pos = np.arange(len(feature_inds))
        ax.grid(True, axis='x')
        ax.barh(y_pos, mean_shap_values[feature_inds],
                0.7, align='center', color="#0e73fa", alpha=0.8)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        ax.set_yticks(y_pos, [feature_names[i] for i in feature_inds])
        ax.set_xlabel("Avg. SHAP value")
        ax.set_title(
            target_name + " $\leftarrow$ " +
            (','.join(selected_features) if selected_features else 'ø'))
        fig = ax.figure if fig is None else fig
        return fig

    def _plot_discrepancies_for_target(
            self,
            X: np.ndarray,
            y: np.ndarray,
            target_name: str,
            **kwargs) -> None:
        """
        Plot the discrepancies for a target. This method extracts data from the pipeline
        computes the shap values again, and plot them, showing the correlation and the 
        slopes of the regression lines for SHAP values and target values.

        Parameters
        ----------
        X : np.ndarray
            All the features except the target in a numpy array.
        y : np.ndarray
            The target values in a numpy array.
        target_name : str
            The name of the target to compute the discrepancies for.

        Returns
        -------
        None
        """
        # Setup plot
        dpi_ = kwargs.get('dpi', 100)
        figsize_ = kwargs.get('figsize', (10, 10))
        num_columns = kwargs.get('num_columns', 3)
        num_rows = int(np.ceil(10 / num_columns))

        _, ax = plt.subplots(num_rows, num_columns, figsize=figsize_, dpi=dpi_)

        # Setup data
        feature_names = list(self.all_feature_names_)
        feature_names.remove(target_name)
        # shap_values = self.shap_values[target_name]

        for idx, feature in enumerate(feature_names):
            row, col = idx//num_columns, idx % num_columns
            # self._compute_shap_alignment(X[:, idx], y, shap_values,
            #                              feature, feature_names, ax=ax[row, col],
            #                              plot=True)
            self._plot_discrepancy(X[:, idx], y, self.shap_values[target_name][:, idx],
                                   self.shap_discrepancies[target_name][feature], 
                                   ax=ax[row, col])

        # Remove empty plots
        for idx in range(len(feature_names), 12):
            row, col = idx//num_columns, idx % num_columns
            ax[row, col].set_visible(False)

        plt.suptitle(f"Discrepancies for {target_name}")
        plt.tight_layout()
        plt.show()

    def _plot_discrepancy(
            self,
            x: np.ndarray,
            y: np.ndarray,
            phi: np.ndarray,
            discrepancy: ShapDiscrepancy,
            ax: matplotlib.axes.Axes):
        """
        """
        # Plot the SHAP values and the regression
        ax.scatter(x, phi, alpha=0.3, marker='.', label="$\phi$")
        # ax.plot(x, m_shap * x + b_shap, color='blue', linewidth=.5)
        ax.plot(x, discrepancy.slope_shap * x + discrepancy.intercept_shap,
                color='blue', linewidth=.5)

        # Plot the target and the regression
        ax.scatter(x, y, alpha=0.3, marker='+', label="target")
        # ax.plot(x, m_parent*x+b_parent, color='red', linewidth=.5)
        ax.plot(x, discrepancy.slope_parent * x + discrepancy.intercept_parent,
                color='red', linewidth=.5)

        ax.set_title(
            f"corr:{discrepancy.correlation:.2f}, $m_S:${discrepancy.slope_shap:.2f}, $m_y:${discrepancy.slope_parent:.2f}",
            fontsize=8)

        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel(discrepancy.parent)
        ax.legend(loc='best', fontsize=8)


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=150)
    warnings.filterwarnings('ignore')

    dataset_name = 'generated_linear_10'
    data = pd.read_csv("~/phd/data/generated_linear_10.csv")
    ref_graph = graph_from_dot_file(
        "/Users/renero/phd/data/generated_linear_10_mini.dot")
    rex = load_experiment('rex', "/Users/renero/phd/output/REX")
    rex.prog_bar = False
    rex.verbose = True
    rex.shaps = ShapEstimator(rex.regressor).fit(data)
