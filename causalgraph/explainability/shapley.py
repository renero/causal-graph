"""
This is a module to be used as a reference for building other modules
"""
import math
import types
import warnings
from dataclasses import dataclass
from typing import List, Union

import matplotlib
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
from sklearn.isotonic import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

from causalgraph.common import *
from causalgraph.common.utils import graph_from_dot_file, load_experiment
from causalgraph.independence.edge_orientation import get_edge_orientation
from causalgraph.independence.feature_selection import select_features
from causalgraph.models.dnn import NNRegressor

AnyGraph = Union[nx.DiGraph, nx.Graph]
K = 180.0 / math.pi


@dataclass
class ShapDiscrepancy:
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
    ks_pvalue: float
    ks_result: str


class ShapEstimator(BaseEstimator):
    """
    """

    def __init__(
            self,
            explainer = shap.Explainer,
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
        self.explainer = explainer
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

        self._fit_desc = f"Running SHAP explainer ({explainer.__name__})"
        self._pred_desc = "Building graph skeleton"

    def __repr__(self):
        forbidden_attrs = ['fit', 'predict',
                           'score', 'get_params', 'set_params']
        ret = f"{GREEN}SHAP object attributes{RESET}\n"
        ret += f"{GRAY}{'-'*80}{RESET}\n"
        for attr in dir(self):
            if attr.startswith('_') or attr in forbidden_attrs or type(getattr(self, attr)) == types.MethodType:
                continue
            elif attr == "X" or attr == "y":
                if isinstance(getattr(self, attr), pd.DataFrame):
                    ret += f"{attr:25} {getattr(self, attr).shape}\n"
                    continue
                elif isinstance(getattr(self, attr), nx.DiGraph):
                    n_nodes = getattr(self, attr).number_of_nodes()
                    n_edges = getattr(self, attr).number_of_edges()
                    ret += f"{attr:25} {n_nodes} nodes, {n_edges} edges\n"
                    continue
            elif isinstance(getattr(self, attr), pd.DataFrame):
                ret += f"{attr:25} DataFrame {getattr(self, attr).shape}\n"
            elif isinstance(getattr(self, attr), dict):
                keys_list = [f"{k}:{type(getattr(self, attr)[k])}" for k in getattr(self, attr).keys()]
                ret += f"{attr:25} dict {keys_list}\n"
            else:
                ret += f"{attr:25} {getattr(self, attr)}\n"

        return ret

    def fit(self, X, y=None):
        """
        """
        # X, y = check_X_y(X, y, accept_sparse=True)

        self.all_feature_names_ = list(self.models.regressor.keys())
        self.shap_values = dict()
        self.shap_scaled_values = dict()
        self.shap_mean_values = dict()
        self.feature_order = dict()

        pbar = tqdm(total=len(self.all_feature_names_),
                    **tqdm_params(self._fit_desc, self.prog_bar))
        
        self.X_train, self.X_test = train_test_split(X, test_size=0.2, random_state=42)

        for target_name in self.all_feature_names_:
            pbar.update(1)

            # Get the model and the data (tensor form)
            model = self.models.regressor[target_name].model
            model = model.cuda() if self.on_gpu else model.cpu()
            # if self.explainer == shap.DeepExplainer or self.explainer == shap.GradientExplainer:
            X_train_tensor = torch.tensor(self.X_train.drop(target_name, axis=1).values).float()
            X_test_tensor = torch.tensor(self.X_test.drop(target_name, axis=1).values).float()
            X_train = X_train_tensor.cpu().numpy()
            X_test = X_test_tensor.cpu().numpy()

            # Run the selected SHAP explainer
            my_explainer = self.explainer(model.predict, X_train)
            self.shap_values[target_name] = my_explainer(X_test).values

            # Scale the SHAP values
            scaler = StandardScaler()
            self.shap_scaled_values[target_name] = scaler.fit_transform(
                self.shap_values[target_name])

            # Create the order list of features, in decreasing mean SHAP value
            self.feature_order[target_name] = np.argsort(
                np.sum(np.abs(self.shap_values[target_name]), axis=0))
            self.shap_mean_values[target_name] = np.abs(
                self.shap_values[target_name]).mean(0)

            pbar.refresh()

        pbar.close()

        self.is_fitted_ = True
        return self

    def _extract_data(self, X, target_name):
        """
        Extract the data and the model for the given target.

        Parameters
        ----------
        model: torch.nn.Module
            The model for the given target.
        X : pd.DataFrame
            The input data.
        target_name : str
            The name of the target feature.

        Returns
        -------
        X_train : PyTorch.Tensor object
            The training data.
        X_test : PyTorch.Tensor object
            The testing data.
        """
        tensor_data = X.drop(target_name, axis=1).values
        tensor_data = torch.from_numpy(tensor_data).float()

        X_train, X_test = self._train_test_split_tensors(
            tensor_data, test_size=0.2, random_state=42)

        # Move to GPU if available
        if self.on_gpu:
            X_train = X_train.cuda()
            X_test = X_test.cuda()

        return X_train, X_test
    
    def predict(self, X):
        """
        Builds a causal graph from the shap values using a selection mechanism based
        on the knee or abrupt methods.
        """
        # X_array = check_array(X)
        check_is_fitted(self, 'is_fitted_')

        pbar = tqdm(total=4,
                    **tqdm_params("Building graph from SHAPs", self.prog_bar))

        self._compute_discrepancies(self.X_test) # (X)

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
        self.discrepancies = pd.DataFrame(columns=self.all_feature_names_)
        self.shap_discrepancies = dict()
        for target_name in tqdm(
                self.all_feature_names_,
                **tqdm_params("Computing Shap discrepancies", self.prog_bar)):

            X_features = X.drop(target_name, axis=1)
            y = X[target_name].values

            feature_names = [
                f for f in self.all_feature_names_ if f != target_name]

            self.discrepancies.loc[target_name] = 0
            self.shap_discrepancies[target_name] = dict()

            # Loop through all features and compute the discrepancy
            for parent_name in feature_names:
                # Take the data that is needed at this iteration
                parent_data = X_features[parent_name].values
                parent_pos = feature_names.index(parent_name)
                shap_data = self.shap_scaled_values[target_name][:, parent_pos]

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
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            x = x.values
        elif not isinstance(x, np.ndarray):
            x = np.array(x)
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
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
        shap_heteroskedasticity = test_shap[1] > 0.05

        test_parent = sms.het_breuschpagan(model_y.resid, X)
        parent_heteroskedasticity = test_parent[1] > 0.05

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
        feature_names = [f for f in self.all_feature_names_ if f != target_name]
        selected_features = [parent for parent in self.parents[target_name]]

        y_pos = np.arange(len(feature_inds))
        ax.grid(True, axis='x')
        ax.barh(y_pos, self.shap_mean_values[target_name][feature_inds],
                0.7, align='center', color="#0e73fa", alpha=0.8)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        ax.set_yticks(y_pos, [feature_names[i] for i in feature_inds])
        ax.set_xlabel("Avg. SHAP value")
        ax.set_title(
            target_name + " $\leftarrow$ " +
            (','.join(selected_features) if selected_features else 'ø'))
        fig = ax.figure if fig is None else fig
        return fig

    def _plot_discrepancies(self, X: pd.DataFrame, target_name: str, **kwargs):
        figsize_ = kwargs.get('figsize', (10, 16))
        feature_names = [
            f for f in self.all_feature_names_ if f != target_name]
        fig, ax = plt.subplots(len(feature_names), 4, figsize=figsize_)

        for i, parent_name in enumerate(feature_names):
            r = self.shap_discrepancies[target_name][parent_name]
            x = self.X_test[parent_name].values.reshape(-1, 1)
            y = self.X_test[target_name].values.reshape(-1, 1)
            parent_pos = feature_names.index(parent_name)
            s = self.shap_scaled_values[target_name][:,
                                                     parent_pos].reshape(-1, 1)
            self._plot_discrepancy(x, y, s, target_name, parent_name, r, ax[i])

        plt.suptitle(f"Discrepancies for {target_name}")
        plt.tight_layout(rect=[0, 0.0, 1, 0.97])
        fig.show()

    def _plot_discrepancy(self, x, y, s, target_name, parent_name, r, ax):
        def _remove_ticks_and_box(ax):
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

        b0_s, b1_s = r.shap_model.params[0], r.shap_model.params[1]
        b0_y, b1_y = r.parent_model.params[0], r.parent_model.params[1]
        shap_label = "HET" if r.shap_heteroskedasticity else "HOM"
        parent_label = "HET" if r.parent_heteroskedasticity else "HOM"

        # Represent scatter plots
        ax[0].scatter(x, s, alpha=0.5, marker='.')
        ax[0].scatter(x, y, alpha=0.5, marker='.')
        ax[0].plot(x, b1_s * x + b0_s, color='blue', linewidth=.5)
        ax[0].plot(x, b1_y * x + b0_y, color='red', linewidth=.5)
        ax[0].set_title(f'$m_s$:{math.atan(b1_s)*K:.1f}°; $m_y$:{math.atan(b1_y)*K:.1f}°',
                        fontsize=11)
        ax[0].set_xlabel(parent_name)
        ax[0].set_ylabel(f"$$ \mathrm{{{target_name}}} / \phi_{{{target_name}}} $$")


        # Represent distributions
        pd.DataFrame(s).plot(kind='density', ax=ax[1], label="shap")
        pd.DataFrame(y).plot(kind='density', ax=ax[1], label="parent")
        ax[1].legend().set_visible(False)
        ax[1].set_ylabel('')
        ax[1].set_title(f'KS({r.ks_pvalue:.2g}) - {r.ks_result}', fontsize=11)

        # Represent fitted vs. residuals
        s_resid = r.shap_model.get_influence().resid_studentized_internal
        y_resid = r.parent_model.get_influence().resid_studentized_internal
        scaler = StandardScaler()
        s_fitted_scaled = scaler.fit_transform(r.shap_model.fittedvalues.reshape(-1, 1))
        y_fitted_scaled = scaler.fit_transform(r.parent_model.fittedvalues.reshape(-1, 1))
        ax[2].scatter(s_fitted_scaled, s_resid, alpha=0.5, marker='.')
        ax[2].scatter(y_fitted_scaled, y_resid, alpha=0.5, marker='.', color='tab:orange')
        ax[2].set_title(f"Shap {shap_label}; Parent {parent_label}", fontsize=11)
        ax[2].set_xlabel(f"$$ \mathrm{{{target_name}}} /  \phi_{{{target_name}}} $$")
        ax[2].set_ylabel(f"$$ \epsilon_{{{target_name}}} / \epsilon_\phi $$")

        # Represent target vs. SHAP values
        ax[3].scatter(s, y, alpha=0.3, marker='.', color='tab:grey')
        ax[3].set_title(f"Corr: {r.shap_correlation:.2f}", fontsize=11)
        ax[3].set_xlabel(f"$$ \phi_{{{target_name}}} $$")
        ax[3].set_ylabel(f"$$ \mathrm{{{target_name}}} $$")

        for ax_idx in range(4):
            _remove_ticks_and_box(ax[ax_idx])


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=150)
    warnings.filterwarnings('ignore')

    dataset_name = 'generated_linear_10'
    data = pd.read_csv("~/phd/data/generated_linear_10_tiny.csv")
    ref_graph = graph_from_dot_file(
        "/Users/renero/phd/data/generated_linear_10_tiny.dot")
    rex = load_experiment('rex', "/Users/renero/phd/output/RC3")
    # rex = Rex(prog_bar=False, verbose=True).fit(data, ref_graph)
    rex.prog_bar = False
    rex.verbose = True
    rex.shaps = ShapEstimator(models=rex.regressor)
    rex.shaps.fit(data)
