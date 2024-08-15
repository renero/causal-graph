"""
Permutation Importance for feature selection.
Wrapper over SKLearn's PermutationImportance and own implementation of
the vanilla version of the algorithm to run over models trained with PyTorch.

(C) J. Renero 2022, 2023

Parameters:
-----------
models: dict
    A dictionary of models, where the keys are the target variables and
    the values are the models trained to predict the target variables.
n_repeats: int
    The number of times to repeat the permutation importance algorithm.
mean_pi_percentile: float
    The percentile of the mean permutation importance to use as a threshold
    for feature selection.
random_state: int
    The random state to use for the permutation importance algorithm.
prog_bar: bool
    Whether to display a progress bar or not.
verbose: bool
    Whether to display explanations on the process or not.
silent: bool
    Whether to display anything or not.

"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from mlforge.progbar import ProgBar
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from causalgraph.common import utils
from causalgraph.common.plot import subplots
from causalgraph.explainability.hierarchies import Hierarchies
from causalgraph.independence.feature_selection import select_features
from causalgraph.models._models import MLPModel


class PermutationImportance(BaseEstimator):
    """
    Permutation Importance for feature selection.
    Wrapper over SKLearn's PermutationImportance and own implementation of
    the vanilla version of the algorithm to run over models trained with PyTorch.
    """

    device = utils.select_device("cpu")

    def __init__(
            self,
            models: dict,
            discrepancies: dict = None,
            correlation_th: float = None,
            n_repeats: int = 10,
            mean_pi_percentile: float = 0.8,
            exhaustive: bool = False,
            threshold: float = None,
            random_state: int = 42,
            prog_bar=True,
            verbose=False,
            silent=False):
        """
        Parameters:
        -----------
        models: dict
            A dictionary of models, where the keys are the target variables and
            the values are the models trained to predict the target variables.
        discrepancies: dict
            A dictionary of discrepancies for each target variable, based on SHAP
            values.
        n_repeats: int
            The number of times to repeat the permutation importance algorithm.
        mean_pi_percentile: float
            The percentile of the mean permutation importance to use as a threshold
            for feature selection.
        random_state: int
            The random state to use for the permutation importance algorithm.
        prog_bar: bool
            Whether to display a progress bar or not.
        verbose: bool
            Whether to display explanations on the process or not.
        silent: bool
            Whether to display anything or not.
        """
        super().__init__()
        self.models = models
        self.shap_discrepancies = discrepancies
        self.regressors = models.regressor
        self.correlation_th = correlation_th
        self.n_repeats = n_repeats
        self.mean_pi_percentile = mean_pi_percentile
        self.exhaustive = exhaustive
        self.threshold = threshold
        self.random_state = random_state
        self.prog_bar = prog_bar
        self.verbose = verbose
        self.silent = silent
        self.feature_names = list(self.regressors.keys())

        self.base_loss = {}
        self.base_std = {}
        self.all_pi = []
        self.pi = {}
        self.connections = {}
        self.corr_matrix = None
        self.correlated_features = None
        self.G_pi = None
        self.mean_pi_threshold = None

        self.is_fitted_ = False

        self._fit_desc = "Running Perm.Importance"
        self._pred_desc = "Predicting w perm. imp."

    def fit(self, X):
        """
        Implementation of the fit method for the PermutationImportance class.
        If the model is a PyTorch model, the fit method will compute the base loss
        for each feature. If the model is a SKLearn model, the fit method will
        compute the permutation importance for each feature.
        """
        self._obtain_correlation_info(X)

        first_key = self.feature_names[0]
        if isinstance(self.regressors[first_key], MLPModel):
            return self._fit_pytorch()
        else:
            return self._fit_sklearn(X)

    def _obtain_correlation_info(self, X):
        if self.correlation_th:
            self.corr_matrix = Hierarchies.compute_correlation_matrix(X)
            self.correlated_features = Hierarchies.compute_correlated_features(
                self.corr_matrix, self.correlation_th, self.feature_names,
                verbose=self.verbose)

    def _fit_pytorch(self):
        """
        Fit the model to compute the base loss for each feature for pyTorch models.
        """
        # pbar = tqdm(
        #     total=len(self.feature_names),
        #     **tqdm_params(self._fit_desc, self.prog_bar, silent=self.silent))
        pbar = ProgBar().start_subtask(len(self.feature_names))
        print("Computing base loss (PyTorch)") if self.verbose else None

        for feature in self.feature_names:
            # pbar.refresh()
            pbar.update_subtask(1)
            print(f"Feature: {feature} ", end="") if self.verbose else None

            regressor = self.regressors[feature]
            model = regressor.model.to(self.device)

            avg_loss, std_loss, _ = self._compute_loss_shuffling_column(
                model, regressor.train_loader)
            self.base_loss[feature] = avg_loss
            self.base_std[feature] = std_loss

            if (self.verbose) and (not self.silent):
                print(f"Base loss: {self.base_loss[feature]:.6f} ", end="")
                print(f"+/- {self.base_std[feature]:.6f}")

            pbar.update_subtask()
        #     pbar.update(1)
        # pbar.close()

        self.is_fitted_ = True

        return self

    def _fit_sklearn(self, X):
        """
        Fit the model to compute the base loss for each feature, for SKLearn models.
        """
        # pbar = tqdm(
        #     total=len(self.feature_names),
        #     **tqdm_params(self._fit_desc, self.prog_bar, silent=self.silent))
        pbar = ProgBar().start_subtask(len(self.feature_names))

        # If me must exclude features due to correlation, we must do it before
        # computing the base loss
        if self.correlation_th:
            self.corr_matrix = Hierarchies.compute_correlation_matrix(X)
            self.correlated_features = Hierarchies.compute_correlated_features(
                self.corr_matrix, self.correlation_th, self.feature_names,
                verbose=self.verbose)

        self.pi = {}
        self.all_pi = []
        X_original = X.copy()
        for target_name in self.feature_names:
            # pbar.refresh()
            pbar.update_subtask(1)
            X = X_original.copy()

            # if correlation_th is not None then, remove features that are highly
            # correlated with the target, at each step of the loop
            if self.correlation_th is not None:
                if len(self.correlated_features[target_name]) > 0:
                    X = X.drop(self.correlated_features[target_name], axis=1)
                    if self.verbose:
                        print("REMOVED CORRELATED FEATURES: ",
                              self.correlated_features[target_name])

            # print(f"Feature: {target_name} ", end="") if self.verbose else None

            regressor = self.regressors[target_name]
            y = X[target_name]
            X = X.drop(columns=[target_name])
            self.pi[target_name] = permutation_importance(
                regressor, X, y, n_repeats=10,
                random_state=self.random_state)

            if self.correlation_th is not None:
                self._add_zeroes(
                    target_name, self.correlated_features[target_name])

            self.all_pi.append(self.pi[target_name]['importances_mean'])

            # pbar.update(1)
            pbar.update_subtask()

        # pbar.close()

        self.all_pi = np.array(self.all_pi).flatten()
        self.mean_pi_threshold = np.quantile(
            self.all_pi, self.mean_pi_percentile)
        self.is_fitted_ = True

        return self

    def predict(self, X=None, root_causes=None, prior: List[List[str]] = None):
        """
        Implementation of the predict method for the PermutationImportance class.

        Parameters:
        -----------
        X: pd.DataFrame
            The data to predict the permutation importance for.

        Returns:
        --------
        G_pi: nx.DiGraph
            The DAG representing the permutation importance for the features.
        """
        if self.verbose:
            print("-----\npermutation_importance.predict()")

        self.prior = prior
        first_key = self.feature_names[0]
        if isinstance(self.regressors[first_key], MLPModel):
            return self._predict_pytorch(X, root_causes)

        #  SKLearn models don't have a predict stage for permutation importance.
        return self._predict_sklearn(X, root_causes)

    def _predict_pytorch(self, X, root_causes) -> nx.DiGraph:
        """
        Predict the permutation importance for each feature, for each target,
        under the PyTorch implementation of the algorithm.
        """
        pbar = ProgBar().start_subtask(len(self.feature_names))
        print("Computing permutation loss (PyTorch)") if self.verbose else None

        self.all_pi = []
        num_vars = len(self.feature_names)
        for target in self.feature_names:
            regressor = self.regressors[target]
            model = regressor.model
            feature_names_wo_target = [
                f for f in self.feature_names if f != target]
            candidate_causes = utils.valid_candidates_from_prior(
                self.feature_names, target, self.prior)
            if self.verbose:
                print(
                    f"Target: {target} (base loss: {self.base_loss[target]:.6f})")

            # Create the dictionary to store the permutation importance, same way
            # as the sklearn implementation
            self.pi[target] = {}
            if self.correlation_th is not None:
                num_vars = len(self.feature_names) - \
                    len(self.correlated_features[target])
                # Filter out features that are highly correlated with the target
                candidate_causes = [f for f in candidate_causes
                                    if f not in self.correlated_features[target]]

            # Compute the permutation importance for each feature
            self.pi[target]['importances_mean'], self.pi[target]['importances_std'] = \
                self._compute_perm_imp(target, regressor, model, num_vars)

            pbar.update_subtask()

            if self.correlation_th is not None:
                self._add_zeroes(target, self.correlated_features[target])

            self.all_pi.append(self.pi[target]['importances_mean'])

            self.connections[target] = select_features(
                values=self.pi[target]['importances_mean'],
                feature_names=feature_names_wo_target,
                exhaustive=self.exhaustive,
                threshold=self.mean_pi_threshold,
                verbose=self.verbose)

        self.G_pi = self._build_pi_dag(X, root_causes)
        self.GP_pi = utils.break_cycles_if_present(
            self.G_pi, self.shap_discrepancies, self.prior, verbose=self.verbose)
        return self.G_pi

    def _predict_sklearn(self, X, root_causes) -> nx.DiGraph:
        """
        Predict the permutation importance for each feature, for each target,
        under the PyTorch implementation of the algorithm.
        """
        print("Computing permutation loss (SKLearn)") if self.verbose else None

        self.connections = {}
        for target in self.feature_names:
            if self.verbose:
                print(f"Target: {target} ")
            candidate_causes = [f for f in self.feature_names if f != target]
            self.connections[target] = select_features(
                values=self.pi[target]['importances_mean'],
                feature_names=candidate_causes,
                verbose=self.verbose)

        self.G_pi = self._build_pi_dag(X, root_causes)
        return self.G_pi

    def _build_pi_dag(self, X, root_causes):
        """
        Build a DAG from the permutation importance results. This is the last stage
        of the algorithm, where we build a DAG from the permutation importance results.
        Placed in a separate method as it is shared in SKLearn and PyTorch.
        """
        self.G_pi = utils.digraph_from_connected_features(
            X, self.feature_names, self.models, self.connections, root_causes,
            reciprocity=True, anm_iterations=10, verbose=self.verbose)

        self.all_pi = np.array(self.all_pi).flatten()
        self.mean_pi_threshold = np.quantile(
            self.all_pi, self.mean_pi_percentile)

        return self.G_pi

    def _compute_perm_imp(self, target, regressor, model, num_vars):
        """
        Compute the permutation importance for each feature, for a given target
        variable.

        Parameters:
        -----------
        target: str
            The target variable to compute the permutation importance for.
        regressor: MLPModel
            The regressor to compute the permutation importance for.
        model: torch.nn.Module
            The model to compute the permutation importance for.
        num_vars: int
            The number of variables to compute the permutation importance for.

        Returns:
        --------
        importances_mean: np.ndarray
            The mean permutation importance for each feature.
        importances_std: np.ndarray
            The standard deviation of the permutation importance for each feature.
        """
        importances_mean = []
        importances_std = []
        for shuffle_col in range(num_vars-1):
            feature = regressor.columns[shuffle_col]
            print(
                f"  ↳ Feature: {feature} ", end="") if self.verbose else None

            _, _, losses = self._compute_loss_shuffling_column(
                model, regressor.train_loader, shuffle_col=shuffle_col)

            axis = 1 if self.n_repeats > 1 else 0
            perm_importances = np.mean(
                losses, axis=axis) - self.base_loss[target]

            importances_mean.append(
                np.mean(perm_importances))
            if self.n_repeats > 1:
                importances_std.append(np.std(perm_importances))
            else:
                importances_std.append(
                    np.abs(np.std(losses) - self.base_loss[target]))

            if self.verbose:
                print(
                    f"Perm.imp.: {importances_mean[-1]:.6f} "
                    f"+/- {importances_std[-1]:.6f}")

        return np.array(importances_mean), np.array(importances_std)

    def _add_zeroes(self, target, correlated_features):
        """
        Add zeroes to the mean perm imp. values for correlated features.
        """
        features = [f for f in self.feature_names if f != target]
        for correlated_feature in correlated_features:
            correlated_feature_position = features.index(correlated_feature)
            self.pi[target]['importances_mean'] = np.insert(
                self.pi[target]['importances_mean'], correlated_feature_position, 0.)
            self.pi[target]['importances_std'] = np.insert(
                self.pi[target]['importances_std'], correlated_feature_position, 0.)

    def fit_predict(self, X, root_causes):
        self._obtain_correlation_info(X)

        first_key = self.feature_names[0]
        if isinstance(self.regressors[first_key], MLPModel):
            return self._fit_predict_pytorch(X, root_causes)
        else:
            return self._fit_predict_sklearn(X)

    def _fit_predict_pytorch(self, X, root_causes):
        self._fit_pytorch()
        return self._predict_pytorch(X, root_causes)

    def _fit_predict_sklearn(self, X):
        self._fit_sklearn(X)
        return self.pi

    def _compute_loss_shuffling_column(
            self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            shuffle_col: int = -1) -> Tuple[float, float, np.ndarray]:
        """
        Computes the average MSE loss for a given model and dataloader.

        Parameters:
        -----------
        model: torch.nn.Module
            The model to compute the loss for.
        dataloader: torch.utils.data.DataLoader
            The dataloader to use for computing the loss.
        shuffle: int
            If > 0, the column of the input data to shuffle.

        Returns:
        --------
        avg_loss: float
            The average MSE loss.
        std_loss: float
            The standard deviation of the MSE loss.
        losses: np.ndarray
            The MSE loss for each batch.
        """
        mse = np.array([])
        num_batches = 0
        print(
            f"(Repeats: {self.n_repeats}) ", end="") if self.verbose else None
        for _ in range(self.n_repeats):
            loss = []
            # Loop over all batches in train loader
            for _, (X, y) in enumerate(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)

                # Shuffle data if specified
                if shuffle_col >= 0:
                    X = self._shuffle_2Dtensor_column(X, shuffle_col)

                # compute MSE loss for each batch
                yhat = model.forward(X)
                loss.append(model.loss_fn(yhat, y).item())
                num_batches += 1

            if len(mse) == 0:
                mse = np.array(loss)
            else:
                mse = np.vstack((mse, [loss]))

        return np.mean(mse), np.std(mse), mse

    def _shuffle_2Dtensor_column(
            self,
            tensor: torch.Tensor,
            column: int) -> torch.Tensor:
        """
        Shuffles the column of a 2D tensor.

        Parameters:
        -----------
        tensor: torch.Tensor
            The tensor to shuffle.
        column: int
            The column to shuffle.

        Returns:
        --------
        shuffled_tensor: torch.Tensor
            The shuffled tensor.
        """
        assert column < tensor.shape[1], "Column index out of bounds"
        assert len(tensor.shape) == 2, "Tensor must be 2D"
        num_rows, num_columns = tensor.shape
        idx = torch.randperm(tensor.shape[0], device=self.device)
        column_reshuffled = torch.reshape(tensor[idx, column], (num_rows, 1))
        if column == 0:
            return torch.cat((column_reshuffled, tensor[:, 1:]), 1)
        else:
            return torch.cat((tensor[:, 0:column], column_reshuffled, tensor[:, column+1:]), 1)

    def plot(self, **kwargs):
        """
        Plot the permutation importance for each feature, by calling the internal
        _plot_perm_imp method.

        Parameters:
        -----------
        kwargs: dict
            Keyword arguments to pass to the _plot_perm_imp method.
            Examples:
                - figsize: tuple

        Returns:
        --------
        fig: matplotlib.figure.Figure
            The figure containing the plot.
        """
        assert self.is_fitted_, "Model not fitted yet"
        plot_args = [(target_name) for target_name in self.feature_names]
        return subplots(self._plot_perm_imp, *plot_args, **kwargs)

    def _plot_perm_imp(self, target, ax, **kwargs):
        """
        Plot the permutation importance for a given target variable.

        Parameters:
        -----------
        target: str
            The target variable to plot the permutation importance for.
        ax: matplotlib.axes.Axes
            The axes to plot the permutation importance on.
        kwargs: dict
            Keyword arguments to pass to the barh method of the axes.
            Examples:
                - figsize: tuple

        Returns:
        --------
        fig: matplotlib.figure.Figure
            The figure containing the subplot.

        """
        feature_names = [f for f in self.feature_names if f != target]
        figsize_ = kwargs.get('figsize', (6, 3))
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize_)

        sorted_idx = self.pi[target]['importances_mean'].argsort()
        ax.barh(
            np.array(feature_names)[sorted_idx.astype(int)],
            self.pi[target]['importances_mean'][sorted_idx],
            xerr=self.pi[target]['importances_std'][sorted_idx],
            align='center', alpha=0.5)

        xlims = ax.get_xlim()
        if xlims[1] < self.mean_pi_threshold:
            ax.set_xlim(right=self.mean_pi_threshold +
                        ((xlims[1] - xlims[0])/20))
        ax.axvline(
            x=self.mean_pi_threshold, color='red', linestyle='--', linewidth=0.5)

        ax.set_title(
            f"Perm.Imp.{target}: {','.join(self.connections[target])}")
        fig = ax.figure if fig is None else fig

        return fig


if __name__ == "__main__":
    path = "/Users/renero/phd/data/RC3/"
    output_path = "/Users/renero/phd/output/RC3/"
    experiment_name = 'rex_generated_gp_mix_1'

    ref_graph = utils.graph_from_dot_file(f"{path}{experiment_name}.dot")
    data = pd.read_csv(f"{path}{experiment_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    rex = utils.load_experiment(f"{experiment_name}_gbt", output_path)
    print(f"Loaded experiment {experiment_name}")

    #  Run the permutation importance algorithm
    pi = PermutationImportance(
        rex.models, n_repeats=10, prog_bar=False, verbose=True)
    pi.fit(data)
    pi.predict(data, rex.root_causes)
    pi.plot(fig_size=(7, 5))
    plt.show()
