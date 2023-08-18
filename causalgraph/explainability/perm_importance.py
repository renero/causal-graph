from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from tqdm.auto import tqdm

from causalgraph.common import tqdm_params, utils
from causalgraph.common.plots import subplots
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
            n_repeats: int = 10,
            mean_pi_percentile: float = 0.8,
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
        self.regressors = models.regressor
        self.n_repeats = n_repeats
        self.mean_pi_percentile = mean_pi_percentile
        self.random_state = random_state
        self.prog_bar = prog_bar
        self.verbose = verbose
        self.silent = silent
        self.feature_names_ = list(self.regressors.keys())

        self.is_fitted_ = False

        self._fit_desc = "Running Perm.Importance"
        self._pred_desc = "Predicting w perm. imp."

    def fit(self, X, y=None):
        """
        Implementation of the fit method for the PermutationImportance class.
        If the model is a PyTorch model, the fit method will compute the base loss
        for each feature. If the model is a SKLearn model, the fit method will
        compute the permutation importance for each feature.
        """
        first_key = self.feature_names_[0]
        if isinstance(self.regressors[first_key], MLPModel):
            return self._fit_pytorch()
        else:
            return self._fit_sklearn(X)

    def _fit_pytorch(self):
        """
        Fit the model to compute the base loss for each feature for pyTorch models.
        """
        pbar = tqdm(
            total=len(self.feature_names_),
            **tqdm_params(self._fit_desc, self.prog_bar, silent=self.silent))
        print("Computing base loss (PyTorch)") if self.verbose else None

        self.base_loss = dict()
        self.base_std = dict()
        for feature in self.feature_names_:
            print(f"Feature: {feature} ", end="") if self.verbose else None
            pbar.refresh()
            regressor = self.regressors[feature]
            model = regressor.model.to(self.device)

            avg_loss, std_loss, _ = self._compute_loss(
                model, regressor.train_loader)
            self.base_loss[feature] = avg_loss
            self.base_std[feature] = std_loss

            if (self.verbose) and (not self.silent):
                print(f"Base loss: {self.base_loss[feature]:.6f} ", end="")
                print(f"+/- {self.base_std[feature]:.6f}")
            pbar.update(1)
        pbar.close()

        self.is_fitted_ = True

        return self

    def _fit_sklearn(self, X):
        """
        Fit the model to compute the base loss for each feature, for SKLearn models.
        """
        pbar = tqdm(
            total=len(self.feature_names_),
            **tqdm_params(self._fit_desc, self.prog_bar, silent=self.silent))

        self.pi = {}
        self.all_pi = []
        self.X = X.copy()
        for target in self.feature_names_:
            pbar.refresh()
            regressor = self.regressors[target]
            X = self.X.drop(columns=[target])
            y = self.X[target]
            self.pi[target] = permutation_importance(
                regressor, X, y, n_repeats=10,
                random_state=self.random_state)
            self.all_pi.append(self.pi[target]['importances_mean'])
            pbar.update(1)
        pbar.close()
        self.all_pi = np.array(self.all_pi).flatten()
        self.mean_pi_threshold = np.quantile(
            self.all_pi, self.mean_pi_percentile)
        self.is_fitted_ = True

        return self

    def predict(self, X=None, y=None):
        first_key = self.feature_names_[0]
        if isinstance(self.regressors[first_key], MLPModel):
            return self._predict_pytorch()
        #  SKLearn models don't have a predict stage for permutation importance.
        return self.pi

    def _predict_pytorch(self):
        """
        Predict the permutation importance for each feature, for each target, 
        under the PyTorch implementation of the algorithm.
        """
        pbar = tqdm(
            total=len(self.feature_names_),
            **tqdm_params(self._fit_desc, self.prog_bar, silent=self.silent))
        print("Computing permutation loss (PyTorch)") if self.verbose else None

        self.all_pi = []
        self.pi = dict()
        num_vars = len(self.feature_names_)
        for target in self.feature_names_:
            pbar.refresh()
            regressor = self.regressors[target]
            model = regressor.model
            print(f"Target: {target} ", end="") if self.verbose else None
            print(
                f" (base loss: {self.base_loss[target]:.6f})") if self.verbose else None

            # Create the dictionary to store the permutation importance, same way
            # as the sklearn implementation
            self.pi[target] = dict()
            self.pi[target]['importances_mean'] = []
            self.pi[target]['importances_std'] = []

            # Compute the permutation importance for each feature
            for shuffle_col in range(num_vars-1):
                feature = regressor.columns[shuffle_col]
                print(
                    f"+-> Feature: {feature} ", end="") if self.verbose else None

                _, _, losses = self._compute_loss(
                    model, regressor.train_loader, shuffle=shuffle_col)

                perm_importances = np.mean(
                    losses, axis=1) - self.base_loss[target]

                self.pi[target]['importances_mean'].append(
                    np.mean(perm_importances))
                self.pi[target]['importances_std'].append(
                    np.std(perm_importances))

                if self.verbose:
                    print(
                        f"Perm.imp.: {self.pi[target]['importances_mean'][-1]:.6f}", end="")
                    print(f"+/- {self.pi[target]['importances_std'][-1]:.6f}")

            pbar.update(1)

            self.pi[target]['importances_mean'] = np.array(
                self.pi[target]['importances_mean'])
            self.pi[target]['importances_std'] = np.array(
                self.pi[target]['importances_std'])
            self.all_pi.append(self.pi[target]['importances_mean'])

        pbar.close()

        self.all_pi = np.array(self.all_pi).flatten()
        self.mean_pi_threshold = np.quantile(
            self.all_pi, self.mean_pi_percentile)

        return self.pi

    def fit_predict(self, X, y=None):
        first_key = self.feature_names_[0]
        if isinstance(self.regressors[first_key], MLPModel):
            return self._fit_predict_pytorch()
        else:
            return self._fit_predict_sklearn(X)

    def _fit_predict_pytorch(self):
        self._fit_pytorch()
        return self._predict_pytorch()

    def _fit_predict_sklearn(self, X):
        self._fit_sklearn(X)
        return self.pi

    def _compute_loss(
            self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            shuffle: int = -1) -> Tuple[float, float, np.ndarray]:
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
                # Shuffle data if specified
                X = X.to(self.device)
                y = y.to(self.device)
                if shuffle >= 0:
                    X = self._shuffle_2Dtensor_column(X, shuffle)
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
        plot_args = [(target_name) for target_name in self.feature_names_]
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
        feature_names = [f for f in self.feature_names_ if f != target]
        figsize_ = kwargs.get('figsize', (6, 3))
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize_)

        sorted_idx = self.pi[target]['importances_mean'].argsort()
        ax.barh(np.array(feature_names)[sorted_idx.astype(int)],
                self.pi[target]['importances_mean'][sorted_idx],
                xerr=self.pi[target]['importances_std'][sorted_idx],
                align='center', alpha=0.5)

        if self.mean_pi_threshold > 0.0 and \
                self.mean_pi_threshold < np.max(self.pi[target]['importances_mean']):
            ax.axvline(
                x=self.mean_pi_threshold, color='red', linestyle='--', linewidth=0.5)

        ax.set_title(f"Perm.Imp. {target}")
        fig = ax.figure if fig is None else fig

        return fig


if __name__ == "__main__":
    path = "/Users/renero/phd/data/RC3/"
    output_path = "/Users/renero/phd/output/RC3/"
    experiment_name = 'rex_generated_polynew_10'

    ref_graph = utils.graph_from_dot_file(f"{path}{experiment_name}.dot")
    data = pd.read_csv(f"{path}{experiment_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    rex = utils.load_experiment(f"{experiment_name}_mps", output_path)
    print(f"Loaded experiment {experiment_name}_mps")

    #  Run the permutation importance algorithm
    pi = PermutationImportance(
        rex.models, n_repeats=10, prog_bar=False, verbose=False)
    pi.fit(data)
    pi.predict()
