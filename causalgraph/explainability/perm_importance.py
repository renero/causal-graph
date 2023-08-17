from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
import torch

from tqdm.auto import tqdm

from causalgraph.models._models import MLPModel
from causalgraph.common.plots import subplots
from causalgraph.common import tqdm_params, utils


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

        self._fit_desc = f"Running Perm.Importance"
        self._pred_desc = "Predicting w perm. imp."

    def fit(self, X, y=None):
        first_key = self.feature_names_[0]
        if isinstance(self.regressors[first_key], MLPModel):
            return self._fit_pytorch()
        else:
            return self._fit_sklearn(X)

    def _fit_sklearn(self, X):
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

    def _fit_pytorch(self):
        pbar = tqdm(
            total=len(self.feature_names_),
            **tqdm_params(self._fit_desc, self.prog_bar, silent=self.silent))

        self.base_loss = dict()
        self.base_std = dict()
        for feature in self.feature_names_:
            pbar.refresh()
            regressor = self.regressors[feature]
            model = regressor.model.to(self.device)

            self.base_loss[feature], self.base_std[feature] = self._compute_loss(
                model, regressor.train_loader)
            if (self.verbose) and (not self.silent):
                print(
                    f"""Base loss: {self.base_loss[feature]:.4f} 
                    +/- {self.base_std[feature]:.4f}""")
            pbar.update(1)
        pbar.close()
        self.is_fitted_ = True

        return self

    def predict(self, X=None, y=None):
        first_key = self.feature_names_[0]
        if isinstance(self.regressors[first_key], MLPModel):
            return self._predict_pytorch()
        else:
            return self.pi

    def _predict_pytorch(self):
        pbar = tqdm(
            total=len(self.feature_names_),
            **tqdm_params(self._fit_desc, self.prog_bar, silent=self.silent))

        self.all_pi = []
        self.pi = dict()
        num_vars = len(self.feature_names_)
        for target in self.feature_names_:
            pbar.refresh()
            self.regressor = self.regressors[target]
            self.model = self.regressor.model

            self.pi[target] = dict()
            self.pi[target] = dict()
            self.pi[target]['importances_mean'] = []
            self.pi[target]['importances_std'] = []
            for shuffle_col in range(num_vars-1):
                var_name = self.regressor.columns[shuffle_col]
                perm_loss, perm_std = self._compute_loss(
                    self.model,
                    self.regressor.train_loader,
                    repeats=self.n_repeats,
                    shuffle=shuffle_col)
                perm_importance = perm_loss / \
                    (self.base_loss[target] * self.n_repeats)
                perm_std = perm_std / (self.base_loss[target] * self.n_repeats)
                self.pi[target]['importances_mean'].append(perm_importance)
                self.pi[target]['importances_std'].append(perm_std)
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
            repeats: int = 10,
            shuffle: int = -1) -> Tuple[float, float]:
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
        """
        mse = []
        num_batches = 0
        for _ in range(repeats):
            # Loop over all batches in train loader
            for _, (X, y) in enumerate(dataloader):
                # Shuffle data if specified
                X = X.to(self.device)
                y = y.to(self.device)
                if shuffle > 0:
                    X = self._shuffle_2Dtensor_column(X, shuffle)
                # compute MSE loss for each batch
                yhat = model.forward(X)
                mse.append(model.loss_fn(yhat, y).item())
                num_batches += 1
        # Compute average/std. MSE loss
        mse = np.array(mse)

        return np.mean(mse), np.std(mse)

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
        assert self.is_fitted_, "Model not fitted yet"
        plot_args = [(target_name) for target_name in self.feature_names_]
        return subplots(self._plot_perm_imp, *plot_args, **kwargs)

    def _plot_perm_imp(self, target, ax, **kwargs):
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
