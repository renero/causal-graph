import types
import warnings
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from tqdm.auto import tqdm

from causalgraph.common import GRAY, GREEN, RESET, tqdm_params
from causalgraph.models._models import MLPModel

# import sys
# sys.path.append("../dnn")


warnings.filterwarnings("ignore")

# TODO: Make this class to inherit from a generic class for all regressors


class NNRegressor(BaseEstimator):
    """
    """

    def __init__(
            self,
            hidden_dim: Union[int, List[int]] = [36, 12],
            activation: nn.Module = nn.ReLU(),
            learning_rate: float = 0.0035,
            dropout: float = 0.065,
            batch_size: int = 45,
            num_epochs: int = 50,
            loss_fn: str = 'mse',
            device: Union[int, str] = "auto",
            test_size: float = 0.1,
            early_stop: bool = True,
            patience: int = 10,
            min_delta: float = 0.001,
            random_state: int = 1234,
            verbose: bool = False,
            prog_bar: bool = False,
            silent: bool = False):
        """
        Train DFF networks for all variables in data. Each network will be trained to
        predict one of the variables in the data, using the rest as predictors plus one
        source of random noise.

        Args:
            data (pandas.DataFrame): The dataframe with the continuous variables.
            model_type (str): The type of model to use. Either 'dff' or 'mlp'.
            hidden_dim (int): The dimension(s) of the hidden layer(s). This value 
                can be a single integer for DFF or an array with the dimension of 
                each hidden layer for the MLP case.
            learning_rate (float): The learning rate for the optimizer.
            dropout (float): The dropout rate for the dropout layer.
            batch_size (int): The batch size for the optimizer.
            num_epochs (int): The number of epochs for the optimizer.
            loss_fn (str): The loss function to use. Default is "mmd".
            device (str): The device to use. Either "cpu", "cuda", or "mps". Default
                is "auto".
            test_size (float): The proportion of the data to use for testing. Default
                is 0.1.
            seed (int): The seed for the random number generator. Default is 1234.
            early_stop (bool): Whether to use early stopping. Default is True.
            patience (int): The patience for early stopping. Default is 10.
            min_delta (float): The minimum delta for early stopping. Default is 0.001.
            prog_bar (bool): Whether to enable the progress bar. Default
                is False.

        Returns:
            dict: A dictionary with the trained DFF networks, using the name of the
                variables as the key.
        """
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss_fn = loss_fn
        self.device = device
        self.test_size = test_size
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.random_state = random_state
        self.verbose = verbose
        self.prog_bar = prog_bar
        self.silent = silent

        self.regressor = None
        self._fit_desc = "Training NNs"

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        # X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.feature_names = list(X.columns)
        self.regressor = dict()

        pbar_in = tqdm(
            total=len(self.feature_names),
            **tqdm_params(self._fit_desc, self.prog_bar, silent=self.silent))

        for target in self.feature_names:
            pbar_in.refresh()
            self.regressor[target] = MLPModel(
                target=target,
                input_size=self.n_features_in_,
                activation=self.activation,
                hidden_dim=self.hidden_dim,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                loss_fn=self.loss_fn,
                dropout=self.dropout,
                num_epochs=self.num_epochs,
                dataframe=X,
                test_size=self.test_size,
                device=self.device,
                seed=self.random_state,
                early_stop=self.early_stop,
                patience=self.patience,
                min_delta=self.min_delta,
                prog_bar=self.prog_bar)
            self.regressor[target].train()
            pbar_in.update(1)
        pbar_in.close()

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return np.ones(X.shape[0], dtype=np.int64)

    def get_input_tensors(self, target_name: str):
        """
        Returns the data used to train the model for the given target.

        Parameters
        ----------
        target_name : str
            The name of the target.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            The data used to train the model for the given target.

        Examples
        --------
        >>> nn = NNRegressor().fit(data)
        >>> X, y = nn.get_input_tensors('target1')

        """
        model = self.regressor[target_name]
        features_tensor = torch.autograd.Variable(
            model.train_loader.dataset.features)
        target_tensor = model.train_loader.dataset.target

        cols = list(self.feature_names)
        cols.remove(target_name)

        X = pd.DataFrame(features_tensor.detach().numpy(), columns=cols)
        y = pd.DataFrame(target_tensor.detach().numpy(), columns=[target_name])

        return X, y

    def __repr__(self):
        forbidden_attrs = [
            'fit', 'predict', 'score', 'get_params', 'set_params']
        ret = f"{GREEN}REX object attributes{RESET}\n"
        ret += f"{GRAY}{'-'*80}{RESET}\n"
        for attr in dir(self):
            if attr.startswith('_') or attr in forbidden_attrs or type(getattr(self, attr)) == types.MethodType:
                continue
            elif attr == "X" or attr == "y":
                if isinstance(getattr(self, attr), pd.DataFrame):
                    ret += f"{attr:25} {getattr(self, attr).shape}\n"
                    continue
            elif isinstance(getattr(self, attr), pd.DataFrame):
                ret += f"{attr:25} DataFrame {getattr(self, attr).shape}\n"
            else:
                ret += f"{attr:25} {getattr(self, attr)}\n"

        return ret


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    dataset_name = 'rex_generated_polynew_10'
    data = pd.read_csv(f"~/phd/data/RC3/{dataset_name}.csv")

    nn = NNRegressor(
        hidden_dim=[10, 5],
        activation=nn.SELU(),
        learning_rate=0.2,
        dropout=0.05,
        batch_size=32,
        num_epochs=50,
        loss_fn="mse",
        device="cpu",
        test_size=0.1,
        early_stop=False,
        patience=10,
        min_delta=0.001,
        random_state=1234,
        prog_bar=False)
    nn.fit(data)
