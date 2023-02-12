from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from tqdm.auto import tqdm

from causalgraph.dnn.models import DFFModel, MLPModel


class NNRegressor(BaseEstimator):
    """
    """
    def __init__(
        self,
        model_type: str,
        hidden_dim: Union[int, List[int]],
        learning_rate: float,
        dropout: float,
        batch_size: int,
        num_epochs: int,
        loss_fn: str,
        gpus: int,
        test_size: float,
        early_stop: bool,
        patience: int,
        min_delta: float,
        random_state: int = 1234,
        enable_progress_bar: bool = False):
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
            gpus (int): The number of GPUs to use. Default is 0.
            test_size (float): The proportion of the data to use for testing. Default
                is 0.1.
            seed (int): The seed for the random number generator. Default is 1234.
            early_stop (bool): Whether to use early stopping. Default is True.
            patience (int): The patience for early stopping. Default is 10.
            min_delta (float): The minimum delta for early stopping. Default is 0.001.
            enable_progress_bar (bool): Whether to enable the progress bar. Default
                is False.

        Returns:
            dict: A dictionary with the trained DFF networks, using the name of the
                variables as the key.
        """
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss_fn = loss_fn
        self.gpus = gpus
        self.test_size = test_size
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.random_state = random_state
        self.enable_progress_bar = enable_progress_bar

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
        #X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.feature_names = list(X.columns)

        model = DFFModel if self.model_type == "dff" else MLPModel
        self.nets = dict()
        pbar = tqdm(total=len(self.feature_names), desc="Training NNs", leave=False)
        for target in self.feature_names:
            pbar.refresh()
            net = model(
                target=target,
                input_size=self.n_features_in_,
                hidden_dim=self.hidden_dim,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                loss_fn=self.loss_fn,
                dropout=self.dropout,
                num_epochs=self.num_epochs,
                dataframe=X,
                test_size=self.test_size,
                gpus=self.gpus,
                seed=self.random_state,
                early_stop=self.early_stop,
                patience=self.patience,
                min_delta=self.min_delta,
                enable_progress_bar=self.enable_progress_bar)
            net.train()
            self.nets[target] = net
            pbar.update(1)
        pbar.close()

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


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    dataset_name = 'generated_linear_10'
    data = pd.read_csv("~/phd/data/generated_linear_10.csv")

    nn = NNRegressor(
        model_type="mlp",
        hidden_dim=[40, 60, 40],
        learning_rate=0.1,
        dropout=0.05,
        batch_size=16,
        num_epochs=50,
        loss_fn="mse",
        gpus=0,
        test_size=0.1,
        early_stop=False,
        patience=10,
        min_delta=0.001,
        random_state=1234,
        enable_progress_bar=False)
    nn.fit(data)
