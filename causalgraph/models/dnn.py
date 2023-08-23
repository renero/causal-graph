import types
import warnings
from typing import List, Union, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from causalgraph.common import GRAY, GREEN, RESET, tqdm_params
from causalgraph.models._columnar import ColumnsDataset
from causalgraph.models._models import MLPModel


warnings.filterwarnings("ignore")


class NNRegressor(BaseEstimator):
    """
    A class to train DFF networks for all variables in data. Each network will be trained to
    predict one of the variables in the data, using the rest as predictors plus one
    source of random noise.

    Attributes:
    -----------
        hidden_dim (int): The dimension(s) of the hidden layer(s). This value
            can be a single integer for DFF or an array with the dimension of
            each hidden layer for the MLP case.
        activation (str): The activation function to use, either 'relu' or 'selu'.
            Default is 'relu'.
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

    Examples:
    ---------
    >>> nn = NNRegressor().fit(data)
    >>> nn.predict(data)

    """

    def __init__(
            self,
            hidden_dim: Union[int, List[int]] = [75, 17],
            activation: str = 'relu',
            learning_rate: float = 0.0046,
            dropout: float = 0.001,
            batch_size: int = 44,
            num_epochs: int = 40,
            loss_fn: str = 'mse',
            device: Union[int, str] = "auto",
            test_size: float = 0.1,
            early_stop: bool = False,
            patience: int = 10,
            min_delta: float = 0.001,
            random_state: int = 1234,
            verbose: bool = False,
            prog_bar: bool = True,
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
            activation (str): The activation function to use, either 'relu' or 'selu'.
                Default is 'relu'.
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
        assert X.shape[1] == self.n_features_in_, \
            f"X has {X.shape[1]} features, expected {self.n_features_in_}"

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        prediction = pd.DataFrame(columns=self.feature_names)
        for target in list(self.X.columns):
            # Creat a data loader for the target variable. The ColumnsDataset will
            # return the target variable as the second element of the tuple, and
            # will drop the target from the features.
            loader = DataLoader(
                ColumnsDataset(target, X), batch_size=self.batch_size,
                shuffle=False)
            model = self.models.regressor[target].model
            preds = []
            for (tensor_X, _) in loader:
                tensor_X = tensor_X.to(self.device)
                y_hat = model.forward(tensor_X)
                preds.append(y_hat.detach().numpy().flatten())
            prediction[target] = preds

        return prediction.values

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
            if attr.startswith('_') or \
                attr in forbidden_attrs or \
                    type(getattr(self, attr)) == types.MethodType:
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

    def tune(
            self,
            training_data: pd.DataFrame,
            test_data: pd.DataFrame,
            study_name: str,
            min_mse: float = 0.05,
            storage: str = 'sqlite:///db.sqlite3',
            load_if_exists: bool = True,
            n_trials: int = 20):
        """
        Tune the hyperparameters of the model using Optuna.
        """
        class Objective:
            """
            A class to define the objective function for the hyperparameter optimization
            Some of the parameters for NNRegressor have been taken to default values to
            reduce the number of hyperparameters to optimize.

            Include this class in the hyperparameter optimization as follows:

            >>> study = optuna.create_study(direction='minimize',
            >>>                             study_name='study_name_here',
            >>>                             storage='sqlite:///db.sqlite3',
            >>>                             load_if_exists=True)
            >>> study.optimize(Objective(train_data, test_data), n_trials=100)

            The only dependency is you need to pass the train and test data to the class
            constructor. Tha class will build the data loaders for them from the 
            dataframes.
            """

            def __init__(self, train_data, test_data, device='cpu'):
                self.train_data = train_data
                self.test_data = test_data
                self.device = device

            def __call__(self, trial):
                self.n_layers = trial.suggest_int('n_layers', 1, 5)
                self.layers = []
                for i in range(self.n_layers):
                    self.layers.append(
                        trial.suggest_int(f'n_units_l{i}', 1, 81))
                self.activation = trial.suggest_categorical(
                    'activation', ['relu', 'selu'])
                self.learning_rate = trial.suggest_loguniform(
                    'learning_rate', 1e-5, 1e-1)
                self.dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
                self.batch_size = trial.suggest_int('batch_size', 1, 64)
                self.num_epochs = trial.suggest_int('num_epochs', 10, 100)

                self.models = NNRegressor(
                    hidden_dim=self.layers,
                    activation=self.activation,
                    learning_rate=self.learning_rate,
                    dropout=self.dropout,
                    batch_size=self.batch_size,
                    num_epochs=self.num_epochs,
                    loss_fn='mse',
                    device=self.device,
                    random_state=42,
                    verbose=False,
                    prog_bar=True,
                    silent=True)

                self.models.fit(self.train_data)

                # Now, measure the performance of the model with the test data.
                mse = []
                for target in list(self.train_data.columns):
                    model = self.models.regressor[target].model
                    loader = DataLoader(
                        ColumnsDataset(target, self.test_data),
                        batch_size=self.batch_size,
                        shuffle=False)
                    avg_loss, _, _ = self.compute_loss(model, loader)
                    mse.append(avg_loss)
                return np.median(mse)

            def compute_loss(
                    self,
                    model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    n_repeats: int = 10) -> Tuple[float, float, np.ndarray]:
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
                for _ in range(n_repeats):
                    loss = []
                    for _, (X, y) in enumerate(dataloader):
                        X = X.to(self.device)
                        y = y.to(self.device)
                        yhat = model.forward(X)
                        loss.append(model.loss_fn(yhat, y).item())
                        num_batches += 1
                    if len(mse) == 0:
                        mse = np.array(loss)
                    else:
                        mse = np.vstack((mse, [loss]))

                return np.mean(mse), np.std(mse), mse

        # Callback to stop the study if the MSE is below a threshold.
        def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            if trial.value < min_mse or study.best_value < min_mse:
                study.stop()

        if self.verbose is False:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create and run the HPO study.
        study = optuna.create_study(
            direction='minimize', study_name=study_name, storage=storage,
            load_if_exists=load_if_exists)
        study.optimize(
            Objective(training_data, test_data), n_trials=n_trials,
            show_progress_bar=self.prog_bar, callbacks=[callback])

        # Capture the best parameters and the minimum MSE obtained.
        best_trials = sorted(study.best_trials, key=lambda x: x.values[0])
        self.best_params = best_trials[0].params
        self.min_tunned_loss = best_trials[0].values[0]

        if self.verbose and not self.silent:
            print(f"Best params (min MSE:{self.min_tunned_loss:.6f}):")
            for k, v in self.best_params.items():
                print(f"\t{k:<15s}: {v}")

        return self.best_params


def main(tune: bool = False):
    warnings.filterwarnings("ignore", category=UserWarning)

    dataset_name = 'rex_generated_polynew_10'
    data = pd.read_csv(f"~/phd/data/RC3/{dataset_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # split data into train and test
    train_data = data.sample(frac=0.9, random_state=42)
    test_data = data.drop(train_data.index)

    if tune:
        #  TUNE the hyperparameters first,
        nn = NNRegressor(prog_bar=True)
        nn.tune(train_data, test_data, study_name='test4', n_trials=25)

        print(f"Best params (min MSE:{nn.min_tunned_loss:.6f}):")
        for k, v in nn.best_params.items():
            print(f"+-> {k:<13s}: {v}")

        # ...and fit the regressor with those found best parameters
        nn = NNRegressor(
            hidden_dim=[nn.best_params[f'n_units_l{i}']
                        for i in nn.best_params['n_layers']],
            activation=nn.best_params['activation'],
            learning_rate=nn.best_params['learning_rate'],
            dropout=nn.best_params['dropout'],
            batch_size=nn.best_params['batch_size'],
            num_epochs=nn.best_params['num_epochs'])
    else:
        nn = NNRegressor()

    nn.fit(data)


if __name__ == "__main__":
    main(tune=False)
