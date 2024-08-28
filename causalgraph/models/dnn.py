"""
A class to train DFF networks for all variables in data. Each network will be trained to
predict one of the variables in the data, using the rest as predictors plus one
source of random noise.

(C) 2022,2023,2024, J. Renero
"""


# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches
# pylint: disable=W0102:dangerous-default-value

import types
import warnings
from typing import List, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader
from mlforge.progbar import ProgBar

from causalgraph.common import GRAY, GREEN, RESET
from causalgraph.explainability.hierarchies import Hierarchies
from causalgraph.models._columnar import ColumnsDataset
from causalgraph.models._models import MLPModel
from causalgraph.common import utils

warnings.filterwarnings("ignore")


class NNRegressor(BaseEstimator):
    """
    A class to train DFF networks for all variables in data. Each network will be
    trained to predict one of the variables in the data, using the rest as predictors
    plus one source of random noise.

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
            device: Union[int, str] = "cpu",
            test_size: float = 0.1,
            early_stop: bool = False,
            patience: int = 10,
            min_delta: float = 0.001,
            correlation_th: float = None,
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
        self.correlation_th = correlation_th
        self.random_state = random_state
        self.verbose = verbose
        self.prog_bar = prog_bar
        self.silent = silent

        self.regressor = None
        self._fit_desc = "Training NNs"

    def fit(self, X):
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
        self.feature_names = utils.get_feature_names(X)
        self.feature_types = utils.get_feature_types(X)
        self.regressor = {}

        if self.correlation_th:
            self.corr_matrix = Hierarchies.compute_correlation_matrix(X)
            self.correlated_features = Hierarchies.compute_correlated_features(
                self.corr_matrix, self.correlation_th, self.feature_names,
                verbose=self.verbose)
            X_original = X.copy()

        pbar = ProgBar().start_subtask("DNN_fit", len(self.feature_names))

        for target_idx, target_name in enumerate(self.feature_names):
            # if correlation_th is not None then, remove features that are highly
            # correlated with the target, at each step of the loop
            if self.correlation_th is not None:
                X = X_original.copy()
                if len(self.correlated_features[target_name]) > 0:
                    X = X.drop(self.correlated_features[target_name], axis=1)
                    if self.verbose:
                        print("REMOVED CORRELATED FEATURES: ",
                              self.correlated_features[target_name])

            # Determine Loss function based on the type of target variable
            if self.feature_types[target_name] == 'categorical':
                loss_fn = 'crossentropy'
            elif self.feature_types[target_name] == 'binary':
                loss_fn = 'binary_crossentropy'
            else:
                loss_fn = self.loss_fn

            self.regressor[target_name] = MLPModel(
                target=target_name,
                input_size=X.shape[1],
                activation=self.activation,
                hidden_dim=self.hidden_dim,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                loss_fn=loss_fn,
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
            self.regressor[target_name].train()

            pbar.update_subtask("DNN_fit", target_idx+1)

        pbar.remove("DNN_fit")
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the values for each target variable.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to make predictions on.

        Returns
        -------
        np.ndarray
            The predictions for each target variable.
        """
        assert X.shape[1] == self.n_features_in_, \
            f"X has {X.shape[1]} features, expected {self.n_features_in_}"

        # X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        if self.correlation_th is not None:
            X_original = X.copy()

        prediction = pd.DataFrame(columns=self.feature_names)
        for target in self.feature_names:
            # Remove those features from the loader that are highly correlated to target
            if self.correlation_th is not None:
                X = X_original.drop(self.correlated_features[target], axis=1)

            # Creat a data loader for the target variable. The ColumnsDataset will
            # return the target variable as the second element of the tuple, and
            # will drop the target from the features.
            loader = DataLoader(
                ColumnsDataset(target, X), batch_size=self.batch_size,
                shuffle=False)
            model = self.regressor[target].model

            # Obtain the predictions for the target variable
            preds = np.empty((0,), dtype=np.float16)
            for (tensor_X, _) in loader:
                tensor_X = tensor_X.to(self.device)
                y_hat = model.forward(tensor_X)
                preds = np.append(preds, y_hat.detach().numpy().flatten())
            prediction[target] = preds


        # Concatenate the numpy array for all the batchs
        np_preds = prediction.values
        final_preds = []
        if np_preds.ndim > 1 and np_preds.shape[0] > 1:
            for i in range(len(self.feature_names)):
                column = np_preds[:, i]
                if column.ndim == 1:
                    final_preds.append(column)
                else:
                    final_preds.append(np.concatenate(column))
            final_preds = np.array(final_preds)
        else:
            final_preds = np_preds

        # If final_preds is still 1D, reshape it to 2D
        if final_preds.ndim == 1:
            final_preds = final_preds.reshape(1, -1)

        if len(final_preds) == 0:
            final_preds = np_preds

        return np.array(final_preds)

    def score(self, X):
        """
        Scores the model using the loss function. It returns the list of losses
        for each target variable.
        """
        assert X.shape[1] == self.n_features_in_, \
            f"X has {X.shape[1]} features, expected {self.n_features_in_}"
        check_is_fitted(self, 'is_fitted_')

        # Call the class method to predict the values for each target variable
        y_hat = self.predict(X)

        # Handle the case where the prediction returned by the model is not a
        # numpy array but a numpy object type
        if isinstance(y_hat, np.ndarray) and y_hat.dtype == np.object_:
            y_hat = np.vstack(y_hat[:, :].flatten()).astype('float')
            y_hat = torch.Tensor(y_hat)
        scores = []
        for i, target in enumerate(self.feature_names):
            y_preds = torch.Tensor(y_hat[i])
            y = torch.Tensor(X[target].values)
            scores.append(self.regressor[target].model.loss_fn(y_preds, y))

        self.scoring = np.array(scores)
        return self.scoring

    def __repr__(self):
        forbidden_attrs = [
            'fit', 'predict', 'score', 'get_params', 'set_params']
        ret = f"REX object attributes\n"
        ret += f"{'-'*80}\n"
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
            study_name: str = None,
            min_loss: float = 0.05,
            storage: str = 'sqlite:///rex_tuning.db',
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

                self.n_layers = None
                self.activation = None
                self.learning_rate = None
                self.dropout = None
                self.batch_size = None
                self.num_epochs = None
                self.models = None

            def __call__(self, trial):
                self.n_layers = trial.suggest_int('n_layers', 1, 6)
                self.layers = []
                for i in range(self.n_layers):
                    self.layers.append(
                        trial.suggest_int(f'n_units_l{i}', 1, 182))
                self.activation = trial.suggest_categorical(
                    'activation', ['relu', 'selu', 'linear'])
                self.learning_rate = trial.suggest_loguniform(
                    'learning_rate', 1e-5, 1e-1)
                self.dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
                self.batch_size = trial.suggest_int('batch_size', 8, 128)
                self.num_epochs = trial.suggest_int('num_epochs', 10, 80)

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
                loss = []
                for target in list(self.train_data.columns):
                    model = self.models.regressor[target].model
                    loader = DataLoader(
                        ColumnsDataset(target, self.test_data),
                        batch_size=self.batch_size,
                        shuffle=False)
                    avg_loss, _, _ = self.compute_loss(model, loader)
                    loss.append(avg_loss)
                return np.median(loss)

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
            if trial.value < min_loss or study.best_value < min_loss:
                study.stop()

        if self.verbose is False:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create and run the HPO study.
        study = optuna.create_study(
            direction='minimize', study_name=study_name, storage=storage,
            load_if_exists=load_if_exists)
        study.optimize(
            Objective(training_data, test_data), n_trials=n_trials,
            show_progress_bar=(self.prog_bar & (not self.silent)), callbacks=[callback])

        # Capture the best parameters and the minimum loss.
        best_trials = sorted(study.best_trials, key=lambda x: x.values[0])
        self.best_params = best_trials[0].params
        self.min_tunned_loss = best_trials[0].values[0]

        if self.verbose and not self.silent:
            print(f"Best params (min loss:{self.min_tunned_loss:.6f}):")
            for k, v in self.best_params.items():
                print(f"\t{k:<15s}: {v}")

        regressor_args = {
            'hidden_dim': [self.best_params[f'n_units_l{i}']
                           for i in range(self.best_params['n_layers'])],
            'activation': self.best_params['activation'],
            'learning_rate': self.best_params['learning_rate'],
            'dropout': self.best_params['dropout'],
            'batch_size': self.best_params['batch_size'],
            'num_epochs': self.best_params['num_epochs']}

        return regressor_args

    def tune_fit(
            self,
            X: pd.DataFrame,
            hpo_study_name: str = None,
            hpo_min_loss: float = 0.05,
            hpo_storage: str = 'sqlite:///rex_tuning.db',
            hpo_load_if_exists: bool = True,
            hpo_n_trials: int = 20):
        """
        Tune the hyperparameters of the model using Optuna, and the fit the model
        with the best parameters.
        """
        # split X into train and test
        train_data = X.sample(frac=0.9, random_state=self.random_state)
        test_data = X.drop(train_data.index)

        # tune the model
        regressor_args = self.tune(
            train_data, test_data, n_trials=hpo_n_trials, study_name=hpo_study_name,
            min_loss=hpo_min_loss, storage=hpo_storage,
            load_if_exists=hpo_load_if_exists)

        if self.verbose and not self.silent:
            print(f"Best params (min loss:{self.min_tunned_loss:.6f}):")
            for k, v in regressor_args.items():
                print(f"\t{k:<15s}: {v}")

        # Set the object parameters to the best parameters found.
        for k, v in regressor_args.items():
            setattr(self, k, v)

        # Fit the model with the best parameters.
        self.fit(X)


#
# Main function
#

def custom_main(score: bool = False, tune: bool = False):
    import os
    from causalgraph.common import utils
    path = "/Users/renero/phd/data/RC4/risks"
    output_path = "/Users/renero/phd/output/RC4/"
    experiment_name = 'transformed_data'

    # ref_graph = utils.graph_from_dot_file(f"{path}{experiment_name}.dot")

    data = pd.read_csv(f"{os.path.join(path, experiment_name)}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Split the dataframe into train and test
    train = data.sample(frac=0.9, random_state=42)
    test = data.drop(train.index)

    if score:
        rex = utils.load_experiment(f"{experiment_name}", output_path)
        rex.is_fitted_ = True
        print(f"Loaded experiment {experiment_name}")
        rex.models.score(test)
    elif tune:
        nn = NNRegressor()
        nn.tune_fit(data, hpo_n_trials=10)


if __name__ == "__main__":
    custom_main(tune=True)
