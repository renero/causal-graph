"""
This module contains the GBTRegressor class, which is a wrapper around the
GradientBoostingRegressor class from the scikit-learn library. The class
implements the fit, predict, and score methods to fit a separate model for
each feature in the dataframe, and predict and score the model for each feature
in the dataframe.

The class also implements a tune method to tune the hyperparameters of the model
using Optuna. The tune method uses the Objective class to define the objective
function for the hyperparameter optimization. The Objective class is a nested
class within the GBTRegressor class, and it defines the objective function for
the hyperparameter optimization. The class is designed to be used with the
Optuna library.

The module also contains a main function that can be used to run the GBTRegressor
class with the tune method. The main function takes the name of the experiment
as an argument, and loads the data and the reference graph for the experiment.
The main function then splits the data into train and test, and runs the tune
method to tune the hyperparameters of the model. The main function can be used
to run the GBTRegressor class with the tune method for any experiment.

The module can be run as a script to run the main function with the tune method
for a specific experiment. The experiment name is passed as an argument to the
script, and the main function is called with the experiment name as an argument.
The script can be used to run the GBTRegressor class with the tune method for
any experiment.

Example:

    $ python gbt.py rex_generated_linear_6

This will run the GBTRegressor class with the tune method for the experiment
'rex_generated_linear_6'.

The module can also be imported and used in other modules or scripts to run the
GBTRegressor class with the tune method for any experiment.

Example:

    from causalgraph.models.gbt import custom_main

    custom_main("rex_generated_linear_6")

"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches
# pylint: disable=W0102:dangerous-default-value


import inspect

import numpy as np
import optuna  # type: ignore
import pandas as pd
from mlforge.progbar import ProgBar  # type: ignore
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from ..common import DEFAULT_HPO_TRIALS, utils
from ..explainability.hierarchies import Hierarchies


class GBTRegressor(GradientBoostingRegressor):

    random_state = 42

    def __init__(
            self,
            loss='squared_error',
            learning_rate=0.1,
            n_estimators=100,
            subsample=1.0,
            criterion='friedman_mse',
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_depth=3,
            min_impurity_decrease=0.0,
            init=None,
            random_state=42,
            max_features=None,
            # alpha=0.9,
            max_leaf_nodes=None,
            warm_start=False,
            validation_fraction=0.1,
            n_iter_no_change=None,
            tol=0.0001,
            ccp_alpha=0.0,
            correlation_th: float = None,
            verbose=False,
            silent=False,
            prog_bar=True,
            optuna_prog_bar=False):

        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        # self.alpha = alpha
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        self.correlation_th = correlation_th

        self.verbose = verbose
        self.silent = silent
        self.prog_bar = prog_bar
        self.optuna_prog_bar = optuna_prog_bar
        self.regressor = None
        self._estimator_name = 'gbt'
        self._estimator_class = GradientBoostingRegressor
        self._fit_desc = "Training GBTs"

    def fit(self, X):
        """
        Call the fit method of the parent class with every feature from the "X"
        dataframe as a target variable. This will fit a separate model for each
        feature in the dataframe.
        """
        self.n_features_in_ = X.shape[1]
        self.feature_names = utils.get_feature_names(X)
        self.feature_types = utils.get_feature_types(X)
        X = utils.cast_categoricals_to_int(X)
        self.regressor = dict()

        if self.correlation_th:
            self.corr_matrix = Hierarchies.compute_correlation_matrix(X)
            self.correlated_features = Hierarchies.compute_correlated_features(
                self.corr_matrix, self.correlation_th, self.feature_names,
                verbose=self.verbose)
            X_original = X.copy()

        # Who is calling me?
        try:
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            caller_name = calframe[1][3]
            if caller_name == "__call__":
                caller_name = "HPO"
        except Exception:  # pylint: disable=broad-except
            caller_name = "unknown"

        if self.prog_bar and not self.verbose:
            pbar_name = f"({caller_name}) GBT_fit"
            pbar = ProgBar().start_subtask(pbar_name, len(self.feature_names))
        else:
            pbar = None

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

            if self.feature_types[target_name] == 'categorical' or \
                    self.feature_types[target_name] == 'binary':
                self.loss = 'log_loss'
                gbt_model = GradientBoostingClassifier
            else:
                self.loss = 'squared_error'
                gbt_model = GradientBoostingRegressor

            self.regressor[target_name] = gbt_model(
                loss=self.loss,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_depth=self.max_depth,
                min_impurity_decrease=self.min_impurity_decrease,
                init=self.init,
                random_state=self.random_state,
                max_features=self.max_features,
                # alpha=self.alpha,
                verbose=False,
                max_leaf_nodes=self.max_leaf_nodes,
                warm_start=self.warm_start,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                ccp_alpha=self.ccp_alpha
            )

            self.regressor[target_name].fit(
                X.drop(target_name, axis=1), X[target_name])

            pbar.update_subtask(pbar_name, target_idx+1) if pbar else None

        pbar.remove(pbar_name) if pbar else None
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Call the predict method of the parent class with every feature from the "X"
        dataframe as a target variable. This will predict a separate value for each
        feature in the dataframe.
        """
        if not self.is_fitted_:
            raise ValueError(
                f"This {self.__class__.__name__} instance is not fitted yet."
                f"Call 'fit' with appropriate arguments before using this method.")
        y_pred = list()

        if self.correlation_th is not None:
            X_original = X.copy()

        for target_name in self.feature_names:
            if self.correlation_th is not None:
                X = X_original.drop(
                    self.correlated_features[target_name], axis=1)

            y_pred.append(
                self.regressor[target_name].predict(X.drop(target_name, axis=1)))

        return np.array(y_pred)

    def score(self, X):
        """
        Call the score method of the parent class with every feature from the "X"
        dataframe as a target variable. This will score a separate model for each
        feature in the dataframe.
        """
        if not self.is_fitted_:
            raise ValueError(
                f"This {self.__class__.__name__} instance is not fitted yet."
                f"Call 'fit' with appropriate arguments before using this method.")

        if self.correlation_th is not None:
            X_original = X.copy()

        scores = list()
        X_eval = utils.cast_categoricals_to_int(X)
        for target_name in self.feature_names:
            if self.correlation_th is not None:
                X_eval = X_original.drop(
                    self.correlated_features[target_name], axis=1)

            R2 = self.regressor[target_name].score(
                X_eval.drop(target_name, axis=1), X_eval[target_name])

            # Append 1.0 if R2 is negative, or 1.0 - R2 otherwise since we're
            # in the minimization mode of the error function.
            scores.append(1.0) if R2 < 0.0 else scores.append(1.0 - R2)

        self.scoring = np.array(scores)
        return self.scoring

    def tune(
        self,
        training_data: pd.DataFrame,
        test_data: pd.DataFrame,
        study_name: str = None,
        min_loss: float = 0.05,
        storage: str = "sqlite:///rex_tuning.db",
        load_if_exists: bool = True,
        n_trials: int = DEFAULT_HPO_TRIALS
    ):
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

            def __init__(
                    self,
                    train_data,
                    test_data,
                    device='cpu',
                    prog_bar=True,
                    verbose=False):
                self.train_data = train_data
                self.test_data = test_data
                self.device = device
                self.random_state = GBTRegressor.random_state
                self.prog_bar = prog_bar
                self.verbose = verbose

            def __call__(self, trial):
                """
                This method is called by Optuna to evaluate the objective function.
                """
                # Define the model hyperparameters
                self.n_iter_no_change = 5
                self.tol = 0.0001

                # Define the hyperparameters to optimize
                self.learning_rate = trial.suggest_float(
                    "learning_rate", 0.001, 0.2)
                self.n_estimators = trial.suggest_int("n_estimators", 10, 1000)
                self.subsample = trial.suggest_float("subsample", 0.1, 1.0)
                self.min_samples_split = trial.suggest_int(
                    "min_samples_split", 2, 10)
                self.min_samples_leaf = trial.suggest_int(
                    "min_samples_leaf", 1, 10)
                self.min_weight_fraction_leaf = trial.suggest_float(
                    "min_weight_fraction_leaf", 0.0, 0.5)
                self.max_depth = trial.suggest_int("max_depth", 3, 20)
                self.max_leaf_nodes = trial.suggest_int(
                    "max_leaf_nodes", 10, 1000)
                self.min_impurity_decrease = trial.suggest_float(
                    "min_impurity_decrease", 0.0, 0.5)

                self.models = GBTRegressor(
                    learning_rate=self.learning_rate,
                    n_estimators=self.n_estimators,
                    subsample=self.subsample,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                    max_depth=self.max_depth,
                    min_impurity_decrease=self.min_impurity_decrease,
                    random_state=self.random_state,
                    verbose=False,
                    max_leaf_nodes=self.max_leaf_nodes,
                    n_iter_no_change=self.n_iter_no_change,
                    tol=self.tol,
                    prog_bar=True & (not self.verbose) & (self.prog_bar),
                    silent=True)

                self.models.fit(self.train_data)

                # Now, measure the performance of the model with the test data.
                loss = []
                X_test = utils.cast_categoricals_to_int(self.test_data)
                for target_name in list(self.train_data.columns):
                    model = self.models.regressor[target_name]
                    # For regressors, this is R2, for classifiers this is accuracy
                    if model.__class__.__name__ == "GradientBoostingClassifier":
                        # Get the F1 score of the model
                        goodness_of_fit = f1_score(
                            X_test[target_name],
                            model.predict(X_test.drop(target_name, axis=1)))
                    elif model.__class__.__name__ == "GradientBoostingRegressor":
                        goodness_of_fit = model.score(
                            X_test.drop(target_name, axis=1), X_test[target_name])
                    else:
                        raise ValueError(
                            f"Model {model.__class__.__name__} is not supported."
                            f"Only GradientBoostingClassifier and"
                            f"GradientBoostingRegressor are supported.")

                    # Append 1.0 if R2 is negative, or 1.0 - R2 otherwise since we're
                    # in the minimization mode of the error function.
                    loss.append(1.0) if goodness_of_fit < 0.0 else loss.append(
                        1.0 - goodness_of_fit)

                return np.median(loss)

        # Callback function to stop the stud if the loss is below a given threshold
        def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            if trial.value < min_loss or study.best_value < min_loss:
                study.stop()

        if self.verbose is False:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create and run the HPO study
        study = optuna.create_study(
            direction='minimize', study_name=study_name, storage=storage,
            load_if_exists=load_if_exists)
        study.optimize(
            Objective(
                training_data, test_data, prog_bar=self.prog_bar, verbose=self.verbose),
            n_trials=n_trials,
            gc_after_trial=True,
            show_progress_bar=(self.optuna_prog_bar & (
                not self.silent) & (not self.verbose)),
            callbacks=[callback])

        # Capture the best hyperparameters and the minimum loss
        best_trials = sorted(study.best_trials, key=lambda x: x.values[0])
        self.best_params = best_trials[0].params
        self.min_tunned_loss = best_trials[0].values[0]

        if self.verbose and not self.silent:
            print(f"Best params (min loss:{self.min_tunned_loss:.6f}):")
            for k, v in self.best_params.items():
                print(f"\t{k:<15s}: {v}")

        regressor_args = {
            'learning_rate': self.best_params['learning_rate'],
            'n_estimators': self.best_params['n_estimators'],
            'subsample': self.best_params['subsample'],
            'min_samples_split': self.best_params['min_samples_split'],
            'min_samples_leaf': self.best_params['min_samples_leaf'],
            'min_weight_fraction_leaf': self.best_params['min_weight_fraction_leaf'],
            'max_depth': self.best_params['max_depth'],
            'max_leaf_nodes': self.best_params['max_leaf_nodes'],
            'min_impurity_decrease': self.best_params['min_impurity_decrease']
        }

        return regressor_args

    def tune_fit(
            self,
            X: pd.DataFrame,
            hpo_study_name: str = None,
            hpo_min_loss: float = 0.05,
            hpo_storage: str = 'sqlite:///rex_tuning.db',
            hpo_load_if_exists: bool = True,
            hpo_n_trials: int = DEFAULT_HPO_TRIALS):
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
        self.fit(train_data)


#
# Main function
#

def custom_main(experiment_name='custom_rex', score: bool = False, tune: bool = False):
    from causalgraph.common import utils
    path = "/Users/renero/phd/data/RC3/"
    output_path = "/Users/renero/phd/output/RC3/"

    ref_graph = utils.graph_from_dot_file(f"{path}{experiment_name}.dot")

    data = pd.read_csv(f"{path}{experiment_name}.csv")
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
        gbt = GBTRegressor(silent=True, prog_bar=False)  # verbose=True)
        gbt.tune_fit(train, hpo_study_name=experiment_name, hpo_n_trials=100)
        print(gbt.score(test))


if __name__ == "__main__":
    custom_main("rex_generated_linear_6", tune=True)
