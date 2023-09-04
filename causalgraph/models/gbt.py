import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from tqdm.auto import tqdm

from causalgraph.common import tqdm_params
from causalgraph.explainability.hierarchies import Hierarchies


class GBTRegressor(GradientBoostingRegressor):

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
            random_state=None,
            max_features=None,
            alpha=0.9,
            max_leaf_nodes=None,
            warm_start=False,
            validation_fraction=0.1,
            n_iter_no_change=None,
            tol=0.0001,
            ccp_alpha=0.0,
            correlation_th: float = None,
            verbose=False,
            silent=False,
            prog_bar=True):

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
        self.alpha = alpha
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
        self.feature_names = list(X.columns)
        self.regressor = dict()
        
        if self.correlation_th:
            self.corr_matrix = Hierarchies.compute_correlation_matrix(X)
            self.correlated_features = Hierarchies.compute_correlated_features(
                self.corr_matrix, self.correlation_th, self.feature_names,
                verbose=self.verbose)
            X_original = X.copy()        

        pbar_in = tqdm(total=len(self.feature_names),
                       **tqdm_params(self._fit_desc, self.prog_bar,
                                     silent=self.silent))

        for target_name in self.feature_names:
            pbar_in.refresh()
            
            # if correlation_th is not None then, remove features that are highly
            # correlated with the target, at each step of the loop
            if self.correlation_th is not None:
                X = X_original.copy()
                if len(self.correlated_features[target_name]) > 0:
                    X = X.drop(self.correlated_features[target_name], axis=1)
                    if self.verbose:
                        print("REMOVED CORRELATED FEATURES: ",
                              self.correlated_features[target_name])            
            
            self.regressor[target_name] = GradientBoostingRegressor(
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
                alpha=self.alpha,
                verbose=False,
                max_leaf_nodes=self.max_leaf_nodes,
                warm_start=self.warm_start,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                ccp_alpha=self.ccp_alpha
            )
            self.regressor[target_name].fit(X.drop(target_name, axis=1), X[target_name])
            pbar_in.update(1)
        pbar_in.close()

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
                X = X_original.drop(self.correlated_features[target_name], axis=1)
            
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
        for target_name in self.feature_names:
            if self.correlation_th is not None:
                X = X_original.drop(self.correlated_features[target_name], axis=1)
                
            scores.append(
                self.regressor[target_name].score(
                    X.drop(target_name, axis=1), X[target_name]))
            
        self.scoring = np.array(scores)
        return self.scoring
