from sklearn.ensemble import GradientBoostingRegressor
from tqdm.auto import tqdm

from causalgraph.common import tqdm_params


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
            verbose=False,
            max_leaf_nodes=None,
            warm_start=False,
            validation_fraction=0.1,
            n_iter_no_change=None,
            tol=0.0001,
            ccp_alpha=0.0,
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
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha

        self.silent = silent
        self.prog_bar = prog_bar
        self._estimator_type = 'regressor'
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

        pbar_in = tqdm(total=len(self.feature_names),
                       **tqdm_params(self._fit_desc, self.prog_bar,
                                     silent=self.silent))

        for target in self.feature_names:
            pbar_in.refresh()
            self.regressor[target] = GradientBoostingRegressor(
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
                verbose=self.verbose,
                max_leaf_nodes=self.max_leaf_nodes,
                warm_start=self.warm_start,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                ccp_alpha=self.ccp_alpha
            )
            self.regressor[target].fit(X.drop(target, axis=1), X[target])
            pbar_in.update(1)
        pbar_in.close()

        self.is_fitted_ = True
        return self

    def predict(self, X):
        return super().predict(X)
