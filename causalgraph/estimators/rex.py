from copy import copy
from pathlib import Path
import types
from typing import Any, List, Tuple, Union
import warnings
import os

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (check_array, check_is_fitted,
                                      check_random_state)

from causalgraph.common import RESET, GREEN, GRAY
from causalgraph.common.pipeline import Pipeline
from causalgraph.common.plots import subplots, plot_dags
from causalgraph.common.utils import graph_from_dot_file, load_experiment, save_experiment
from causalgraph.explainability.shapley import ShapEstimator
from causalgraph.independence.graph_independence import GraphIndependence
from causalgraph.models.dnn import NNRegressor
from causalgraph.metrics.compare_graphs import evaluate_graph


class Rex(BaseEstimator, ClassifierMixin):
    """ Regression with Explainability (Rex) is a causal inference discovery that
    uses a regression model to predict the outcome of a treatment and uses
    explainability to identify the causal variables.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from causalgraph import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    def __init__(
            self,
            model_type: str = 'mlp',
            hidden_dim: Union[int, List[int]] = [20, 20, 40],
            learning_rate: float = 0.0035,
            dropout: float = 0.065,
            batch_size: int = 45,
            num_epochs: int = 25, #250,
            loss_fn='mse',
            devices="auto",
            test_size=0.1,
            sensitivity=1.0,
            descending=False,
            tolerance=0.04,
            shap_selection: str = 'cluster',
            iters=10,
            reciprocal=False,
            min_impact=1e-06,
            early_stop: bool = True,
            patience: int = 10,
            min_delta: float = 0.001,
            corr_method: str = 'spearman',
            corr_alpha: float = 0.6,
            corr_clusters: int = 15,
            shap_diff_upper_bound: float = 0.1,
            correction_method: str = 'heuristic',
            correction_model: Union[str, Path] = None,
            increase_tolerance: float = 0.0,
            condlen: int = 1,
            condsize: int = 0,
            verbose: bool = False,
            prog_bar=True,
            do_plot_correlations: bool = False,
            do_plot_shap: bool = False,
            do_plot_discrepancies: bool = False,
            do_compare_shap: bool = False,
            do_compare_fci: bool = False,
            do_compare_final: bool = False,
            shap_fsize: Tuple[int, int] = (10, 10),
            dpi: int = 75,
            pdf_filename: str = None,
            random_state=1234):
        """
        Arguments:
        ----------
            model_type (str): The type of neural network to use. Either 'dff' or 'mlp'.
            hidden_dim (int): The dimensions of each hidden layer. For DFF networks
                this is a single integer. For MLP networks this is an array with the
                dimensions of each hidden layer.
            learning_rate (float): The learning rate for the optimizer.
            dropout (float): The dropout rate for the dropout layer.
            batch_size (int): The batch size for the optimizer.
            num_epochs (int): The number of epochs for the optimizer.
            loss_fn (str): The loss function to use. Default is "mse".
            gpus (int): The number of GPUs to use. Default is 0.
            test_size (float): The proportion of the data to use for testing. Default
                is 0.1.
            sensitivity (float): The sensitivity of the Knee algorithm. Default is 1.0.
            descending (bool): Whether to determine the cutoff for the most
                influencing shap values starting from the higher one. Default is False.
            tolerance (float): The tolerance for the causal graph. Default is None.
                which implies that is automatically determined.
            shape_selection (str): The method to use for the shap value selection.
                Default is "abrupt", but it can also be 'linear' or 'knee'.
            iters (int): The number of iterations for getting the correct
                orientation for every edge. Default is 10.
            reciprocal (bool): Whether to force reciprocal feature selection to assign
                an edge between two features. Default is False.
            min_impact (float): The minimum impact of all features to be selected.
                If all features have an impact below this value, then none of them
                will be selected. Default is 1e-06.
            early_stop (bool): Whether to use early stopping. Default is True.
            patience (int): The patience for the early stopping. Default is 10.
            min_delta (float): The minimum change in the loss to trigger the early
                stopping. Default is 0.001.
            corr_method (str): The method to use for the correlation.
                Default is "spearman", but it can also be 'pearson', 'kendall or 'mic'.
            corr_alpha (float): The alpha value for the correlation. Default is 0.6.
            corr_clusters (int): The number of clusters to use for the correlation.
                Default is 15.
            shap_diff_upper_bound (float): The upper bound for the shap values
                difference. This value is the maximum difference between the forward
                and backward direction in the discrepancies matrix. Default is 0.1.
            correction_method (str): The method to use for the correction, can be
                'heuristic' or 'model'. Default is 'heuristic'.
            correction_model (str or Path): The path to the model to use for the
                correction. Default is None.
            increase_tolerance (float): The increase in the tolerance for the
                correction. Default is 0.0. This increase only occurs when certain
                SHAP discrepancies matrix properties have been met.
            condlen (int): The depth of the conditioning sequence. Default is 1.
            condsize (int): The size of the conditioning sequence. Default is 0.
            prog_bar (bool): Whether to display a progress bar.
                Default is False.
            verbose (bool): Whether to print the progress of the training. Default
                is False.
            random_state (int): The seed for the random number generator.
                Default is 1234.

            Additional arguments:
                do_plot_correlations: Whether to plot the correlations between the
                    features. Default is True.
                do_plot_shap: Whether to plot the shap values. Default is True.
                do_plot_discrepancies: Whether to plot the discrepancies between the
                    shap values as a matrix. Default is False.
                do_compare_shap: Whether to plot the dot comparison between the
                    causal graph and the shap graph. Default is True.
                do_compare_fci: Whether to plot the dot comparison between the
                    causal graph and the after-FCI graph. Default is True.
                do_compare_final: Whether to plot the dot comparison between the
                    causal graph and the final graph. Default is True.
                shap_fsize: The size of the figure for the shap values.
                    Default is (5, 3).
                dpi: The dpi for the figures. Default is 75.
                pdf_filename: The filename for the pdf file where final comparison will
                    be saved. Default is None, producing no pdf file.
        """
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss_fn = loss_fn
        self.devices = devices
        self.have_gpu = (isinstance(self.devices, int) and self.devices != 0)
        self.test_size = test_size
        self.min_impact = min_impact
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.sensitivity = sensitivity
        self.descending = descending
        self.tolerance = tolerance
        self.shap_selection = shap_selection
        self.iters = iters
        self.reciprocal = reciprocal
        self.corr_method = corr_method
        self.corr_alpha = corr_alpha
        self.corr_clusters = corr_clusters
        self.shap_diff_upper_bound = shap_diff_upper_bound
        self.correction_method = correction_method
        self.correction_model = correction_model
        self.increase_tolerance = increase_tolerance
        self.condlen = condlen
        self.condsize = condsize
        self.prog_bar = prog_bar
        self.verbose = verbose
        self.random_state = random_state

        self.do_plot_correlations = do_plot_correlations
        self.do_plot_shap = do_plot_shap
        self.do_plot_discrepancies = do_plot_discrepancies
        self.do_compare_shap = do_compare_shap
        self.do_compare_fci = do_compare_fci
        self.do_compare_final = do_compare_final
        self.shap_fsize = shap_fsize
        self.dpi = dpi
        self.pdf_filename = pdf_filename

        self._fit_desc = "Fitting ReX method"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def _more_tags(self):
        return {
            'multioutput_only': True,
            "non_deterministic": True,
            "no_validation": True,
            "poor_score": True,
            "_xfail_checks": {
                "check_methods_sample_order_invariance": "This test shouldn't be running at all!",
                "check_methods_subset_invariance": "This test shouldn't be running at all!",
            }
        }

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """
        self.random_state_ = check_random_state(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.X = copy(X)
        self.y = copy(y) if y is not None else None

        pipeline = Pipeline(self, prog_bar=self.prog_bar, verbose=self.verbose)
        pipeline.set_default_object_method('fit')
        pipeline.set_default_method_params(['X', 'y'])
        # TODO:
        # I can eliminate the need to specify the list of parameters of the method 
        # by using the inspect module to get the list of parameters of the method.
        # Example: inspect.getfullargspec(method).args
        steps = {
            ('regressor', NNRegressor): [
                "model_type", "hidden_dim", "learning_rate", "dropout", "batch_size", 
                "num_epochs", "loss_fn", "devices", "test_size", "early_stop", "patience", 
                "min_delta", "random_state", "verbose", "prog_bar"],
            ('shaps', ShapEstimator): [
                "regressor", "shap_selection", "sensitivity", "tolerance", "descending", 
                "iters", "reciprocal", "min_impact", "have_gpu", "verbose", "prog_bar"]
        }
        pipeline.run(steps, "Training REX")

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)

        # Create a new pipeline for the prediction stages.
        prediction = Pipeline(self, prog_bar=self.prog_bar, verbose=self.verbose)
        prediction.set_default_object_method('fit')
        prediction.set_default_method_params(['X', 'y'])
        
        # Overwrite values for prog_bar and verbosity with current pipeline
        # values, in case predict is called from a loaded experiment
        self.shaps.prog_bar = self.prog_bar
        self.shaps.verbose = self.verbose

        steps = {
            ('G_shap', 'shaps.predict'): ["X"],
            ('indep', GraphIndependence): ["G_shap", "condlen", "condsize", 
                                           "prog_bar", "verbose"],
            ('G_indep', 'indep.predict'): [],
            ('G_final', 'shaps.adjust'): ['X', 'G_indep']
        }
        prediction.run(steps, "Predicting graph")
        if '\\n' in self.G_final.nodes:
            self.G_final.remove_node('\\n')

        return self.G_final

    def score(self, X, y):
        return np.random.randint(self.n_features_in_**2)

    def __repr__(self):
        forbidden_attrs = ['fit', 'predict', 'score', 'get_params', 'set_params']
        ret = f"{GREEN}REX object attributes{RESET}\n"
        ret += f"{GRAY}{'-'*80}{RESET}\n"
        for attr in dir(self):
            if attr.startswith('_') or attr in forbidden_attrs or type(getattr(self, attr)) == types.MethodType:
                continue
            elif attr == "X" or attr == "y":
                if isinstance(getattr(self, attr), pd.DataFrame):
                    ret += f"{attr:25} {getattr(self, attr).shape}\n"
                    continue
                elif isinstance(getattr(self, attr), nx.DiGraph):
                    n_nodes = getattr(self, attr).number_of_nodes()
                    n_edges = getattr(self, attr).number_of_edges()
                    ret += f"{attr:25} {n_nodes} nodes, {n_edges} edges\n"
                    continue
            else:
                ret += f"{attr:25} {getattr(self, attr)}\n"

        return ret

    def plot_shap_discrepancies(self, target_name:str, **kwargs):
        assert self.is_fitted_, "Model not fitted yet"
        # X = self.X.drop(target_name, axis=1)
        # y = self.X[target_name].values
        # return self.shaps._plot_discrepancies_for_target(X, y, target_name, **kwargs)
        self.shaps._plot_discrepancies(self.X, target_name, **kwargs)

    def plot_shap_values(self, **kwargs):
        assert self.is_fitted_, "Model not fitted yet"
        plot_args = [(target_name) for target_name in self.shaps.all_feature_names_]
        return subplots(self.shaps.summary_plot, *plot_args, **kwargs);


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=150)
    warnings.filterwarnings('ignore')

    load = True
    save = False
    dataset_name = 'generated_linear_10'

    data = pd.read_csv("~/phd/data/generated_linear_10.csv")
    ref_graph = graph_from_dot_file("/Users/renero/phd/data/generated_linear_10.dot")

    if load:
        rex = load_experiment('rex', "/Users/renero/phd/output/REX")
    else:
        rex = Rex().fit(data, ref_graph)

    # rex.prog_bar = False
    # rex.verbose = True

    pred_graph = rex.predict(data)
    rex.plot_shap_discrepancies('V6')

    # Plot the SHAP values for each regression
    plot_args = [(target_name) for target_name in rex.shaps.all_feature_names_]
    subplots(rex.shaps.summary_plot, *plot_args, dpi=100);

    # Plot the predicted graph
    plot_dags(pred_graph, ref_graph)

    metric = evaluate_graph(ref_graph, pred_graph, rex.shaps.all_feature_names_)
    print(metric)

    if save:
        save_experiment('rex', "/Users/renero/phd/output/REX", rex)
