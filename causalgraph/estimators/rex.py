import os
import types
import warnings
from copy import copy
from pathlib import Path
from typing import Any, List, Tuple, Union
import inspect

import networkx as nx
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (check_array, check_is_fitted,
                                      check_random_state)

from causalgraph.common import GRAY, GREEN, RESET
from causalgraph.common.pipeline import Pipeline
from causalgraph.common.plots import plot_dags, subplots
from causalgraph.common.utils import (graph_from_dot_file, load_experiment,
                                      save_experiment)
from causalgraph.explainability.shapley import ShapEstimator
from causalgraph.independence.graph_independence import GraphIndependence
from causalgraph.metrics.compare_graphs import evaluate_graph
from causalgraph.explainability.hierarchies import Hierarchies
from causalgraph.models import NNRegressor, GBTRegressor


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
    """
            method: str = 'cluster',
            sensitivity: float = 1.0,
            tolerance: float = None,
            descending: bool = False,
            iters: int = 20,
            reciprocity: False = False,
            min_impact: float = 1e-06,
            on_gpu: bool = False,
"""

    def __init__(
            self,
            model_type: BaseEstimator = NNRegressor,
            explainer=shap.Explainer,
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
            silent: bool = False,
            do_plot_correlations: bool = False,
            do_plot_shap: bool = False,
            do_plot_discrepancies: bool = False,
            do_compare_shap: bool = False,
            do_compare_fci: bool = False,
            do_compare_final: bool = False,
            shap_fsize: Tuple[int, int] = (10, 10),
            dpi: int = 75,
            pdf_filename: str = None,
            random_state=1234, 
            **kwargs):
        """
        Arguments:
        ----------
            model_type (BaseEstimator): The type of model to use. Either NNRegressor 
                or SKLearn GradientBoostingRegressor.
            explainer (shap.Explainer): The explainer to use for the shap values.
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
            silent (bool): Whether to print anything. Default is False. This overrides
                the verbose argument and the prog_bar argument.
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
        self.explainer = explainer
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
        self.silent = silent
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

        self._get_param_values_from_kwargs(self.model_type, kwargs)
        self._get_param_values_from_kwargs(ShapEstimator, kwargs)

        self._fit_desc = "Running Causal Discovery pipeline"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def _get_param_values_from_kwargs(self, object_attribute, kwargs):
        # Check if any of the arguments required by Regressor (model_type) are
        # present in the kwargs. If so, take them as a property of the class.
        arguments = inspect.signature(object_attribute.__init__).parameters
        constructor_parameters = {arg: arguments[arg].default for arg in arguments.keys()}
        constructor_parameters.pop('self', None)
        for param in constructor_parameters.keys():
            if param in kwargs:
                setattr(self, param, kwargs[param])

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
        self.feature_names_ = list(X.columns)
        self.X = copy(X)
        self.y = copy(y) if y is not None else None

        pipeline = Pipeline(self, prog_bar=self.prog_bar, verbose=self.verbose,
                            silent=self.silent)
        steps = [
            ('regressor', self.model_type),
            'regressor.fit',
            ('shaps', ShapEstimator, {'models': 'regressor'}),
            'shaps.fit',
            ('hierarchies', Hierarchies),
            'hierarchies.fit'
        ]
        pipeline.run(steps, self._fit_desc)

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
        
        # Overwrite values for prog_bar and verbosity with current pipeline
        # values, in case predict is called from a loaded experiment
        self.shaps.prog_bar = self.prog_bar
        self.shaps.verbose = self.verbose

        steps = [
            ('G_shap', 'shaps.predict'),
            ('indep', GraphIndependence, {'base_graph': 'G_shap'}),
            'indep.fit',
            ('G_indep', 'indep.predict'),
            ('G_final', 'shaps.adjust', {'graph': 'G_indep'})
        ]
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
            elif isinstance(getattr(self, attr), pd.DataFrame):
                ret += f"{attr:25} DataFrame {getattr(self, attr).shape}\n"
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
        plot_args = [(target_name) for target_name in self.feature_names_]
        return subplots(self.shaps._plot_shap_summary, *plot_args, **kwargs);


def main():
    np.set_printoptions(precision=4, linewidth=150)
    warnings.filterwarnings('ignore')

    load = False
    save = False
    input_path = "/Users/renero/phd/data/RC3/"
    output_path = input_path.replace('data', 'output')
    dataset_name = 'rex_generated_polynomial_1'

    data = pd.read_csv(f"{input_path}{dataset_name}.csv")
    ref_graph = graph_from_dot_file(f"{input_path}{dataset_name}.dot")

    if load:
        rex = load_experiment('rex', output_path)
    else:
        rex = Rex(explainer=shap.Explainer, num_epochs=100, hidden_dim=[10],
                  early_stop=False, learning_rate=0.002, batch_size=64, dropout=0.05)
        rex.fit(data, ref_graph)

    rex.prog_bar = True
    rex.verbose = False
    pred_graph = rex.predict(data)

    metric = evaluate_graph(ref_graph, pred_graph, rex.shaps.all_feature_names_)
    print(metric)

    rex.plot_shap_discrepancies('V6')

    # Plot the SHAP values for each regression
    plot_args = [(target_name) for target_name in rex.shaps.all_feature_names_]
    subplots(rex.shaps._plot_shap_summary, *plot_args, dpi=75);

    # Plot the predicted graph
    plot_dags(pred_graph, ref_graph)


    if save:
        save_experiment('rex', "/Users/renero/phd/output/REX", rex)


def main2():
    import warnings
    import shap

    import numpy as np
    import pandas as pd
    from causalgraph.common.utils import graph_from_dot_file
    from causalgraph.estimators import Rex
    from causalgraph.models import GBTRegressor, NNRegressor
    from causalgraph.metrics.compare_graphs import evaluate_graph
    from sklearn.preprocessing import StandardScaler

    np.set_printoptions(precision=4, linewidth=150)
    warnings.filterwarnings('ignore')

    path = "/Users/renero/phd/data/RC3/"
    dataset_name = 'rex_generated_polynomial_1'

    ref_graph = graph_from_dot_file(f"{path}{dataset_name}.dot")
    data = pd.read_csv(f"{path}{dataset_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    rex = Rex(model_type=GBTRegressor, explainer=shap.Explainer, 
              num_epochs=1, hidden_dim=[10],
            early_stop=False, learning_rate=0.002, batch_size=64, dropout=0.2)
    print(rex)
    rex.fit(data)
    pred_graph = rex.predict(data)
    print(evaluate_graph(ref_graph, rex.G_shap, rex.shaps.feature_names_))


if __name__ == "__main__":
    main2()