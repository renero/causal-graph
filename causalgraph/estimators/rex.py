#
# Main class for the REX estimator.
#
# (C) J. Renero, 2022, 2023
#

import os
import types
import warnings
from copy import copy
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, check_random_state

from causalgraph.common import GRAY, GREEN, RESET
from causalgraph.common.pipeline import Pipeline
from causalgraph.common.plots import (_cleanup_graph, _draw_graph_subplot,
                                      _format_graph, formatting_kwargs,
                                      setup_plot, subplots)
from causalgraph.common.utils import (graph_from_dot_file, load_experiment,
                                      save_experiment)
from causalgraph.estimators.knowledge import Knowledge
from causalgraph.explainability import (Hierarchies, PermutationImportance,
                                        ShapEstimator)
from causalgraph.independence.graph_independence import GraphIndependence
from causalgraph.metrics.compare_graphs import evaluate_graph
from causalgraph.models import GBTRegressor, NNRegressor

np.set_printoptions(precision=4, linewidth=120)
warnings.filterwarnings('ignore')


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
            model_type: BaseEstimator = NNRegressor,
            explainer=shap.Explainer,
            tune_model: bool = False,
            corr_th: float = None,
            corr_method: str = 'spearman',
            corr_alpha: float = 0.6,
            corr_clusters: int = 15,
            shap_diff_upper_bound: float = 0.1,
            correction_method: str = 'heuristic',
            correction_model: Union[str, Path] = None,
            increase_tolerance: float = 0.0,
            condlen: int = 1,
            condsize: int = 0,
            mean_pi_percentile: float = 0.8,
            verbose: bool = False,
            prog_bar=True,
            silent: bool = False,
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
            tune_model (bool): Whether to tune the model for HPO. Default is False.
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
                shap_fsize: The size of the figure for the shap values.
                    Default is (5, 3).
                dpi: The dpi for the figures. Default is 75.
                pdf_filename: The filename for the pdf file where final comparison will
                    be saved. Default is None, producing no pdf file.
        """
        self.model_type = model_type
        self.explainer = explainer
        self.tune_model = tune_model
        self.correlation_th = corr_th
        self.corr_method = corr_method
        self.corr_alpha = corr_alpha
        self.corr_clusters = corr_clusters
        self.shap_diff_upper_bound = shap_diff_upper_bound
        self.correction_method = correction_method
        self.correction_model = correction_model
        self.increase_tolerance = increase_tolerance
        self.condlen = condlen
        self.condsize = condsize
        self.mean_pi_percentile = mean_pi_percentile
        self.mean_pi_threshold = 0.0
        self.prog_bar = prog_bar
        self.verbose = verbose
        self.silent = silent
        self.random_state = random_state

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
        # arguments = inspect.signature(object_attribute.__init__).parameters
        # constructor_parameters = {
        #     arg: arguments[arg].default for arg in arguments.keys()}
        # constructor_parameters.pop('self', None)
        # for param in constructor_parameters.keys():
        #     if param in kwargs:
        #         setattr(self, param, kwargs[param])
        for k, v in kwargs.items():
            setattr(self, k, v)

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
        self.random_state_state = check_random_state(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.feature_names = list(X.columns)
        self.X = copy(X)
        self.y = copy(y) if y is not None else None

        # If the model is to be tuned for HPO, then the step for fitting the regressors
        # is different.
        fit_step = 'models.tune_fit' if self.tune_model else 'models.fit'

        # Create the pipeline for the training stages.
        pipeline = Pipeline(host=self, prog_bar=self.prog_bar, verbose=self.verbose,
                            silent=self.silent)
        steps = [
            ('hierarchies', Hierarchies),
            ('hierarchies.fit'),
            ('models', self.model_type),
            (fit_step),
            ('shaps', ShapEstimator, {'models': 'models'}),
            ('shaps.fit'),
            ('pi', PermutationImportance, {'models': 'models'}),
            ('pi.fit_predict', {'X': self.X}),
        ]
        pipeline.run(steps, self._fit_desc)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, ref_graph: nx.DiGraph = None):
        """
        Predicts the causal graph from the given data.

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

        # Create a new pipeline for the prediction stages.
        prediction = Pipeline(
            self, prog_bar=self.prog_bar, verbose=self.verbose)

        # Overwrite values for prog_bar and verbosity with current pipeline
        #  values, in case predict is called from a loaded experiment
        self.shaps.prog_bar = self.prog_bar
        self.shaps.verbose = self.verbose

        steps = [
            ('G_shap', 'shaps.predict'),
            ('indep', GraphIndependence, {'base_graph': 'G_shap'}),
            ('G_indep', 'indep.fit_predict'),
            ('G_final', 'shaps.adjust', {'graph': 'G_indep'})
        ]
        prediction.run(steps, "Predicting graph")
        if '\\n' in self.G_final.nodes:
            self.G_final.remove_node('\\n')

        # Obtain the regression loss metric from each regressor trained to predict
        # each of the variables.
        self.models.score(X)

        # If a reference graph is provided, compute the metrics for the predicted
        # graph against the reference graph.
        if ref_graph is not None:
            self.knowledge(ref_graph)

        return self.G_final

    def fit_predict(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            ref_graph: nx.DiGraph):
        """
        Fit the model according to the given training data and predict
        the outcome of the treatment.

        Parameters
        ----------
        train : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        test : {array-like, sparse matrix}, shape (n_samples, n_features)
            Test vector, where n_samples is the number of samples and
            n_features is the number of features.
        ref_graph : nx.DiGraph
            The reference graph, or ground truth.

        Returns
        -------
        G_final : nx.DiGraph
            The final graph, after the correction stage.
        """
        self.fit(train)
        self.predict(test, ref_graph)

    def score(
            self,
            ref_graph: nx.DiGraph,
            predicted_graph: Union[str, nx.DiGraph] = 'final'):
        """
        Obtains the score of the predicted graph against the reference graph.
        The score contains different metrics, such as the precision, recall,
        F1-score, SHD or SID.

        Parameters:
        -----------
            ref_graph (nx.DiGraph): The reference graph, or ground truth.
            predicted_graph (str): The name of the graph to use for the score.
                Default is 'final', but other possible intermediate graphs are
                'shap' and 'indep', for those stages of the pipeline corresponding
                to the graph constructed by interpreting only the SHAP values and
                the graph constructed after the FCI algorithm, respectively.
        """
        if isinstance(predicted_graph, str):
            if predicted_graph == 'final':
                pred_graph = self.G_final
            elif predicted_graph == 'shap':
                pred_graph = self.G_shap
            elif predicted_graph == 'indep':
                pred_graph = self.G_indep
            else:
                raise ValueError(
                    f"Predicted graph must be one of 'final', 'shap' or 'indep'.")
        elif isinstance(predicted_graph, nx.DiGraph):
            pred_graph = predicted_graph

        return evaluate_graph(ref_graph, pred_graph, self.feature_names)

    def knowledge(self, ref_graph: nx.DiGraph):
        """
        Returns a dataframe with the knowledge about each edge in the graph
        The dataframe is obtained from the Knowledge class.

        Parameters:
        -----------
            ref_graph (nx.DiGraph): The reference graph, or ground truth.
        """
        K = Knowledge(self, ref_graph)
        self.learnings = K.data()
        return self.learnings

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
                elif isinstance(getattr(self, attr), nx.DiGraph):
                    n_nodes = getattr(self, attr).number_of_nodes()
                    n_edges = getattr(self, attr).number_of_edges()
                    ret += f"{attr:25} {n_nodes} nodes, {n_edges} edges\n"
                    continue
            elif isinstance(getattr(self, attr), pd.DataFrame):
                ret += f"{attr:25} DataFrame {getattr(self, attr).shape}\n"
            # check if attr is an object
            elif isinstance(getattr(self, attr), BaseEstimator):
                ret += f"{attr:25} {BaseEstimator}\n"
            else:
                ret += f"{attr:25} {getattr(self, attr)}\n"

        return ret

    @staticmethod
    def plot_dags(
            dag: nx.DiGraph,
            reference: nx.DiGraph = None,
            names: List[str] = ["REX Prediction", "Ground truth"],
            figsize: Tuple[int, int] = (10, 5),
            dpi: int = 75,
            save_to_pdf: str = None,
            **kwargs):
        """
        Compare two graphs using dot.

        Parameters:
        -----------
        reference: The reference DAG.
        dag: The DAG to compare.
        names: The names of the reference graph and the dag.
        figsize: The size of the figure.
        **kwargs: Additional arguments to format the graphs:
            - "node_size": 500
            - "node_color": 'white'
            - "edgecolors": "black"
            - "font_family": "monospace"
            - "horizontalalignment": "center"
            - "verticalalignment": "center_baseline"
            - "with_labels": True
        """
        ncols = 1 if reference is None else 2

        # Overwrite formatting_kwargs with kwargs if they are provided
        formatting_kwargs.update(kwargs)

        G = nx.DiGraph()
        G.add_edges_from(dag.edges())
        if reference:
            Gt = _cleanup_graph(reference.copy())
            for missing in set(list(Gt.nodes)) - set(list(G.nodes)):
                G.add_node(missing)

            # Gt = _format_graph(Gt, Gt, inv_color="red", wrong_color="black")
            # G  = _format_graph(G, Gt, inv_color="red", wrong_color="gray")
            Gt = _format_graph(
                Gt, G, inv_color="lightgreen", wrong_color="lightgreen")
            G = _format_graph(G, Gt, inv_color="orange", wrong_color="red")
        else:
            G = _format_graph(G)

        ref_layout = None
        setup_plot(dpi=dpi)
        f, ax = plt.subplots(ncols=ncols, figsize=figsize)
        ax_graph = ax[1] if reference else ax
        if save_to_pdf is not None:
            with PdfPages(save_to_pdf) as pdf:
                if reference:
                    ref_layout = nx.drawing.nx_agraph.graphviz_layout(
                        Gt, prog="dot")
                    _draw_graph_subplot(Gt, layout=ref_layout, title=None, ax=ax[0],
                                        **formatting_kwargs)
                _draw_graph_subplot(G, layout=ref_layout, title=None, ax=ax_graph,
                                    **formatting_kwargs)
                pdf.savefig(f, bbox_inches='tight', pad_inches=0)
                plt.close()
        else:
            if reference:
                ref_layout = nx.drawing.nx_agraph.graphviz_layout(
                    Gt, prog="dot")
                _draw_graph_subplot(Gt, layout=ref_layout, title=names[1], ax=ax[0],
                                    **formatting_kwargs)
            _draw_graph_subplot(G, layout=ref_layout, title=names[0], ax=ax_graph,
                                **formatting_kwargs)
            plt.show()

    @staticmethod
    def plot_dag(
            dag: nx.DiGraph,
            figsize: Tuple[int, int] = (5, 5),
            dpi: int = 75,
            save_to_pdf: str = None,
            **kwargs):
        """
        Plot a DAG without formatting edges.

        Parameters:
        -----------
        dag: The DAG to plot.
        figsize: The size of the figure.
        **kwargs: Additional arguments to format the graphs:
            - "node_size": 500
            - "node_color": 'white'
            - "edgecolors": "black"
            - "font_family": "monospace"
            - "horizontalalignment": "center"
            - "verticalalignment": "center_baseline"
            - "with_labels": True
        """
        # Overwrite formatting_kwargs with kwargs if they are provided
        formatting_kwargs.update(kwargs)

        G = nx.DiGraph()
        G.add_edges_from(dag.edges())
        G = _format_graph(G)
        ref_layout = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")

        setup_plot(dpi=dpi)
        f, ax = plt.subplots(figsize=figsize)
        if save_to_pdf is not None:
            with PdfPages(save_to_pdf) as pdf:
                _draw_graph_subplot(G, layout=ref_layout,
                                    title=None, ax=ax, **formatting_kwargs)
                pdf.savefig(f, bbox_inches='tight', pad_inches=0)
                plt.close()
        else:
            _draw_graph_subplot(G, layout=ref_layout, title=None,
                                ax=ax, **formatting_kwargs)

    def plot_shap_discrepancies(self, target_name: str, **kwargs):
        assert self.is_fitted_, "Model not fitted yet"
        # X = self.X.drop(target_name, axis=1)
        # y = self.X[target_name].values
        # return self.shaps._plot_discrepancies_for_target(X, y, target_name, **kwargs)
        self.shaps._plot_discrepancies(self.X, target_name, **kwargs)

    def plot_shap_values(self, **kwargs):
        assert self.is_fitted_, "Model not fitted yet"
        plot_args = [(target_name) for target_name in self.feature_names]
        return subplots(self.shaps._plot_shap_summary, *plot_args, **kwargs)


def main2(dataset_name, path="/Users/renero/phd/data/RC3/"):

    ref_graph = graph_from_dot_file(f"{path}{dataset_name}.dot")
    data = pd.read_csv(f"{path}{dataset_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    train = data.sample(frac=0.9, random_state=42)
    test = data.drop(train.index)

    # rex = Rex(model_type=GBTRegressor, explainer=shap.Explainer)
    rex = Rex(
        model_type=NNRegressor, explainer=shap.GradientExplainer,
        correlation_th=0.9, prog_bar=False, verbose=True
    )
    rex.fit_predict(train, test, ref_graph)
    print(rex.score(ref_graph, 'shap'))


if __name__ == "__main__":
    main2('rex_generated_polynew_1')
