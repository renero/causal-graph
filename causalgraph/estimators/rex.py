"""
Main class for the REX estimator.
(C) J. Renero, 2022, 2023
"""

import os
import warnings
from copy import copy
from typing import List, Tuple, Union
import deprecated

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, check_random_state

from causalgraph.common import utils
from causalgraph.common.pipeline import Pipeline
from causalgraph.estimators.knowledge import Knowledge
from causalgraph.explainability.hierarchies import Hierarchies
from causalgraph.explainability.perm_importance import PermutationImportance
from causalgraph.explainability.shapley import ShapEstimator
from causalgraph.explainability.regression_quality import RegQuality
from causalgraph.independence.graph_independence import GraphIndependence
from causalgraph.metrics.compare_graphs import evaluate_graph
from causalgraph.models import GBTRegressor, NNRegressor

np.set_printoptions(precision=4, linewidth=120)
warnings.filterwarnings('ignore')


# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name, W0221:arguments-differ
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches


# TODO:
# - Instead of building a DAG in a single step, build it in several steps, using
#   different samples.

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
    >>> estimator = Rex()
    >>> estimator.fit(X, y)
    """

    def __init__(
            self,
            name: str,
            model_type: str = "nn",
            explainer: str = "gradient",
            tune_model: bool = False,
            correlation_th: float = None,
            corr_method: str = 'spearman',
            corr_alpha: float = 0.6,
            corr_clusters: int = 15,
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
            model_type (str): The type of model to use. Either "nn" for MLP
                or "gbt" for GradientBoostingRegressor.
            explainer (str): The explainer to use for the shap values. The default
                values is "explainer", which uses the shap.Explainer class. Other 
                options are "gradient", which uses the shap.GradientExplainer class,
                and "kernel", which uses the shap.KernelExplainer class.
            tune_model (bool): Whether to tune the model for HPO. Default is False.
            correlation_th (float): The threshold for the correlation. Default is None.
            corr_method (str): The method to use for the correlation.
                Default is "spearman", but it can also be 'pearson', 'kendall or 'mic'.
            corr_alpha (float): The alpha value for the correlation. Default is 0.6.
            corr_clusters (int): The number of clusters to use for the correlation.
                Default is 15.
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
        self.name = name
        self.hpo_study_name = kwargs.get(
            'hpo_study_name', f"{self.name}_{model_type}")
        self.model_type = NNRegressor if model_type == "nn" else GBTRegressor
        self.explainer = explainer
        self._check_model_and_explainer(model_type, explainer)

        self.tune_model = tune_model
        self.correlation_th = correlation_th
        self.corr_method = corr_method
        self.corr_alpha = corr_alpha
        self.corr_clusters = corr_clusters
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

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._fit_desc = "Running Causal Discovery pipeline"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def _check_model_and_explainer(self, model_type, explainer):
        """ Check that the explainer is supported for the model type. """
        if (model_type == "nn" and explainer != "gradient"):
            print(
                f"WARNING: SHAP '{explainer}' not supported for model '{model_type}'. "
                f"Using 'gradient' instead.")
        if (model_type == "gbt" and explainer != "explainer"):
            print(
                f"WARNING: SHAP '{explainer}' not supported for model '{model_type}'. "
                f"Using 'explainer' instead.")

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
            ('models.score', {'X': X}),
            ('root_causes', 'compute_regression_quality'),
            ('shaps', ShapEstimator, {'models': 'models'}),
            ('shaps.fit'),  # , {'exhaustive': False}),
            ('pi', PermutationImportance, {'models': 'models'}),
            ('pi.fit'),
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
            self, prog_bar=self.prog_bar, verbose=self.verbose, silent=self.silent)

        # Overwrite values for prog_bar and verbosity with current pipeline
        #  values, in case predict is called from a loaded experiment
        self.shaps.prog_bar = self.prog_bar
        self.shaps.verbose = self.verbose

        steps = [
            ('G_shap', 'shaps.predict', {'root_causes': 'root_causes'}),
            ('G_pi', 'pi.predict', {'root_causes': 'root_causes'}),
            ('indep', GraphIndependence, {'base_graph': 'G_shap'}),
            ('G_indep', 'indep.fit_predict'),
            ('G_final', 'shaps.adjust', {'graph': 'G_indep'}),
            ('summarize_knowledge', {'ref_graph': ref_graph}),
            ('G_adj', 'adjust_discrepancy', {'dag': 'G_shap'}),
            ('G_shag', 'break_cycles', {'dag': 'G_shap'}),
            ('G_adjnc', 'break_cycles', {'dag': 'G_adj'}),
            # Metrics Generation, here
            ('metrics_shap', 'score', {
             'ref_graph': ref_graph, 'predicted_graph': 'G_shap'}),
            ('metrics_shag', 'score', {
             'ref_graph': ref_graph, 'predicted_graph': 'G_shag'}),
            ('metrics_adj', 'score', {
             'ref_graph': ref_graph, 'predicted_graph': 'G_adj'}),
            ('metrics_adjnc', 'score', {
             'ref_graph': ref_graph, 'predicted_graph': 'G_adjnc'}),
            ('metrics_indep', 'score', {
             'ref_graph': ref_graph, 'predicted_graph': 'G_indep'}),
            ('metrics_final', 'score', {
             'ref_graph': ref_graph, 'predicted_graph': 'G_final'})
        ]
        prediction.run(steps, "Predicting graph")
        if '\\n' in self.G_final.nodes:
            self.G_final.remove_node('\\n')

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
            elif predicted_graph == 'pi':
                pred_graph = self.G_pi
            elif predicted_graph == 'indep':
                pred_graph = self.G_indep
            else:
                raise ValueError(
                    "Predicted graph must be one of 'final', 'shap' or 'indep'.")
        elif isinstance(predicted_graph, nx.DiGraph):
            pred_graph = predicted_graph

        return evaluate_graph(ref_graph, pred_graph, self.feature_names)

    def compute_regression_quality(self):
        """
        Compute the regression quality for each feature in the dataset.
        """
        root_causes = RegQuality.predict(self.models.scoring)
        root_causes = set([self.feature_names[i] for i in root_causes])
        return root_causes

    def summarize_knowledge(self, ref_graph: nx.DiGraph):
        """
        Returns a dataframe with the knowledge about each edge in the graph
        The dataframe is obtained from the Knowledge class.

        Parameters:
        -----------
            ref_graph (nx.DiGraph): The reference graph, or ground truth.
        """
        if ref_graph is None:
            return None

        self.knowledge = Knowledge(self, ref_graph)
        self.learnings = self.knowledge.info()

        return self.learnings

    def break_cycles(self, dag: nx.DiGraph):
        """ Break a cycle in the given DAG.

        Parameters:
        -----------
            dag (nx.DiGraph): The DAG to break the cycle from.
        """
        return utils.break_cycles_if_present(dag, self.learnings, verbose=self.verbose)

    def adjust_discrepancy(self, dag: nx.DiGraph):
        """
        Adjusts the discrepancy in the directed acyclic graph (DAG) by adding new 
        edges based on the goodness-of-fit (GOF) R2 values calculated from the 
        learning data.

        Args:
            dag (nx.DiGraph): The original DAG.

        Returns:
            nx.DiGraph: The adjusted DAG with new edges added based on GOF values.
        """
        G_adj = dag.copy()
        gof = np.zeros((len(self.feature_names), len(self.feature_names)))

        # Filter rows based on the condition that origin and target are linked in DAG
        filtered_rows = self.learnings[
            ~self.learnings.apply(
                lambda row: G_adj.has_edge(row['origin'], row['target']) or
                G_adj.has_edge(row['target'], row['origin']), axis=1)]

        for _, row in filtered_rows.iterrows():
            i = self.feature_names.index(row['origin'])
            j = self.feature_names.index(row['target'])
            gof[i, j] = row['shap_gof']

        new_edges = set()
        # Loop through the i, j positions in the matrix `gof` that are
        # greater than zero.
        for i, j in zip(*np.where(gof > 0)):
            # If the edge (i, j) is not present in the graph, then add it,
            # but only if position (i, j) is greater than position (j, i).
            if not G_adj.has_edge(self.feature_names[i], self.feature_names[j]) and \
                not G_adj.has_edge(self.feature_names[j], self.feature_names[i]) \
                    and gof[i, j] > 0.0 and gof[j, i] > 0.0:
                if gof[j, i] < gof[i, j]:
                    new_edges.add(
                        (self.feature_names[i], self.feature_names[j]))
        # Add the new edges to the graph `G_adj`, if any.
        if new_edges:
            G_adj.add_edges_from(new_edges)

        return G_adj

    def __str__(self):
        return utils.stringfy_object(self)


def custom_main(dataset_name,
                input_path="/Users/renero/phd/data/RC3/",
                output_path="/Users/renero/phd/output/RC3/",
                tune_model: bool = False,
                model_type="nn", explainer="gradient",
                save=False):

    ref_graph = utils.graph_from_dot_file(f"{input_path}{dataset_name}.dot")
    data = pd.read_csv(f"{input_path}{dataset_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    train = data.sample(frac=0.8, random_state=42)
    test = data.drop(train.index)

    rex = Rex(
        name=dataset_name, tune_model=tune_model,
        model_type=model_type, explainer=explainer)
    # rex.fit(train)
    rex.fit_predict(train, test, ref_graph)
    if save:
        where_to = utils.save_experiment(rex.name, output_path, rex)
        print(f"Saved '{rex.name}' to '{where_to}'")

    # rex = load_experiment(dataset_name, output_path)

    # print(rex.score(ref_graph, 'shap'))
    # rex.plot_dags(rex.G_shap, ref_graph)
    # rex.plot_dags(rex.G_pi, ref_graph)


if __name__ == "__main__":
    custom_main('sachs',  model_type="nn", explainer="gradient",
                tune_model=False, save=False)
