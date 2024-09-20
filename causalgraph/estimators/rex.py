"""
Main class for the REX estimator.
(C) J. Renero, 2022, 2023
"""

import os
import warnings
import time
from collections import defaultdict
from copy import deepcopy, copy
from typing import List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from mlforge.mlforge import Pipeline  # type: ignore
from mlforge.progbar import ProgBar   # type: ignore
from rich.progress import Progress
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, check_random_state

from causalgraph.common import utils, DEFAULT_ITERATIVE_TRIALS, DEFAULT_HPO_TRIALS
from causalgraph.estimators.knowledge import Knowledge
from causalgraph.explainability.hierarchies import Hierarchies
from causalgraph.explainability.perm_importance import PermutationImportance
from causalgraph.explainability.regression_quality import RegQuality
from causalgraph.explainability.shapley import ShapEstimator
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


class Rex(BaseEstimator, ClassifierMixin):
    """
    Regression with Explainability (Rex) is a causal inference discovery that
    uses a regression model to predict the outcome of a treatment and uses
    explainability to identify the causal variables.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from causalgraph.estimators.rex import Rex   # doctest: +SKIP
    >>> import numpy as np   # doctest: +SKIP

    >>> dataset_name = 'rex_generated_linear_0'  # doctest: +SKIP
    >>> ref_graph = utils.graph_from_dot_file(f"../data/{dataset_name}.dot")  # doctest: +SKIP
    >>> data = pd.read_csv(f"{input_path}{dataset_name}.csv")  # doctest: +SKIP
    >>> scaler = StandardScaler()  # doctest: +SKIP
    >>> data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)  # doctest: +SKIP
    >>> train = data.sample(frac=0.8, random_state=42)  # doctest: +SKIP
    >>> test = data.drop(train.index)  # doctest: +SKIP

    >>> rex = Rex(   # doctest: +SKIP
        name=dataset_name, tune_model=tune_model,   # doctest: +SKIP
        model_type=model_type, explainer=explainer)   # doctest: +SKIP
    >>> rex.fit_predict(train, test, ref_graph)   # doctest: +SKIP

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
            discrepancy_threshold: float = 0.99,
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
            mean_pi_percentile (float): The percentile for the mean permutation
                importance. Default is 0.8.
            discrepancy_threshold (float): The threshold for the discrepancy.
                Default is 0.99.
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
        self.prior = None
        self.hpo_study_name = kwargs.get(
            'hpo_study_name', f"{self.name}_{model_type}")
        self.model_type = NNRegressor if model_type == "nn" else GBTRegressor
        self.explainer = explainer
        self._check_model_and_explainer(model_type, explainer)

        # Get a copy of kwargs
        self.kwargs = deepcopy(kwargs)

        self.tune_model = tune_model
        self.correlation_th = correlation_th
        self.corr_method = corr_method
        self.corr_alpha = corr_alpha
        self.corr_clusters = corr_clusters
        self.condlen = condlen
        self.condsize = condsize
        self.mean_pi_percentile = mean_pi_percentile
        self.mean_pi_threshold = 0.0
        self.discrepancy_threshold = discrepancy_threshold
        self.prog_bar = prog_bar
        self.verbose = verbose
        self.silent = silent
        self.random_state = random_state

        self.is_fitted_ = False
        self.is_iterative_fitted_ = False

        self.shap_fsize = shap_fsize
        self.dpi = dpi
        self.pdf_filename = pdf_filename

        # This is used to decide how to fit the model in the pipeline
        self.fit_step = 'models.tune_fit' if self.tune_model else 'models.fit'

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._fit_desc = "Running Causal Discovery pipeline"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def fit(self, X, y=None, pipeline: list | str = None):
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
            Returns self
        """
        self.init_fit_time = time.time()

        self.random_state_state = check_random_state(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.feature_names = utils.get_feature_names(X)
        self.feature_types = utils.get_feature_types(X)

        self.X = copy(X)
        self.y = copy(y) if y is not None else None

        # Create the pipeline for the training stages.
        self.fit_pipeline = Pipeline(
            self, description="Fitting models", prog_bar=self.prog_bar,
            verbose=self.verbose, silent=self.silent, subtask=True)
        if pipeline is not None:
            if isinstance(pipeline, list):
                self.fit_pipeline.from_list(pipeline)
            elif isinstance(pipeline, str):
                self.fit_pipeline.from_config(pipeline)
        else:
            steps = [
                ('hierarchies', Hierarchies),
                ('hierarchies.fit'),
                ('models', self.model_type),
                (self.fit_step),
                ('models.score', {'X': X}),
                ('root_causes', 'compute_regression_quality'),
                ('shaps', ShapEstimator, {'models': 'models'}),
                ('shaps.fit'),
                ('pi', PermutationImportance, {
                    'models': 'models', 'discrepancies': 'shaps.shap_discrepancies'}),
                ('pi.fit'),
            ]
            pipeline.from_list(steps)

        n_steps = self.fit_pipeline.len()
        n_steps += self._steps_from_hpo(self.fit_pipeline)
        n_steps += (self._steps_from_iterations(self.fit_pipeline))

        self.fit_pipeline.run(n_steps)
        self.fit_pipeline.close()
        self.is_fitted_ = True

        self.end_fit_time = time.time()
        self.fit_time = self.end_fit_time - self.init_fit_time

        return self

    def predict(self,
                X: pd.DataFrame,
                ref_graph: nx.DiGraph = None,
                prior: List[List[str]] = None,
                pipeline: list | str = None
                ):
        """
        Predicts the causal graph from the given data.

        Parameters
        ----------
        - X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.
        - ref_graph: nx.DiGraph
            The reference graph, or ground truth.
        - prior: str
            The prior to use for building the DAG. This prior is a list of lists
            of node/feature names, ordered according to a temporal structure so that
            the first list contains the first set of nodes to be considered as
            root causes, the second list contains the set of nodes to be
            considered as potential effects of the first set, and the nodes in this
            second list, and so on. The number of lists in the prior is the depth of
            the conditioning sequence. This prior imposes the rule that the nodes in
            the first list are the only ones that can be root causes, and the nodes
            in the following lists cannot be the cause of the nodes in the previous
            lists. If the prior is not provided, the DAG is built without any prior
            information.

        Returns
        -------
        - G_final : nx.DiGraph
            The final graph, after the correction stage.

        Examples
        --------
        In the following example, where four features are used, the prior is
        defined as [['A', 'B'], ['C', 'D']], which means that the first set of
        features to be considered as root causes are 'A' and 'B', and the second
        set of features to be considered as potential effects of the first set are
        'C' and 'D'.

        The resulting DAG cannot contain any edge from 'C' or 'D' to 'A' or 'B'.

            ```python
            rex.predict(X_test, ref_graph, prior=[['A', 'B'], ['C', 'D']])
            ```
        """
        check_is_fitted(self, "is_fitted_")

        self.init_predict_time = time.time()

        # Check that prior is a list of lists and does not contain repeated elements.
        if prior is not None:
            if not isinstance(prior, list):
                raise ValueError("The prior must be a list of lists.")
            if any([len(p) != len(set(p)) for p in prior]):
                raise ValueError("The prior cannot contain repeated elements.")
            self.prior = prior

        # If reference graph is passed, store it in the object.
        if ref_graph is not None:
            self.ref_graph = ref_graph

        # Create a new pipeline for the prediction stages.
        self.predict_pipeline = Pipeline(
            self,
            description="Predicting causal graph",
            prog_bar=self.prog_bar,
            verbose=self.verbose,
            silent=self.silent,
            subtask=True)

        # Overwrite values for prog_bar and verbosity with current pipeline
        #  values, in case predict is called from a loaded experiment
        self.shaps.prog_bar = self.prog_bar
        self.shaps.verbose = self.verbose

        if pipeline is not None:
            if isinstance(pipeline, list):
                self.predict_pipeline.from_list(pipeline)
            elif isinstance(pipeline, str):
                self.predict_pipeline.from_config(pipeline)
        else:
            steps = [
                # DAGs construction
                ('G_shap', 'shaps.predict', {
                    'root_causes': 'root_causes', 'prior': prior}),
                ('G_rho', 'dag_from_discrepancy', {
                    'discrepancy_upper_threshold': self.discrepancy_threshold,
                    "verbose": False}),
                ('G_adj', 'adjust_discrepancy', {'dag': 'G_shap'}),
                ('G_pi', 'pi.predict', {
                    'root_causes': 'root_causes', 'prior': prior}),
                ('indep', GraphIndependence, {'base_graph': 'G_shap'}),
                ('G_indep', 'indep.fit_predict'),
                ('G_final', 'shaps.adjust', {'graph': 'G_indep'}),

                # Knowledge Summarization
                ('summarize_knowledge', {'ref_graph': ref_graph}),

                # Metrics Generation, here
                ('metrics_shap', 'score', {
                    'ref_graph': ref_graph, 'predicted_graph': 'G_shap'}),
                ('metrics_rho', 'score', {
                    'ref_graph': ref_graph, 'predicted_graph': 'G_rho'}),
                ('metrics_adj', 'score', {
                    'ref_graph': ref_graph, 'predicted_graph': 'G_adj'}),
                ('metrics_indep', 'score', {
                    'ref_graph': ref_graph, 'predicted_graph': 'G_indep'}),
                ('metrics_final', 'score', {
                    'ref_graph': ref_graph, 'predicted_graph': 'G_final'})
            ]
            self.predict_pipeline.from_list(steps)
        self.predict_pipeline.run()
        # Check if "G_final" exists in this object (self)
        if 'G_final' in self.__dict__:
            if '\\n' in self.G_final.nodes:
                self.G_final.remove_node('\\n')
        self.predict_pipeline.close()

        self.end_predict_time = time.time()
        self.predict_time = self.end_predict_time - self.init_predict_time

        return self

    def fit_predict(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            ref_graph: nx.DiGraph,
            prior: List[List[str]] = None):
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
        self.predict(test, ref_graph, prior)
        return self

    def _build_iterative_adjacency_matrix(
            self,
            X: pd.DataFrame,
            num_iterations: int = DEFAULT_ITERATIVE_TRIALS,
            sampling_split: float = 0.5,
            prior: list = None,
            random_state: int = 1234) -> np.ndarray:
        """
        Performs iterative prediction on the given directed acyclic graph (DAG)
        and adjacency matrix.

        Parameters:
            X (pd.DataFrame): The input data.
            num_iterations (int): The number of iterations to perform. Defaults to 10.
            sampling_split (float): The proportion of the data to use for
                training. Defaults to 0.5.
            prior (list): The prior knowledge. Defaults to None.
            dag_names (list): A list of names for the DAG. Defaults to ['shap', 'rho',
                'adjusted', 'perm_imp', 'indep', 'final'].
            random_state (int): The random state. Defaults to 1234.

        Returns:
            dict: The predicted adjacency matrices, normalized by the number
                of iterations.
        """

        # Assert that 'shaps' exists and has been fit
        check_is_fitted(self, "is_fitted_")

        iter_adjacency_matrix = np.zeros(
            (self.n_features_in_, self.n_features_in_))

        for iter in range(num_iterations):
            data_sample = X.sample(frac=sampling_split,
                                   random_state=iter*random_state)
            self.shaps.fit(data_sample)
            dag = self.shaps.predict(data_sample, prior=prior)
            iter_adjacency_matrix += utils.graph_to_adjacency(
                dag, self.feature_names)

        iter_adjacency_matrix = iter_adjacency_matrix / num_iterations

        return iter_adjacency_matrix

    def iterative_predict(
            self,
            X: pd.DataFrame,
            ref_graph: nx.DiGraph,
            num_iterations: int = DEFAULT_ITERATIVE_TRIALS,
            sampling_split: float = 0.2,
            prior: list = None,
            random_state: int = 1234,
            tolerance: Union[float, str] = 'auto',
            key_metric: str = 'f1',
            direction: str = 'maximize') -> nx.DiGraph:
        """
        Finds the best tolerance value for the iterative predict method by iterating
        over different tolerance values and selecting the one that gives the best
        `key_metric` with respect to the reference graph.

        Parameters
        ----------
        ref_graph : nx.DiGraph
            The reference graph to evaluate the F1 score against.
        target : str, optional
            The target DAG to evaluate. Defaults to 'shap'.
            Possible values: 'shap', 'rho', 'adjusted', 'perm_imp', 'indep', and 'final'
        key_metric : str, optional
            The key metric to evaluate. Defaults to 'f1'.
            Possible values: 'f1', 'precision', 'recall', 'shd', sid', 'aupr',
            'Tp', 'Tn', 'Fp', 'Fn'    '
        direction : str, optional
            The direction of the key metric. Defaults to 'maximize'.
            Possible values: 'maximize' or 'minimize'

        Returns
        -------
        nx.DiGraph : The best DAG found by the iterative predict method.
        """
        if ref_graph is None:
            raise ValueError("ref_graph must be specified")
        if direction != 'maximize' and direction != 'minimize':
            raise ValueError("direction must be 'maximize' or 'minimize'")

        iter_adjacency_matrix = self._build_iterative_adjacency_matrix(
            X, num_iterations, sampling_split, prior, random_state)

        if tolerance == 'auto':
            # This lambda expression is used to compare values, depending on the direction
            _is_better_value = {
                'maximize': lambda x, y: x >= y,
                'minimize': lambda x, y: x < y
            }[direction]

            reference_key_metric = -100000.0 if direction == 'maximize' else +100000.0
            self.iterative_metrics = []
            self.tolerance = 0.0
            for tol in np.arange(0.1, 1.0, 0.05):
                dag = self.build_dag_from_iter_adj_matrix(
                    iter_adjacency_matrix, tolerance=tol)

                metric = evaluate_graph(ref_graph, dag)
                self.iterative_metrics.append(metric)
                value_obtained = getattr(metric, key_metric)
                if _is_better_value(value_obtained, reference_key_metric):
                    reference_key_metric = value_obtained
                    self.tolerance = tol

            if self.verbose:
                print(f"Best tolerance: {self.tolerance:.2f}, "
                      f"{key_metric}: {reference_key_metric:.4f}")

        # Now, predict with selected tolerance
        return self.build_dag_from_iter_adj_matrix(
            iter_adjacency_matrix, tolerance=self.tolerance)

    def build_dag_from_iter_adj_matrix(
            self,
            iter_adjacency_matrix: np.ndarray,
            tolerance: float = 0.3) -> dict:
        """
        Performs iterative prediction on the given directed acyclic graph (DAG)
        and adjacency matrix. Prediction is based on the adjacency matrices previously
        computed, which are here normalized and filtered, so values below tolerance
        are set to zero. This way, only those edges present in more than "tolerance"
        percent of the iterations are kept.

        Parameters:
            dag (dict): The input DAG.
            num_iterations (int): The number of iterations to perform. Defaults to 10.
            tolerance (float): The tolerance value for filtering the adjacency matrix.
                Defaults to 0.3.
            inplace (bool): Whether to store the predicted DAG in the object.
                Defaults to True. If False, the predicted DAG is returned.

        Returns:
            dict: The predicted DAG.
        """
        assert isinstance(iter_adjacency_matrix, np.ndarray), \
            "Adjacency must be a 2D numpy array"
        assert isinstance(tolerance, (int, float)
                          ), "Tolerance must be a number"
        assert (tolerance <= 1.0) and (tolerance >= 0.0), \
            "Tolerance must be range between 0.0 and 1.0"

        filtered_matrix = self._filter_adjacency_matrix(
            iter_adjacency_matrix, tolerance)

        dag = utils.graph_from_adjacency(filtered_matrix, self.feature_names)

        return dag

    def custom_pipeline(self, steps):
        """
        Execute a pipeline formed by a list of custom steps previously defined.

        Parameters
        ----------
        steps : list
            A list of tuples with the steps to add to the pipeline.

        Returns
        -------
        self : object
            Returns self.
        """
        # Create the pipeline for the training stages.
        pipeline = Pipeline(self, description="Custom pipeline", prog_bar=self.prog_bar,
                            verbose=self.verbose, silent=self.silent, subtask=True)
        pipeline.from_list(steps)
        pipeline.run()
        pipeline.close()

        return self

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
        return utils.break_cycles_if_present(
            dag, self.shaps.shap_discrepancies, self.prior, verbose=self.verbose)

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

        # Loop through all pairs of nodes where the edge is not present in the graph.
        for origin in self.feature_names:
            for target in self.feature_names:
                if origin == target:
                    continue
                if not G_adj.has_edge(origin, target) and not G_adj.has_edge(target, origin):
                    i = self.feature_names.index(origin)
                    j = self.feature_names.index(target)
                    gof[i, j] = self.shaps.shap_discrepancies[target][origin].shap_gof

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

        G_adj = self.break_cycles(G_adj)

        return G_adj

    def dag_from_discrepancy(
            self,
            discrepancy_upper_threshold: float = 0.99,
            verbose: bool = False) -> nx.DiGraph:
        """
        Build a directed acyclic graph (DAG) from the discrepancies in the SHAP values.
        The discrepancies are calculated as 1.0 - GoodnessOfFit, so that a low
        discrepancy means that the GoodnessOfFit is close to 1.0, which means that
        the SHAP values are similar.

        Parameters:
        -----------
            discrepancy_upper_threshold (float): The threshold for the discrepancy.
                Default is 0.99, which means that the GoodnessOfFit must be
                at least 0.01.

        Returns:
        --------
            nx.DiGraph: The directed acyclic graph (DAG) built from the discrepancies.
        """
        if verbose:
            print("-----\ndag_from_discrepancies()")

        # Find out what pairs of features have low discrepancy, and add them as edges.
        # A low discrepancy means that 1.0 - GoodnesOfFit is lower than the threshold.
        low_discrepancy_edges = defaultdict(list)
        if verbose:
            print('    ' + ' '.join([f"{f:^5s}" for f in self.feature_names]))
        for child in self.feature_names:
            if verbose:
                print(f"{child}: ", end="")
            for parent in self.feature_names:
                if child == parent:
                    if verbose:
                        print("  X  ", end=" ")
                    continue
                discrepancy = 1. - \
                    self.shaps.shap_discrepancies[child][parent].shap_gof
                if verbose:
                    print(f"{discrepancy:+.2f}", end=" ")
                if discrepancy < discrepancy_upper_threshold:
                    if low_discrepancy_edges[child]:
                        low_discrepancy_edges[child].append(parent)
                    else:
                        low_discrepancy_edges[child] = [parent]
            if verbose:
                print()

        # Build a DAG from the connected features.
        self.G_rho = utils.digraph_from_connected_features(
            self.X, self.feature_names, self.models, low_discrepancy_edges,
            root_causes=self.root_causes, prior=self.prior, verbose=verbose)

        self.G_rho = self.break_cycles(self.G_rho)

        return self.G_rho

    def get_prior_from_ref_graph(self) -> List[List[str]]:
        """
        Get the prior from a reference graph.

        Returns
        -------
        List[List[str]]
            A list of two lists of nodes. The first list contains the root nodes,
            and the second list contains the rest of the nodes. The root nodes are
            the nodes with no incoming edges, and the rest of the nodes are the
            ones with at least one incoming edge.
        """
        ref_graph_file = self.name.replace(f"_{self.model_type}", "")
        ref_graph_file = f"{ref_graph_file}.dot"
        ref_graph = utils.graph_from_dot_file(os.path.join(
            os.path.expanduser("~/phd/data/RC4"), ref_graph_file))

        root_nodes = [node for node,
                      degree in ref_graph.in_degree() if degree == 0]
        return [root_nodes, [node for node in ref_graph.nodes if node not in root_nodes]]

    def _steps_from_hpo(self, fit_steps) -> int:
        """
        Update the number of trials for the HPO.

        Parameters
        ----------
        fit_steps: Pipeline
            The pipeline where looking for the number of trials.

        Returns
        -------
        int
        """
        if fit_steps.contains_method('tune_fit', exact_match=False):
            if 'hpo_n_trials' in self.kwargs:
                return self.kwargs['hpo_n_trials'] - 1
            else:
                if fit_steps.contains_argument('hpo_n_trials'):
                    return fit_steps.get_argument_value('hpo_n_trials')
                else:
                    return DEFAULT_HPO_TRIALS

        return 0

    def _steps_from_iterations(self, fit_steps) -> int:
        """
        Check if the pipeline contains a stage called iterative_fit and
        retrieve the number of iterations.

        Parameters
        ----------
        fit_steps: Pipeline
            The pipeline where looking for the number of trials.

        Returns
        -------
        int
        """
        num_iterations = 0
        num_iterative_steps = fit_steps.contains_method(
            'iterative_predict', exact_match=False)
        if num_iterative_steps > 0:
            if fit_steps.contains_argument('num_iterations'):
                all_iterations = fit_steps.all_argument_values(
                    'num_iterations')
                if all_iterations != []:
                    return sum(all_iterations)
            else:
                num_iterations += DEFAULT_ITERATIVE_TRIALS

        return num_iterations

    def _filter_adjacency_matrix(
            self,
            adjacency_matrix: np.ndarray,
            tolerance: float) -> np.ndarray:
        """
        Given an adjacency matrix, return a filtered version of it, where
        all weights with absolute value less than the tolerance are
        set to zero.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            The adjacency matrix.
        tolerance : float
            The tolerance value.

        Returns
        -------
        np.ndarray
            The filtered adjacency matrix.
        """
        filtered_adjacency = adjacency_matrix.copy()
        filtered_adjacency[np.abs(filtered_adjacency) < tolerance] = 0
        return filtered_adjacency

    def _check_model_and_explainer(self, model_type, explainer):
        """ Check that the explainer is supported for the model type. """
        if (model_type == "nn" and explainer != "gradient"):
            print(
                f"WARNING: SHAP '{explainer}' not supported for model '{model_type}'. "
                f"Using 'gradient' instead.")
            self.explainer = "gradient"
        if (model_type == "gbt" and explainer != "explainer"):
            print(
                f"WARNING: SHAP '{explainer}' not supported for model '{model_type}'. "
                f"Using 'explainer' instead.")
            self.explainer = "explainer"

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

    def __str__(self):
        return utils.stringfy_object(self)

    def verbose_on(self):
        self.verbose = True
        self.silent = False
        self.prog_bar = False
        self.shaps.verbose = True
        self.shaps.prog_bar = False
        self.hierarchies.verbose = True
        self.hierarchies.prog_bar = False
        self.pi.verbose = True
        self.pi.prog_bar = False
        self.models.verbose = True
        self.models.prog_bar = False

    def verbose_off(self):
        self.verbose = False
        self.prog_bar = True
        self.shaps.verbose = False
        self.shaps.prog_bar = True
        self.hierarchies.verbose = False
        self.hierarchies.prog_bar = True
        self.pi.verbose = False
        self.pi.prog_bar = True
        self.models.verbose = False
        self.models.prog_bar = True


def custom_main(dataset_name,
                input_path="/Users/renero/phd/data/RC4/",
                output_path="/Users/renero/phd/output/RC4/",
                scale_data: bool = False,
                tune_model: bool = False,
                model_type="nn", explainer="gradient",
                save=False):

    def get_prior(ref_graph):
        root_nodes = [n for n, d in ref_graph.in_degree() if d == 0]
        return [root_nodes, [n for n in ref_graph.nodes if n not in root_nodes]]

    ref_graph = utils.graph_from_dot_file(f"{input_path}{dataset_name}.dot")
    data = pd.read_csv(f"{input_path}{dataset_name}.csv")
    if scale_data:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    train = data.sample(frac=0.8, random_state=42)
    test = data.drop(train.index)

    rex = Rex(
        name=dataset_name, tune_model=tune_model,
        model_type=model_type, explainer=explainer, hpo_n_trials=1)

    # rex.fit_predict(train, test, ref_graph, prior=get_prior(ref_graph))
    rex.fit(data, pipeline=".fast_fit_pipeline.yaml")

    # prior = get_prior(ref_graph)
    prior = rex.get_prior_from_ref_graph()
    rex.predict(data, ref_graph, prior=prior,
                pipeline=".fast_predict_pipeline.yaml")

    if save:
        where_to = utils.save_experiment(rex.name, output_path, rex)
        print(f"Saved '{rex.name}' to '{where_to}'")


def prior_main(dataset_name,
               input_path="/Users/renero/phd/data/RC3/",
               output_path="/Users/renero/phd/output/RC4/",
               model_type="nn"):
    from causalgraph.common.notebook import Experiment

    experiment_name = f"{dataset_name}_{model_type}"
    data = pd.read_csv(f"{input_path}{dataset_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    experiment = Experiment(dataset_name, output_path=output_path).load()
    print(f"Loaded {experiment_name} from {experiment.output_path}")

    def get_prior(ref_graph):
        root_nodes = [n for n, d in ref_graph.in_degree() if d == 0]
        return [root_nodes, [n for n in ref_graph.nodes if n not in root_nodes]]

    # train = exp_prior.rex.X
    experiment.rex.verbose = True
    experiment.rex.G_shap = experiment.rex.shaps.predict(
        experiment.rex.X, prior=get_prior(experiment.ref_graph))


if __name__ == "__main__":
    custom_main('generated_10vars_linear_0',
                input_path="/Users/renero/phd/data/RC4/",
                model_type="nn",
                explainer="gradient",
                tune_model=False,
                save=False)
    # prior_main('rex_generated_gp_add_5')
