"""
A module to run experiments with the causalgraph package, and simplify the
process of loading and saving experiments in notebooks.

Example:
    >> from causalgraph.common.notebook import Experiment
    >> experiment = Experiment("linear", csv_filename="linear.csv")
    >> rex = experiment.load()

(C) 2023, 2024 J. Renero
"""

import glob
import os
import time
import warnings
from collections import defaultdict
from os import path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from causalgraph.common import plot, utils
from causalgraph.common import utils
from causalgraph.estimators.cam.cam import CAM
from causalgraph.estimators.fci.fci import FCI
from causalgraph.estimators.ges.ges import GES
from causalgraph.estimators.lingam.lingam import DirectLiNGAM as LiNGAM
from causalgraph.estimators.pc.pc import PC
from causalgraph.estimators.notears.notears import NOTEARS
from causalgraph.estimators.rex import Rex
from causalgraph.metrics.compare_graphs import evaluate_graph

warnings.filterwarnings('ignore')


# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches


global_metric_types = [
    'mlp', 'gbt', 'intersection', 'union',
    'mlp_adj', 'gbt_adj', 'intersection_adj', 'union_adj',
    'union_all', 'int_indep', 'int_final', 'union_indep', 'union_final']
global_nc_metric_types = [
    'mlp_nc', 'gbt_nc', 'intersection_nc', 'union_nc',
    'mlp_adjnc', 'gbt_adjnc', 'intersection_adjnc', 'union_adjnc',
    'union_all_nc', 'int_indep', 'int_final', 'union_indep', 'union_final']
metric_labels = {
    'mlp': 'DFN',
    'gbt': 'GBT',
    'intersection': r'$\textrm{DAG}_\cap$',
    'union': r'$\textrm{DAG}_\cup$',
    'union_all': 'all',
    'int_indep': '∩i',
    'int_final': '∩f',
    'union_indep': '∪i',
    'union_final': '∪f',
    'mlp_nc': 'DFN',
    'gbt_nc': 'GBT',
    'intersection_nc': '∩',
    'union_nc': '∪',
    'union_all_nc': 'all'
}
score_titles = {
    'f1': r'$\textrm{F1}$',
    'precision': r'$\textrm{Precision}$',
    'recall': r'$\textrm{Recall}$',
    'aupr': r'$\textrm{AuPR}$',
    'Tp': r'$\textrm{TP}$',
    'Tn': r'$\textrm{TN}$',
    'Fp': r'$\textrm{FP}$',
    'Fn': r'$\textrm{FN}$',
    'shd': r'$\textrm{SHD}$',
    'sid': r'$\textrm{SID}$',
    'n_edges': r'$\textrm{Nr. Edges}$',
    'ref_n_edges': r'$\textrm{Edges in Ground Truth}$',
    'diff_edges': r'$\textrm{Diff. Edges}$',
}
method_labels = {
    'nn': r'$\textrm{R\textsc{e}X}_{\textrm{\tiny DFN}}$',
    'rex_mlp': r'$\textrm{R\textsc{e}X}_{\textrm{\tiny DFN}}$',
    'nn_adj': r'$\textrm{R\textsc{e}X}_{\textrm{\tiny DFN}}^{\textrm{\tiny adj}}$',
    'rex_mlp_adj': r'$\textrm{R\textsc{e}X}_{\textrm{\tiny DFN}}^{\textrm{\tiny adj}}$',
    'gbt': r'$\textrm{R\textsc{e}X}_{\textrm{\tiny GBT}}$',
    'rex_gbt': r'$\textrm{R\textsc{e}X}_{\textrm{\tiny GBT}}$',
    'gbt_adj': r'$\textrm{R\textsc{e}X}_{\textrm{\tiny GBT}}^{\textrm{\tiny adj}}$',
    'rex_gbt_adj': r'$\textrm{R\textsc{e}X}_{\textrm{\tiny GBT}}^{\textrm{\tiny adj}}$',
    'union': r'$\textrm{R\textsc{e}X}_{\cup}$',
    'rex_union': r'$\textrm{R\textsc{e}X}_{\cup}$',
    'union_adj': r'$\textrm{R\textsc{e}X}_{\cup}^{\textrm{\tiny adj}}$',
    'rex_union_adj': r'$\textrm{R\textsc{e}X}_{\cup}^{\textrm{\tiny adj}}$',
    'rex_union_adjnc': r'$\textrm{R\textsc{e}X}_{\cup}^{\textrm{\tiny adj}}$',
    'intersection': r'$\textrm{R\textsc{e}X}_{\cap}$',
    'rex_intersection': r'$\textrm{R\textsc{e}X}_{\cap}$',
    'intersection_adj': r'$\textrm{R\textsc{e}X}_{\cap}^{\textrm{\tiny adj}}$',
    'rex_intersection_adj': r'$\textrm{R\textsc{e}X}_{\cap}^{\textrm{\tiny adj}}$',
    'rex_intersection_adjnc': r'$\textrm{R\textsc{e}X}_{\cap}^{\textrm{\tiny adj}}$',
    'pc': r'$\textrm{PC}$',
    'fci': r'$\textrm{FCI}$',
    'ges': r'$\textrm{GES}$',
    'lingam': r'$\textrm{LiNGAM}$',
    'cam': r'$\textrm{CAM}$',
    'notears': r'$\textrm{NOTEARS}$',
    'G_pc': r'$\textrm{PC}$',
    'G_fci': r'$\textrm{FCI}$',
    'G_ges': r'$\textrm{GES}$',
    'G_lingam': r'$\textrm{LiNGAM}$',
    'G_cam': r'$\textrm{CAM}$',
    'G_notears': r'$\textrm{NOTEARS}$',
    'un_G_iter': r'$\textrm{R\textsc{e}X}$'
}
estimators = {
    'rex': Rex,
    'fci': FCI,
    'pc': PC,
    'lingam': LiNGAM,
    'ges': GES,
    'cam': CAM,
    'notears': NOTEARS
}
method_names = ['pc', 'fci', 'ges', 'lingam', 'cam', 'notears']
synth_data_types = ['linear', 'polynomial', 'gp_add', 'gp_mix', 'sigmoid_add']
synth_data_labels = ['Linear', 'Polynomial',
                     'Gaussian(add)', 'Gaussian(mix)', 'Sigmoid(add)']
metric_columns = ['method', 'data_type', 'f1', 'precision',
                  'recall', 'aupr', 'Tp', 'Tn', 'Fp', 'Fn', 'shd', 'sid',
                  'n_edges', 'ref_n_edges', 'diff_edges', 'name']
RAW_DAG_NAMES = ['G_shap', 'G_prior', 'G_iter', 'G_iter_prior']
COMBINED_DAG_NAMES = ['un_G_shap', 'in_G_shap',
                      'un_G_prior', 'in_G_prior',
                      'un_G_iter', 'in_G_iter',
                      'un_G_iter_prior', 'in_G_iter_prior']


def list_files(input_pattern: str, where: str) -> list:
    """
    List all the files in the input path matching the input pattern

    Parameters:
    -----------
    input_pattern : str
        The pattern to match the files
    where : str
        The path to use to look for the files matching the pattern
    """
    input_files = glob.glob(os.path.join(
        where, input_pattern))
    input_files = sorted([os.path.basename(os.path.splitext(f)[0])
                         for f in input_files])

    assert len(input_files) > 0, \
        f"No files found in {where} matching <{input_pattern}>"

    return sorted(list(set(input_files)))


class BaseExperiment:
    """
    Base class for experiments.

    Args:
    input_path (str): The path to the input data.
    output_path (str): The path to save the experiment output.
    train_anyway (bool, optional): Whether to train the model even if the experiment exists. Defaults to False.
    save_anyway (bool, optional): Whether to save the experiment even if it exists. Defaults to False.
    train_size (float, optional): The proportion of data to use for training. Defaults to 0.9.
    random_state (int, optional): The random state for reproducibility. Defaults to 42.
    verbose (bool, optional): Whether to display verbose output. Defaults to False.
    """

    def __init__(
            self,
            input_path: str,
            output_path: str,
            train_anyway: bool = False,
            save_anyway: bool = False,
            scale: bool = False,
            train_size: float = 0.9,
            random_state: int = 42,
            verbose: bool = False):

        self.input_path = input_path
        self.output_path = output_path
        self.train_anyway = train_anyway
        self.save_anyway = save_anyway
        self.scale = scale
        self.train_size = train_size
        self.random_state = random_state
        self.verbose = verbose

        # Display Options
        np.set_printoptions(precision=4, linewidth=100)
        pd.set_option('display.precision', 4)
        pd.set_option('display.float_format', '{:.4f}'.format)

    def prepare_experiment_input(
            self,
            experiment_filename,
            csv_filename=None,
            dot_filename=None):
        """
        - Loads the data and
        - splits it into train and test,
        - scales it
        - loads the reference graph from the dot file, which has to be named
          as the experiment file, with the .dot extension
        """
        self.experiment_name = path.splitext(
            path.basename(experiment_filename))[0]
        if csv_filename is None:
            csv_filename = f"{path.join(self.input_path, self.experiment_name)}.csv"
        if dot_filename is None:
            dot_filename = f"{path.join(self.input_path, self.experiment_name)}.dot"

        self.data = pd.read_csv(csv_filename)
        self.data = self.data.apply(pd.to_numeric, downcast='float')
        if self.scale:
            scaler = StandardScaler()
            self.data = pd.DataFrame(
                scaler.fit_transform(self.data), columns=self.data.columns)
            self.train_data = self.data.sample(
                frac=self.train_size, random_state=self.random_state)
            self.test_data = self.data.drop(self.train_data.index)
        else:
            self.train_data = self.data.sample(
                frac=self.train_size, random_state=self.random_state)
            self.test_data = self.data.drop(self.train_data.index)

        self.ref_graph = utils.graph_from_dot_file(dot_filename)

    def experiment_exists(self, name):
        """Checks whether the experiment exists in the output path"""
        return os.path.exists(
            os.path.join(self.output_path, f"{os.path.basename(name)}.pickle"))

    def decide_what_to_do(self):
        """
        Decides whether to load or train the model, and whether to save the
        experiment or not. The decision is based on the following rules:

        - If the experiment exists, it will be loaded unless train_anyway is True
        - If the experiment does not exist, it will be trained unless train_anyway
            is False
        - If the experiment exists, it will be saved unless save_anyway is False
        - If the experiment does not exist, it will be saved unless save_anyway
            is False
        """
        experiment_exists = self.experiment_exists(self.experiment_name)
        if experiment_exists:
            self.load_experiment = True and not self.train_anyway
        else:
            self.load_experiment = False and not self.train_anyway
        self.save_experiment = (
            True if self.load_experiment is False else False) or self.save_anyway

        if self.verbose:
            if self.load_experiment:
                print(
                    f"      ↳ Experiment '{self.experiment_name}' will be LOADED")
            else:
                print(
                    f"      ↳ Experiment '{self.experiment_name}' will be TRAINED")

        self.save_experiment = True

    def list_files(self, input_pattern, where='input') -> list:
        """
        List all the files in the input path matching the input pattern
        """
        where = self.input_path if where == 'input' else self.output_path
        input_files = glob.glob(os.path.join(
            where, input_pattern))
        input_files = sorted([os.path.splitext(f)[0] for f in input_files])

        assert len(input_files) > 0, \
            f"No files found in {where} matching <{input_pattern}>"

        return sorted(list(set(input_files)))

    def create_estimator(self, estimator_name: str, name: str, **kwargs):
        """
        Dynamically creates an instance of a class based on the estimator name.

        Args:
        estimator_name (str): The name of the estimator (key in the 'estimators'
            dictionary).
        name (str): The name of the estimator instance.
        *args: Variable length argument list to be passed to the class constructor.
        **kwargs: Arbitrary keyword arguments to be passed to the class constructor.

        Returns:
        An instance of the specified class, or None if the class does not exist.
        """
        estimator_class = estimators.get(estimator_name)
        if estimator_class is None:
            print(f"Estimator '{estimator_name}' not found.")
            return None

        # Special case: when estimator is ReX, model_type needs also to be passed to
        # the constructor
        if estimator_name == 'rex':
            kwargs['model_type'] = self.model_type

        return estimator_class(name=name, **kwargs)


class Experiment(BaseExperiment):
    """
    Represents an experiment for causal graph analysis.

    Methods:
        load: Loads the experiment data.
        fit: Fits the experiment data.
        save: Saves the experiment data.
    """

    estimator_name = None

    def __init__(
        self,
        experiment_name,
        csv_filename: str = None,
        dot_filename: str = None,
        model_type: str = 'nn',
        input_path="/Users/renero/phd/data/RC4/",
        output_path="/Users/renero/phd/output/RC4/",
        train_size: float = 0.9,
        random_state: int = 42,
        verbose=False
    ):
        """
        Initializes a new instance of the Experiment class.

        Args:
            experiment_name (str): The name of the experiment.
            csv_filename (str, optional): The filename of the CSV file containing
                the data. Defaults to None.
            dot_filename (str, optional): The filename of the DOT file containing
                the causal graph. Defaults to None.
            model_type (str, optional): The type of model to use. Defaults to 'nn'.
                Other options are: 'gbt', 'pc', 'fci', 'ges' and 'lingam'.
            input_path (str, optional): The path to the input data.
                Defaults to "/Users/renero/phd/data/RC4/".
            output_path (str, optional): The path to save the output.
                Defaults to "/Users/renero/phd/output/RC4/".
            train_size (float, optional): The proportion of data to use for training.
                Defaults to 0.9.
            random_state (int, optional): The random seed for reproducibility.
                Defaults to 42.
            verbose (bool, optional): Whether to print verbose output.
                Defaults to False.
        """

        super().__init__(
            input_path, output_path, train_size=train_size,
            random_state=random_state, verbose=verbose)
        self.model_type = self._check_model_type(model_type)
        self.is_fitted = False
        self.verbose = verbose

        # Prepare the input
        self.prepare_experiment_input(
            experiment_name, csv_filename, dot_filename)

    def _check_model_type(self, model_type):
        """
        Checks if the model type is valid.
        """
        model_type = model_type.lower()
        if model_type in ['dnn', 'nn']:
            model_type = 'nn'
        elif model_type == 'gbt':
            model_type = 'gbt'
        else:
            raise ValueError(
                f"Model type '{model_type}' not supported. "
                f"Supported options are: 'nn', 'gbt', 'pc', 'fci', 'ges' and 'lingam'.")

        return model_type

    def fit(self, estimator_name='rex', **kwargs):
        """
        Fits the experiment data.

        Args:
            **kwargs: Additional keyword arguments to pass to the Rex constructor.

        Returns:
            Rex: The fitted experiment data.
        """
        self.estimator_name = estimator_name
        kwargs['model_type'] = self.model_type

        estimator_object = self.create_estimator(
            estimator_name, name=self.experiment_name, **kwargs)

        pipeline = kwargs.pop('pipeline') if 'pipeline' in kwargs else None

        estimator_object.fit(
            self.train_data, y=self.test_data, pipeline=pipeline)

        setattr(self, estimator_name, estimator_object)
        self.is_fitted = True

        return self

    def predict(self, estimator='rex', **kwargs):
        """
        Predicts with the experiment data.

        Args:
            **kwargs: Additional keyword arguments to pass to the `predict()` method

        Returns:
            Rex: The fitted experiment data.
        """
        estimator = getattr(self, self.estimator_name)
        estimator.predict(self.train_data, **kwargs)

        return self

    def fit_predict(self, estimator='rex', **kwargs):
        """
        Fits and predicts with the experiment data.

        Args:
            **kwargs: Additional keyword arguments to pass to the Rex constructor.

        Returns:
            Rex: The fitted experiment data.
        """
        start_time = time.time()
        self.estimator_name = estimator
        estimator_object = self.create_estimator(
            estimator, name=self.experiment_name, **kwargs)
        estimator_object.fit_predict(
            self.train_data, self.test_data, self.ref_graph)
        setattr(self, estimator, estimator_object)
        end_time = time.time()
        self.fit_predict_time = end_time - start_time

        return self

    def load(self, exp_name=None) -> "Experiment":
        """
        Loads the experiment data.

        Args:
            exp_name (str, optional): The name of the experiment to load.
            If None, loads the current experiment. Defaults to None.

        Returns:
            Rex: The loaded experiment data.
        """

        if exp_name is None:
            exp_name = self.experiment_name

        if self.model_type:
            exp_object = utils.load_experiment(
                f"{exp_name}_{self.model_type}", self.output_path)
        else:
            exp_object = utils.load_experiment(exp_name, self.output_path)

        # A priori, I don't know which estimator was used to train the experiment
        # so I have to check the type of the object
        if isinstance(exp_object, Rex):
            self.estimator_name = 'rex'
        elif isinstance(exp_object, PC):
            self.estimator_name = 'pc'
        elif isinstance(exp_object, LiNGAM):
            self.estimator_name = 'lingam'
        elif isinstance(exp_object, GES):
            self.estimator_name = 'ges'
        elif isinstance(exp_object, FCI):
            self.estimator_name = 'fci'
        elif isinstance(exp_object, CAM):
            self.estimator_name = 'cam'
        else:
            raise ValueError(
                f"Estimator '{exp_name}' not recognized.")

        setattr(self, self.estimator_name, exp_object)
        setattr(self, 'estimator', exp_object)

        if self.verbose:
            print(f"Loaded '{exp_name}' ({self.model_type.upper()}) "
                  f"from '{self.output_path}'")
            fit_time = utils.format_time(self.rex.fit_time)
            predict_time = utils.format_time(self.rex.predict_time)
            print(f"This model took {fit_time[0]:.1f}{fit_time[1]}. to fit, and "
                  f"{predict_time[0]:.1f}{predict_time[1]}. to build predicted DAGs")

        return self

    def save(self, exp_name=None, overwrite: bool = False):
        """
        Saves the experiment data.

        Args:
        -----
        - exp_name (str, optional): The name to save the experiment as.
            If None, uses the experiment name. Defaults to None.
        - overwrite (bool, optional): Whether to overwrite an existing
            experiment with the same name. Defaults to False.
        """
        save_as = exp_name if exp_name is not None else self.experiment_name
        where_to = utils.save_experiment(
            f"{save_as}_{self.model_type}",
            self.output_path, getattr(self, self.estimator_name),
            overwrite)

        if self.verbose:
            print(f"Saved '{self.experiment_name}' to '{where_to}'")

        return where_to


def get_combined_metrics(subtype: str, where: str = None) -> dict:
    """
    Obtain the metrics for all the experiments matching the input pattern

    Parameters
    ----------
    subtype : str
        The subtype of the experiment, e.g. "linear" or "polynomial"
    where : str
        The path to use to look for the files matching the pattern

    Returns
    -------
    dict
        A dictionary with the metrics for all the experiments
    """
    if where is None:
        where = "/Users/renero/phd/output/RC4/"
    files = list_files(f"rex_generated_{subtype}_*_nn.pickle", where=where)

    metrics = defaultdict(list)
    for exp_name in files:
        exp_name = exp_name.replace("_nn", "")
        mlp = Experiment(exp_name).load(f"{exp_name}_nn")
        gbt = Experiment(exp_name).load(f"{exp_name}_gbt")

        # Base metrics from methods before removing cycles, after removing cycles,
        # and after adjusting from R2 discrepancy
        metrics['mlp'].append(mlp.rex.metrics_shap)
        metrics['gbt'].append(gbt.rex.metrics_shap)
        metrics['mlp_nc'].append(mlp.rex.metrics_shag)
        metrics['gbt_nc'].append(gbt.rex.metrics_shag)
        metrics['mlp_adj'].append(mlp.rex.metrics_adj)
        metrics['gbt_adj'].append(gbt.rex.metrics_adj)
        metrics['mlp_adjnc'].append(mlp.rex.metrics_adjnc)
        metrics['gbt_adjnc'].append(gbt.rex.metrics_adjnc)

        # Metric from the INTERSECTION graphs
        metrics['intersection'].append(evaluate_graph(
            mlp.ref_graph, utils.graph_intersection(mlp.rex.G_shap, gbt.rex.G_shap)))
        inter_nc = utils.break_cycles_if_present(
            utils.graph_intersection(mlp.rex.G_shag, gbt.rex.G_shag),
            mlp.rex.learnings)
        metrics['intersection_nc'].append(evaluate_graph(
            mlp.ref_graph, inter_nc))

        # Metric from the UNION graphs
        metrics['union'].append(evaluate_graph(
            mlp.ref_graph, utils.graph_union(mlp.rex.G_shap, gbt.rex.G_shap)))
        union_nc = utils.break_cycles_if_present(
            utils.graph_union(mlp.rex.G_shag, gbt.rex.G_shag),
            mlp.rex.learnings)
        metrics['union_nc'].append(evaluate_graph(
            mlp.ref_graph, union_nc))

        # Metrics from the DAGs generated after discrepancy adjustment
        if hasattr(mlp.rex, 'G_adj') and hasattr(gbt.rex, 'G_adj'):
            metrics['intersection_adj'].append(evaluate_graph(
                mlp.ref_graph, utils.graph_intersection(mlp.rex.G_adj, gbt.rex.G_adj)))
            metrics['union_adj'].append(evaluate_graph(
                mlp.ref_graph, utils.graph_union(mlp.rex.G_adj, gbt.rex.G_adj)))
        if hasattr(mlp.rex, 'G_adjnc') and hasattr(gbt.rex, 'G_adjnc'):
            inter_adjnc = utils.break_cycles_if_present(
                utils.graph_intersection(mlp.rex.G_adjnc, gbt.rex.G_adjnc),
                mlp.rex.learnings)
            metrics['intersection_adjnc'].append(evaluate_graph(
                mlp.ref_graph, inter_adjnc))
            union_adjnc = utils.break_cycles_if_present(
                utils.graph_union(mlp.rex.G_adjnc, gbt.rex.G_adjnc),
                mlp.rex.learnings)
            metrics['union_adjnc'].append(evaluate_graph(
                mlp.ref_graph, union_adjnc))

        # Metrics from the UNION of ALL DAGs based on explainability
        metrics['union_all'].append(evaluate_graph(
            mlp.ref_graph, utils.graph_union(
                mlp.rex.G_pi, utils.graph_union(
                    gbt.rex.G_pi, utils.graph_union(
                        mlp.rex.G_shap, gbt.rex.G_shap)
                ))))
        metrics['union_all_nc'].append(evaluate_graph(
            mlp.ref_graph, utils.graph_union(
                mlp.rex.G_pi, utils.graph_union(
                    gbt.rex.G_pi, utils.graph_union(
                        mlp.rex.G_shag, gbt.rex.G_shag)
                ))))

        # Metrics for the intersection and union graphs after checking independence
        # and final graphs
        metrics['int_indep'].append(evaluate_graph(
            mlp.ref_graph, utils.graph_intersection(mlp.rex.G_indep, gbt.rex.G_indep)))
        metrics['int_final'].append(evaluate_graph(
            mlp.ref_graph, utils.graph_intersection(mlp.rex.G_final, gbt.rex.G_final)))
        metrics['union_indep'].append(evaluate_graph(
            mlp.ref_graph, utils.graph_union(mlp.rex.G_indep, gbt.rex.G_indep)))
        metrics['union_final'].append(evaluate_graph(
            mlp.ref_graph, utils.graph_union(mlp.rex.G_final, gbt.rex.G_final)))

    for key in metrics:
        metrics[key] = pd.DataFrame(metrics[key])

    return metrics


def combined_dag_from_experiment(
    full_path_filename: str,
    combined: str,
    prior: bool = False,
    iterative: bool = False,
    verbose: bool = False
) -> nx.DiGraph:

    dnn = Experiment(
        full_path_filename, model_type='nn', verbose=verbose).load()
    gbt = Experiment(
        full_path_filename, model_type='gbt', verbose=verbose).load()

    dags = defaultdict(nx.DiGraph)
    for dag_name in RAW_DAG_NAMES:
        un_name = f'un_{dag_name}'
        in_name = f'in_{dag_name}'
        dag1 = getattr(dnn.rex, dag_name)
        dag2 = getattr(gbt.rex, dag_name)
        if 'prio' not in dag_name:
            prior = None
        else:
            prior = dnn.rex.prior
        _, _, dags[un_name], dags[in_name] = utils.combine_dags(
            dag1, dag2, dnn.rex.shaps.shap_discrepancies, prior=prior)

    comb = "un_G" if combined == 'union' else "in_G"
    prior = "_prior" if prior else ""
    iter = "_iter" if iterative else ""
    if not prior and not iterative:
        comb = comb + "_shap"

    # Build the DAG name, from those in the 'combined_dag_names' list, based
    # on the values from "comb", "prior", and "iter".
    dag_name = comb + iter + prior

    return dags[dag_name]


def plot_combined_metrics(
        metrics: dict,
        metric_types: list = None,
        title: str = None,
        acyclic=False,
        medians=False,
        pdf_filename=None):
    """
    Plot the metrics for all the experiments matching the input pattern

    Parameters
    ----------
    metrics : dict
        A dictionary with the metrics for all the experiments
    title : str
        The title of the plot
    acyclic : bool
        Whether to plot the metrics for the no_cycles graphs
    medians: bool
        Whether to plot the median lines

    Returns
    -------
    None
    """
    what = ['f1', 'precision', 'recall', 'shd', 'sid']
    axs = plt.figure(layout="constrained").subplot_mosaic(
        """
        AABBCC
        .DDEE.
        """
    )
    # fig, axs = plt.subplots(2, 2, figsize=(6, 5))
    if metric_types is None:
        if acyclic:
            metric_types = global_nc_metric_types
        else:
            metric_types = global_metric_types

    ax_labels = list(axs.keys())
    for i, metric in enumerate(what):
        # row, col = i // 2, i % 2
        ax = axs[ax_labels[i]]
        metric_values = [metrics[key].loc[:, metric] for key in metric_types]

        if medians:
            combined_median = np.median(metrics['intersection'].loc[:, metric])
            added_median = np.median(metrics['union'].loc[:, metric])
            ax.axhline(
                combined_median, color='g', linestyle='--', linewidth=0.5)
            ax.axhline(
                added_median, color='b', linestyle='--', linewidth=0.5)

        ax.boxplot(
            metric_values,
            labels=[metric_labels[key] for key in metric_types])
        # check if any value is above 1
        if np.all(np.array(metric_values) <= 1.0):
            ax.set_ylim([0, 1.05])
        ax.set_title(score_titles[metric])

    if title is not None:
        fig = plt.gcf()
        fig.suptitle(title)

    if pdf_filename is not None:
        plt.savefig(pdf_filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_score_by_subtype(
        metrics: pd.DataFrame,
        score_name: str,
        methods=None,
        pdf_filename=None,
        **kwargs):
    """
    Plots the score by subtype.

    Parameters:
    - metrics (pandas DataFrame): The metrics for all the experiments. This dataframe
    contains the following columns:
        - method (str): The name of the method used.
        - data_type (str): The type of data used.
        - f1 (float): The F1 score.
        - precision (float): The precision score.
        - recall (float): The recall score.
        - aupr (float): The area under the precision-recall curve.
        - Tp (int): The number of true positives.
        - Tn (int): The number of true negatives.
        - Fp (int): The number of false positives.
        - Fn (int): The number of false negatives.
        - shd (int): The structural Hamming distance.
        - sid (int): The structural intervention distance.
        - n_edges (int): The number of edges in the graph.
        - ref_n_edges (int): The number of edges in the reference graph.
        - diff_edges (int): The difference between the number of edges in the graph
        and the reference graph.
        - name (str): The name of the experiment.
    and stores one experiment per row.
    - score_name (str): The name of the score to plot. Valid names are 'f1',
    'precision', 'recall', 'aupr', 'shd', 'sid', 'n_edges', 'ref_n_edges' and
    'diff_edges'.
    - methods (list, optional): The list of methods to plot. If None, all the methods
    will be plotted. The methods included are: 'rex_intersection', 'rex_union',
    'pc', 'fci', 'ges', 'lingam'
    - pdf_filename (str, optional): The filename to save the plot to. If None, the plot
    will be displayed on screen, otherwise it will be saved to the specified filename.

    Optional parameters:
    - figsize (tuple, optional): The size of the figure. Default is (2, 1).
    - dpi (int, optional): The resolution of the figure in dots per inch. Default is 300.
    - method_column (str, optional): The name of the column in the metrics dataframe
        that contains the method name. Default is 'method'.

    Returns:
    None
    """
    figsize_ = kwargs.get('figsize', (9, 5))
    dpi_ = kwargs.get('dpi', 300)
    method_column = kwargs.get('method_column', 'method')

    if methods is None:
        methods = ['pc', 'fci', 'ges', 'lingam', 'cam', 'notears', 'un_G_iter']
    x_labels = [method_labels[m] for m in methods]
    axs = plt.figure(layout="constrained", figsize=figsize_, dpi=dpi_).\
        subplot_mosaic('AABBCC;.EEFF.')

    # Loop through all the subtypes
    ax_labels = list(axs.keys())
    for i, subtype in enumerate(synth_data_types):  # + ['all']):
        ax = axs[ax_labels[i]]
        if subtype == 'all':
            sub_df = metrics
        else:
            sub_df = metrics[metrics['data_type'] == subtype]
        metric_values = [sub_df[sub_df[method_column] == m][score_name]
                         for m in methods]
        ax.boxplot(metric_values)
        ax.set_xticklabels(labels=x_labels, fontsize=6)

        if np.all(np.array(metric_values) <= 1.0) and \
                np.all(np.array(metric_values) >= 0.0):
            ax.set_ylim([0, 1.05])
        if not np.any(np.array(metric_values) < 0.0):
            ax.set_ylim(bottom=0)

        yticks = ax.get_yticks()
        if np.max(metric_values) <= yticks[-2]:
            yticks = ax.get_yticks()[:-1]
        ax.set_yticklabels(labels=[f"{t:.1f}" for t in yticks], fontsize=6)

        ax.grid(axis='y', linestyle='--', linewidth=0.5, which='both')

        #  Remove the top and right axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_bounds(high=yticks[-1])
        ax.spines['left'].set_visible(False)

        # Set the title to the score name, in Latex math mode
        if subtype == 'all':
            ax.set_title(
                r'$\textrm{{all data}}$',
                fontsize=10)  # , y=-0.25)
        else:
            ax.set_title(
                rf'{synth_data_labels[synth_data_types.index(subtype)]} data',
                fontsize=10)  # , y=-0.25)

    plt.suptitle(f"{score_titles[score_name]} score")
    if pdf_filename is not None:
        plt.savefig(pdf_filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_scores_by_method(
        metrics: pd.DataFrame,
        methods=None,
        title: str = None,
        pdf_filename=None,
        **kwargs):
    """
    Plot the metrics for all the experiments matching the input pattern

    Parameters
    ----------
    metrics : pandas DataFrame
        A DataFrame with the metrics for all the experiments
    method_types : list
        The list of methods to plot. If None, all the methods will be plotted
        The methods included are: 'rex_mlp', 'rex_gbt', 'rex_intersection' and
        'rex_union'
    title : str
        The title of the plot
    pdf_filename : str
        The filename to save the plot to. If None, the plot will be displayed
        on screen, otherwise it will be saved to the specified filename.

    Optional parameters:
    - figsize (tuple, optional): The size of the figure. Default is (7, 5).
    - dpi (int, optional): The resolution of the figure in dots per inch.
        Default is 300.
    - ylim (tuple, optional): The y-axis limits of the plot. Default is None.


    Returns
    -------
    None
    """
    figsize_ = kwargs.get('figsize', (9, 5))
    dpi_ = kwargs.get('dpi', 300)
    method_column = kwargs.get('method_column', 'method')
    if methods is None:
        methods = ['rex_mlp', 'rex_gbt', 'rex_intersection', 'rex_union']

    what = ['f1', 'precision', 'recall', 'shd', 'sid']
    axs = plt.figure(layout="constrained", figsize=figsize_, dpi=dpi_).\
        subplot_mosaic('AABBCC;.DDEE.')

    ax_labels = list(axs.keys())
    for i, metric in enumerate(what):
        ax = axs[ax_labels[i]]
        metric_values = [metrics[metrics[method_column] == m][metric]
                         for m in methods]

        ax.boxplot(metric_values)
        ax.set_xticklabels(labels=[method_labels[key]
                                   for key in methods], fontsize=6)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, which='both')

        # check if any value is above 1
        if np.all(np.array(metric_values) <= 1.0) and \
                np.all(np.array(metric_values) >= 0.0):
            ax.set_ylim([0, 1.05])
        if not np.any(np.array(metric_values) < 0.0):
            ax.set_ylim(bottom=0)

        # Set the Y tick labels and left axis bounds
        yticks = ax.get_yticks()
        if np.max(metric_values) <= yticks[-1]:
            yticks = ax.get_yticks()[:-1]
        ax.set_yticklabels(labels=[f"{t:.1f}" for t in yticks], fontsize=6)

        #  Remove the top and right axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_bounds(high=yticks[-1])
        ax.spines['left'].set_visible(False)

        ax.set_title(score_titles[metric], fontsize=10)

    if title is not None:
        fig = plt.gcf()
        fig.suptitle(title)

    if pdf_filename is not None:
        plt.savefig(pdf_filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_score_by_method(metrics, metric, methods, **kwargs):
    """
    Plots the score by method.

    Parameters:
    - metrics: DataFrame containing the metrics data.
    - metric: The metric to be plotted.
    - methods: List of methods to be included in the plot.
    - **kwargs: Additional keyword arguments for customization, like
        - figsize: The size of the figure. Default is (4, 3).
        - dpi: The resolution of the figure in dots per inch. Default is 300.
        - title: The title of the plot. Default is None.
        - pdf_filename: The filename to save the plot to. If None, the plot
            will be displayed on screen, otherwise it will be saved to the
        - method_column: The name of the column containing the method names.
            Default is 'method'.

    Returns:
    None
    """
    assert metric in list(score_titles.keys()), \
        ValueError(f"Metric '{metric}' not recognized.")
    figsize_ = kwargs.get('figsize', (4, 3))
    dpi_ = kwargs.get('dpi', 300)
    title_ = kwargs.get('title', None)
    pdf_filename = kwargs.get('pdf_filename', None)
    method_column = kwargs.get('method_column', 'method')

    if methods is None:
        methods = ['rex_mlp', 'rex_gbt', 'rex_intersection', 'rex_union']

    _, ax = plt.subplots(1, 1, figsize=figsize_, dpi=dpi_)

    metric_values = [metrics[metrics[method_column] == m][metric]
                     for m in methods]

    plt.boxplot(metric_values)
    ax.set_xticklabels(labels=[method_labels[key]
                               for key in methods], fontsize=7)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, which='both')

    # check if any value is above 1
    if np.all(np.array(metric_values) <= 1.0) and \
            np.all(np.array(metric_values) >= 0.0):
        ax.set_ylim([0, 1.05])
    if not np.any(np.array(metric_values) < 0.0):
        ax.set_ylim(bottom=0)

    # Set the Y tick labels and left axis bounds
    yticks = ax.get_yticks()
    if np.max(metric_values) <= yticks[-1]:
        yticks = ax.get_yticks()[:-1]
    ax.set_yticklabels(labels=[f"{t:.1f}" for t in yticks], fontsize=6)

    #  Remove the top and right axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds(high=yticks[-1])
    ax.spines['left'].set_visible(False)

    if title_ is None:
        ax.set_title(score_titles[metric], fontsize=10)
    else:
        ax.set_title(title_, fontsize=10)

    if pdf_filename is not None:
        plt.savefig(pdf_filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def format_mean_std(data):
    # """\scalemath{0.6}{\ \pm\ 0.05}"""
    return rf'${data.mean():.2f} \scalemath{{0.6}}{{\ \pm\ {data.std():.1f}}}$'


def latex_table_by_datatype(df, method, metrics=None):
    if metrics is None:
        metrics = ['precision', 'recall', 'f1', 'shd', 'sid']

    table = "\\begin{tabular}{l|" + 'c'*len(metrics) + "}\n\\toprule\n"
    # table += "{} & Precision & Recall & F1 & SHD & SID \\\\ \\midrule\n"
    table += "{} " + \
        ''.join(
            f"& {score_titles[m]}" for m in metrics) + " \\\\ \\midrule\n"
    for i, data_type in enumerate(synth_data_types):
        table += synth_data_labels[i]
        for metric in metrics:
            data = df[(df.method == method) & (
                df.data_type == data_type)][metric]
            table += f" & {format_mean_std(data)}"
        table += " \\\\\n"
    table += "\\bottomrule\n"
    table += "\\end{tabular}"
    print(table)


def latex_table_by_method(df, methods=None, metric_names=None):

    if methods is None:
        methods = ['rex_mlp', 'rex_gbt', 'rex_union',
                   'rex_union_adjnc', 'pc', 'fci', 'ges', 'lingam']

    if metric_names is None:
        metric_names = ['precision', 'recall', 'f1', 'shd', 'sid']

    table = "\\begin{tabular}{l|" + 'c'*len(metric_names) + "}\n\\toprule\n"
    table += "{} " + \
        ''.join(
            f"& {score_titles[m]}" for m in metric_names) + " \\\\ \\midrule\n"
    for method in methods:
        table += method_labels[method]
        for metric in metric_names:
            data = df[(df.method == method)][metric]
            table += f" & {format_mean_std(data)}"
        table += " \\\\\n"
    table += "\\bottomrule\n"
    table += "\\end{tabular}"
    print(table)


def plot_all_dags(what, include_others=True, **kwargs):
    """
    Plot the directed acyclic graphs (DAGs) for various methods.

    Parameters:
    - what (str): The name of the experiment.
    - include_others (bool, optional): Whether to include the DAGs for the other
        methods. Default is True. Other methods are: 'pc', 'lingam', 'ges' and 'fci'.

    Returns:
    None
    """
    pdf_filename = kwargs.get("pdf_filename", None)
    figsize_ = kwargs.get("figsize", (18, 15))
    dpi_ = kwargs.get("dpi", 300)

    if include_others:
        pc = Experiment(f"{what}").load(f"{what}_pc")
        lingam = Experiment(f"{what}").load(f"{what}_lingam")
        ges = Experiment(f"{what}").load(f"{what}_ges")
        fci = Experiment(f"{what}").load(f"{what}_fci")
    nn = Experiment(f"{what}").load(f"{what}_nn")
    gbt = Experiment(f"{what}").load(f"{what}_gbt")
    union = utils.graph_union(nn.rex.G_shag, gbt.rex.G_shag)
    union = utils.break_cycles_if_present(union, nn.rex.learnings)
    inter = utils.graph_intersection(nn.rex.G_shag, gbt.rex.G_shag)
    inter = utils.break_cycles_if_present(inter, nn.rex.learnings)
    union_adj = utils.graph_union(nn.rex.G_adjnc, gbt.rex.G_adjnc)
    union_adj = utils.break_cycles_if_present(union_adj, nn.rex.learnings)
    inter_adj = utils.graph_intersection(nn.rex.G_adjnc, gbt.rex.G_adjnc)
    inter_adj = utils.break_cycles_if_present(inter_adj, nn.rex.learnings)

    if include_others:
        _, ax = plt.subplots(3, 4, figsize=figsize_, dpi=dpi_,
                             gridspec_kw={'hspace': 0.5, 'wspace': 0.2})
    else:
        _, ax = plt.subplots(2, 4, figsize=figsize_, dpi=dpi_,
                             gridspec_kw={'hspace': 0.5, 'wspace': 0.2})

    plot.setup_plot()
    plot.dag(graph=nn.rex.G_shag, reference=nn.ref_graph, show_node_fill=False,
             ax=ax[0, 0], title=method_labels["nn"])
    plot.dag(gbt.rex.G_shag, nn.ref_graph, show_node_fill=False,
             ax=ax[0, 1], title=method_labels["gbt"])
    plot.dag(union, nn.ref_graph, ax=ax[0, 2], title=method_labels["union"])
    plot.dag(inter, nn.ref_graph, ax=ax[0, 3],
             title=method_labels["intersection"])

    plot.dag(graph=nn.rex.G_adj, reference=nn.ref_graph, show_node_fill=False,
             ax=ax[1, 0], title=method_labels["nn_adj"])
    plot.dag(gbt.rex.G_adj, nn.ref_graph, show_node_fill=False,
             ax=ax[1, 1], title=method_labels["gbt_adj"])
    plot.dag(union_adj, nn.ref_graph,
             ax=ax[1, 2], title=method_labels["union_adj"])
    plot.dag(inter_adj, nn.ref_graph,
             ax=ax[1, 3], title=method_labels["intersection_adj"])

    if include_others:
        plot.dag(graph=pc.pc.dag, reference=nn.ref_graph,
                 ax=ax[2, 0], title=method_labels["pc"])
        plot.dag(graph=lingam.lingam.dag, reference=nn.ref_graph,
                 ax=ax[2, 1], title=method_labels["lingam"])
        plot.dag(graph=ges.ges.dag, reference=nn.ref_graph,
                 ax=ax[2, 2], title=method_labels["ges"])
        plot.dag(graph=fci.fci.dag, reference=nn.ref_graph,
                 ax=ax[2, 3], title=method_labels["fci"])

    plt.suptitle(what)

    if pdf_filename is not None:
        plt.savefig(pdf_filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=150)
    warnings.filterwarnings('ignore')
    extra_args = {
        'rex': {
            'prog_bar': True,
            'verbose': False,
            'hpo_n_trials': 1,
            'bootstrap_trials': 10,
            'bootstrap_parallel_jobs': -1,
            'parallel_jobs': -1
        },
        'pc': {},
        'ges': {},
        'lingam': {},
        'fci': {},
        'cam': {
            'pruning': True,
            'pruneMethodPars': {"cutOffPVal": 0.05, "numBasisFcts": 10}
        },
        'notears': {}
    }

    input_path = os.path.expanduser("~/phd/data/RC4/")
    output_path = os.path.expanduser("~/phd/output/")

    method_name = "rex"
    # dataset_name =  "toy_dataset"
    dataset_name =  "generated_10vars_linear_0"

    exp = Experiment(
        experiment_name=dataset_name,
        csv_filename=os.path.join(input_path, f"{dataset_name}.csv"),
        dot_filename=os.path.join(input_path, f"{dataset_name}.dot"),
        model_type="gbt",
        input_path=input_path,
        output_path=output_path)

    exp = exp.fit_predict(method_name, **extra_args[method_name])
    method = getattr(exp, method_name)
    print(method.dag.edges())
    print(method.metrics)
    t, u = utils.format_time(exp.fit_predict_time)
    print(f"Elapsed time: {t:.1f}{u}")
