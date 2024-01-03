"""
A module to run experiments with the causalgraph package, and simplify the
process of loading and saving experiments in notebooks.

Example:
    >>> from causalgraph.common.experiment import init_experiment, run_experiment
    >>> experiment = init_experiment("RC3")
    >>> rex = run_experiment(experiment)
    
(C) 2023 J. Renero
"""

import glob
import os
import warnings
from collections import defaultdict
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from rich.progress import Progress

from causalgraph.common import utils
from causalgraph.common.utils import (graph_from_dot_file, load_experiment,
                                      save_experiment)
from causalgraph.estimators.rex import Rex
from causalgraph.estimators.fci.fci import FCI
from causalgraph.estimators.pc.pc import PC
from causalgraph.estimators.ges.ges import GES
from causalgraph.estimators.lingam.lingam import DirectLiNGAM as LiNGAM
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
    'nn': r'$\textrm{Rex}_{\textrm{\tiny MLP}}$',
    'rex_mlp': r'$\textrm{Rex}_{\textrm{\tiny MLP}}$',
    'nn_adj': r'$\textrm{Rex}_{\textrm{\tiny MLP}}^{\textrm{\tiny adj}}$',
    'rex_mlp_adj': r'$\textrm{Rex}_{\textrm{\tiny MLP}}^{\textrm{\tiny adj}}$',
    'gbt': r'$\textrm{Rex}_{\textrm{\tiny GBT}}$',
    'rex_gbt': r'$\textrm{Rex}_{\textrm{\tiny GBT}}$',
    'gbt_adj': r'$\textrm{Rex}_{\textrm{\tiny GBT}}^{\textrm{\tiny adj}}$',
    'rex_gbt_adj': r'$\textrm{Rex}_{\textrm{\tiny GBT}}^{\textrm{\tiny adj}}$',
    'union': r'$\textrm{Rex}_{\cup}$',
    'rex_union': r'$\textrm{Rex}_{\cup}$',
    'union_adj': r'$\textrm{Rex}_{\cup}^{\textrm{adj}}$',
    'rex_union_adj': r'$\textrm{Rex}_{\cup}^{\textrm{\tiny adj}}$',
    'rex_union_adjnc': r'$\overline{\textrm{Rex}}_{\cup}^{\textrm{\tiny adj}}$',
    'intersection': r'$\textrm{Rex}_{\cap}$',
    'rex_intersection': r'$\textrm{Rex}_{\cap}$',
    'intersection_adj': r'$\textrm{Rex}_{\cap}^{\textrm{\tiny adj}}$',
    'rex_intersection_adj': r'$\textrm{Rex}_{\cap}^{\textrm{\tiny adj}}$',
    'rex_intersection_adjnc': r'$\overline{\textrm{Rex}}_{\cap}^{\textrm{\tiny adj}}$',
    'pc': r'$\textrm{PC}$',
    'fci': r'$\textrm{FCI}$',
    'ges': r'$\textrm{GES}$',
    'lingam': r'$\textrm{LiNGAM}$'
}
estimators = {
    'rex': Rex,
    'fci': FCI,
    'pc': PC,
    'lingam': LiNGAM,
    'ges': GES
}
method_names = ['pc', 'fci', 'ges', 'lingam']
synth_data_types = ['linear', 'polynomial', 'gp_add', 'gp_mix', 'sigmoid_add']
metric_columns = ['method', 'data_type', 'f1', 'precision',
                  'recall', 'aupr', 'Tp', 'Tn', 'Fp', 'Fn', 'shd', 'sid',
                  'n_edges', 'ref_n_edges', 'diff_edges', 'name']


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
            train_size: float = 0.9,
            random_state: int = 42,
            verbose: bool = False):

        self.input_path = input_path
        self.output_path = output_path
        self.train_anyway = train_anyway
        self.save_anyway = save_anyway
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
        scaler = StandardScaler()
        self.data = pd.DataFrame(
            scaler.fit_transform(self.data), columns=self.data.columns)
        self.train_data = self.data.sample(
            frac=self.train_size, random_state=42)
        self.test_data = self.data.drop(self.train_data.index)

        self.ref_graph = graph_from_dot_file(dot_filename)

        if self.verbose:
            print(
                f"Data for {self.experiment_name}\n"
                f"+-> Train....: {self.data.shape[0]} rows, "
                f"{self.data.shape[1]} cols\n"
                f"+-> Test.....: {self.test_data.shape[0]} rows, "
                f"{self.data.shape[1]} cols\n"
                f"+-> CSV.data.: {csv_filename}\n"
                f"+-> Ref.graph: {dot_filename}")

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
                    f"    +-> Experiment '{self.experiment_name}' will be LOADED")
            else:
                print(
                    f"    +-> Experiment '{self.experiment_name}' will be TRAINED")

        self.save_experiment = True
        # if self.save_experiment and not experiment_exists:
        #     print("    +-> Experiment will be saved.") if self.verbose else None
        # elif self.save_experiment and experiment_exists:
        #     print("        +-> Experiment exists and will be overwritten.") \
        #         if self.verbose else None
        # else:
        #     print("    +-> Experiment will NOT be saved.") if self.verbose else None
        #     self.save_experiment = False

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
        input_path="/Users/renero/phd/data/RC3/",
        output_path="/Users/renero/phd/output/RC3/",
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
            input_path (str, optional): The path to the input data. 
                Defaults to "/Users/renero/phd/data/RC3/".
            output_path (str, optional): The path to save the output. 
                Defaults to "/Users/renero/phd/output/RC3/".
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

        # Prepare the input
        self.prepare_experiment_input(
            experiment_name, csv_filename, dot_filename)

    def fit(self, estimator='rex', **kwargs):
        """
        Fits the experiment data.

        Args:
            **kwargs: Additional keyword arguments to pass to the Rex constructor.

        Returns:
            Rex: The fitted experiment data.
        """

        # self.rex = Rex(name=self.experiment_name, **kwargs)
        # self.rex.fit_predict(self.train_data, self.test_data, self.ref_graph)
        self.estimator_name = estimator
        estimator_object = self.create_estimator(
            estimator, name=self.experiment_name, **kwargs)
        estimator_object.fit_predict(
            self.train_data, self.test_data, self.ref_graph)
        setattr(self, estimator, estimator_object)

        return self

    def load(self, exp_name=None) -> Rex:
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

        exp_object = load_experiment(exp_name, self.output_path)

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
        else:
            raise ValueError(
                f"Estimator '{exp_name}' not recognized.")

        setattr(self, self.estimator_name, exp_object)
        setattr(self, 'estimator', exp_object)

        if self.verbose:
            print(f"Loaded '{exp_name}' from '{self.output_path}'")

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
        # estimator_object = getattr(self, self.estimator)
        save_as = exp_name if exp_name is not None else self.experiment_name
        where_to = save_experiment(
            save_as, self.output_path, getattr(self, self.estimator_name), overwrite)
        if self.verbose:
            print(f"Saved '{self.experiment_name}' to '{where_to}'")


class Experiments(BaseExperiment):
    """
    Class representing a collection of experiments.

    Parameters:
    - input_pattern: str
        The pattern used to match input files for the experiments.
    - input_path: str, optional
        The path to the input files. Default is "/Users/renero/phd/data/RC3/".
    - output_path: str, optional
        The path to save the output files. Default is "/Users/renero/phd/output/RC3/".
    - train_anyway: bool, optional
        Whether to train the experiment even if it already exists. Default is False.
    - save_anyway: bool, optional
        Whether to save the experiment even if it already exists. Default is False.
    - train_size: float, optional
        The proportion of the data to use for training. Default is 0.9.
    - random_state: int, optional
        The random state for reproducibility. Default is 42.
    - verbose: bool, optional
        Whether to print verbose output. Default is True.
    """

    estimator_name = None

    def __init__(
            self,
            input_pattern,
            input_path="/Users/renero/phd/data/RC3/",
            output_path="/Users/renero/phd/output/RC3/",
            train_anyway=False,
            save_anyway=False,
            train_size: float = 0.9,
            random_state: int = 42,
            verbose=True):

        super().__init__(
            input_path, output_path, train_anyway, save_anyway, train_size,
            random_state, verbose)
        self.input_files = self.list_files(input_pattern)
        self.experiment_name = None

        if self.verbose:
            print(
                f"Found {len(self.input_files)} files matching <{input_pattern}>")

    def load(self, pattern=None) -> dict:
        """
        Loads all the experiments matching the input pattern.

        Parameters:
        - pattern: str, optional
            The pattern used to match output files for loading experiments.
            If not provided, the input pattern will be used.

        Returns:
        - dict
            A dictionary containing the loaded experiments, where the keys are the experiment names.
        """
        if pattern is not None:
            pickle_files = self.list_files(pattern, where='output')
        else:
            pickle_files = self.input_files

        self.experiment = {}
        with Progress(transient=True) as progress:
            task = progress.add_task(
                "Loading experiments...", total=len(pickle_files))
            for input_filename, pickle_filename in zip(self.input_files, pickle_files):
                self.experiment_name = path.basename(pickle_filename)
                self.input_name = path.basename(input_filename)
                self.decide_what_to_do()

                if self.load_experiment:
                    self.experiment[self.experiment_name] = load_experiment(
                        self.experiment_name, self.output_path)
                    self.experiment[self.experiment_name].ref_graph = \
                        graph_from_dot_file(
                            f"{path.join(self.input_path, self.input_name)}.dot")
                    if self.verbose:
                        print(f"        +-> Loaded {self.experiment_name} "
                              f"({type(self.experiment[self.experiment_name])})")
                    progress.update(task, advance=1, refresh=True,
                                    visible=not self.verbose)
                else:
                    if self.verbose:
                        print(
                            f"No trained experiment for {pickle_filename}...")
            progress.stop()

        self.is_fitted = True
        self.names = list(self.experiment.keys())
        self.experiments = list(self.experiment.values())
        return self

    def fit(self, estimator="rex", save_as_pattern=None, **kwargs) -> list:
        """
        Fits the experiments.

        Parameters:
        - estimator: str, optional
            The estimator to use. Default is "rex". Other options are "pc", "lingam",
            "ges" and "fci".
        - save_as_pattern: str, optional
            The pattern used to save the experiments. "*" will be replaced with the 
            experiment name. If not provided, the experiment name will be used.

        Returns:
        - list
            A list of the fitted experiments.
        """
        self.experiment = {}
        for filename in self.input_files:
            self.prepare_experiment_input(filename)
            if save_as_pattern is None:
                save_as = self.experiment_name
            else:
                save_as = save_as_pattern.replace("*", self.experiment_name)

            if self.experiment_exists(save_as) and not self.train_anyway:
                print(f"Experiment {save_as} already exists, skipping...")
                continue

            self.estimator_name = estimator
            print(f"Training '{estimator}' on {filename}...")
            # self.experiment[self.experiment_name] = Rex(
            #     name=self.experiment_name, **kwargs)
            self.experiment[self.experiment_name] = self.create_estimator(
                estimator, name=self.experiment_name, **kwargs)
            self.experiment[self.experiment_name].fit_predict(
                train=self.train_data, test=self.test_data, ref_graph=self.ref_graph)

            saved_to = save_experiment(
                save_as, self.output_path, self.experiment[self.experiment_name])
            print(f"\rSaved to: {saved_to}")

        self.is_fitted = True
        self.names = list(self.experiment.keys())
        self.experiments = list(self.experiment.values())
        return self


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
        where = "/Users/renero/phd/output/RC3/"
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

    Returns:
    None
    """
    figsize_ = kwargs.get('figsize', (9, 5))
    dpi_ = kwargs.get('dpi', 300)
    if methods is None:
        methods = ['rex_intersection', 'rex_union',
                   'pc', 'fci', 'ges', 'lingam']
    x_labels = [method_labels[m] for m in methods]
    axs = plt.figure(layout="constrained", figsize=figsize_, dpi=dpi_).\
        subplot_mosaic('AABBCC;DDEEFF')

    # Loop through all the subtypes
    ax_labels = list(axs.keys())
    for i, subtype in enumerate(synth_data_types + ['all']):
        ax = axs[ax_labels[i]]
        if subtype == 'all':
            sub_df = metrics
        else:
            sub_df = metrics[metrics['data_type'] == subtype]
        metric_values = [sub_df[sub_df['method'] == m][score_name]
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
            ax.set_title(  # rf'$\textrm{{{score_titles[score_name]}}} '
                # rf'\textrm{{ for }} '
                r'$\textrm{{all data}}$',
                fontsize=10)  # , y=-0.25)
        else:
            ax.set_title(  # rf'$\textrm{{{score_titles[score_name]}}} '
                # rf'\textrm{{ for }} '
                rf'$\texttt{{{subtype}}} \textrm{{ data}}$',
                fontsize=10)  # , y=-0.25)

    plt.suptitle(score_titles[score_name])
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
    if methods is None:
        methods = ['rex_mlp', 'rex_gbt', 'rex_intersection', 'rex_union']

    what = ['f1', 'precision', 'recall', 'shd', 'sid']
    axs = plt.figure(layout="constrained", figsize=figsize_, dpi=dpi_).\
        subplot_mosaic('AABBCC;.DDEE.')

    ax_labels = list(axs.keys())
    for i, metric in enumerate(what):
        ax = axs[ax_labels[i]]
        metric_values = [metrics[metrics['method'] == m][metric]
                         for m in methods]

        ax.boxplot(metric_values)
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

    Returns:
    None
    """
    assert metric in list(score_titles.keys()), \
        ValueError(f"Metric '{metric}' not recognized.")
    figsize_ = kwargs.get('figsize', (4, 3))
    dpi_ = kwargs.get('dpi', 300)
    title_ = kwargs.get('title', None)
    pdf_filename = kwargs.get('pdf_filename', None)

    if methods is None:
        methods = ['rex_mlp', 'rex_gbt', 'rex_intersection', 'rex_union']

    _, ax = plt.subplots(1, 1, figsize=figsize_, dpi=dpi_)

    metric_values = [metrics[metrics['method'] == m][metric]
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


if __name__ == "__main__":
    # experiments = Experiments("rex_generated_linear_*.csv", verbose=False)
    # experiments.load("rex_generated_linear_*_gbt.pickle")
    # main_metrics = experiments.metrics()
    # print(main_metrics)

    e = Experiment('rex_generated_linear_1')\
        .load(exp_name="rex_generated_linear_1_nn")
