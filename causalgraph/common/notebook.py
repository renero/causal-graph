"""
A module to run experiments with the causalgraph package, and simplify the
process of loading and saving experiments in notebooks.

Example:
    >>> from causalgraph.common.experiment import init_experiment, run_experiment
    >>> experiment = init_experiment("RC3")
    >>> rex = run_experiment(experiment)
    
(C) 2023 J. Renero
"""

from collections import defaultdict
import glob
from sklearn.preprocessing import StandardScaler
from causalgraph.metrics.compare_graphs import evaluate_graph
from causalgraph.estimators.rex import Rex
from causalgraph.common.utils import (graph_from_dot_file, load_experiment,
                                      save_experiment)
import shap
import pandas as pd
import numpy as np
import os
from os import path
import warnings
from causalgraph.common import utils
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches


metric_types = ['mlp', 'gbt', 'intersection', 'union', 'union_all',
                'int_indep', 'int_final', 'union_indep', 'union_final']
nc_metric_types = ['mlp_nc', 'gbt_nc', 'intersection_nc', 'union_nc', 'union_all_nc',
                   'int_indep', 'int_final', 'union_indep', 'union_final']


class BaseExperiment:
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
        if self.save_experiment and not experiment_exists:
            print("    +-> Experiment will be saved.") if self.verbose else None
        elif self.save_experiment and experiment_exists:
            print("    +-> Experiment exists and will be overwritten.") \
                if self.verbose else None
        else:
            print("    +-> Experiment will NOT be saved.") if self.verbose else None
            self.save_experiment = False

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


class Experiment(BaseExperiment):

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

        super().__init__(
            input_path, output_path, train_size=train_size,
            random_state=random_state, verbose=verbose)

        # Prepare the input
        self.prepare_experiment_input(
            experiment_name, csv_filename, dot_filename)

    def load(self, exp_name=None) -> Rex:
        if exp_name is not None:
            self.rex = load_experiment(exp_name, self.output_path)
            if self.verbose:
                print(f"Loaded '{exp_name}' from '{self.output_path}'")
        else:
            self.rex = load_experiment(self.experiment_name, self.output_path)
            if self.verbose:
                print(
                    f"Loaded '{self.experiment_name}' from '{self.output_path}'")

        return self

    def fit(self, **kwargs) -> Rex:
        self.rex = Rex(name=self.experiment_name, **kwargs)
        self.rex.fit_predict(self.train_data, self.test_data, self.ref_graph)

        return self

    def save(self, exp_name=None, overwrite: bool = False):
        save_as = exp_name if exp_name is not None else self.experiment_name
        where_to = save_experiment(
            save_as, self.output_path, self.rex, overwrite)
        print(f"Saved '{self.experiment_name}' to '{where_to}'")


class Experiments(BaseExperiment):

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
        Loads all the experiments matching the input pattern
        """
        if pattern is not None:
            pickle_files = self.list_files(pattern, where='output')
        else:
            pickle_files = self.input_files

        self.experiment = {}
        for input_filename, pickle_filename in zip(self.input_files, pickle_files):
            self.experiment_name = path.basename(pickle_filename)
            self.input_name = path.basename(input_filename)
            self.decide_what_to_do()

            if self.load_experiment:
                self.experiment[self.experiment_name] = load_experiment(
                    self.experiment_name, self.output_path)
                self.experiment[self.experiment_name].ref_graph = graph_from_dot_file(
                    f"{path.join(self.input_path, self.input_name)}.dot")
                if self.verbose:
                    print(f"        +-> Loaded {self.experiment_name} "
                          f"({type(self.experiment[self.experiment_name])})")
            else:
                print(
                    f"No trained experiment for {pickle_filename}...") if self.verbose else None

        self.is_fitted = True
        self.names = list(self.experiment.keys())
        self.experiments = list(self.experiment.values())
        return self

    def fit(self, save_as_pattern=None, **kwargs) -> list:
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

            print(f"Training Rex on {filename}...")
            self.experiment[self.experiment_name] = Rex(
                name=self.experiment_name, **kwargs)
            self.experiment[self.experiment_name].fit_predict(
                self.train_data, self.test_data, self.ref_graph)

            saved_to = save_experiment(
                save_as, self.output_path, self.experiment[self.experiment_name])
            print(f"\rSaved to: {saved_to}")

        self.is_fitted = True
        self.names = list(self.experiment.keys())
        self.experiments = list(self.experiment.values())
        return self


def get_combined_metrics(subtype: str):
    """
    Obtain the metrics for all the experiments matching the input pattern

    Parameters
    ----------
    subtype : str
        The subtype of the experiment, e.g. "linear" or "nonlinear"

    Returns
    -------
    dict
        A dictionary with the metrics for all the experiments
    """
    holder = Experiment(f"rex_generated_{subtype}_1")
    files = [path.basename(f) for f in holder.list_files(
        f"rex_generated_{subtype}_*")]

    metrics = defaultdict(list)
    for exp_name in files:
        mlp = Experiment(exp_name).load(f"{exp_name}_nn")
        gbt = Experiment(exp_name).load(f"{exp_name}_gbt")

        metrics['mlp'].append(mlp.rex.metrics_shap)
        metrics['gbt'].append(gbt.rex.metrics_shap)
        metrics['mlp_nc'].append(mlp.rex.metrics_shag)
        metrics['gbt_nc'].append(gbt.rex.metrics_shag)
        metrics['intersection'].append(evaluate_graph(
            mlp.ref_graph, utils.graph_intersection(mlp.rex.G_shap, gbt.rex.G_shap)))
        metrics['intersection_nc'].append(evaluate_graph(
            mlp.ref_graph, utils.graph_intersection(mlp.rex.G_shag, gbt.rex.G_shag)))
        metrics['union'].append(evaluate_graph(
            mlp.ref_graph, utils.graph_union(mlp.rex.G_shap, gbt.rex.G_shap)))
        metrics['union_nc'].append(evaluate_graph(
            mlp.ref_graph, utils.graph_union(mlp.rex.G_shag, gbt.rex.G_shag)))
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


def plot_combined_metrics(metrics: dict, title: str, acyclic=False):
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

    Returns
    -------
    None
    """
    what = ['f1', 'precision', 'recall', 'shd']
    fig, axs = plt.subplots(2, 2, figsize=(6, 5))

    for i, metric in enumerate(what):
        row, col = i // 2, i % 2

        combined_median = np.median(metrics['intersection'].loc[:, metric])
        added_median = np.median(metrics['union'].loc[:, metric])

        if acyclic:
            metric_values = [metrics[key].loc[:, metric]
                             for key in nc_metric_types]
        else:
            metric_values = [metrics[key].loc[:, metric]
                             for key in metric_types]

        axs[row, col].axhline(combined_median, color='g',
                              linestyle='--', linewidth=0.5)
        axs[row, col].axhline(added_median, color='b',
                              linestyle='--', linewidth=0.5)

        axs[row, col].boxplot(
            metric_values,
            labels=['nn', 'gbt', '∩', '[∪]', 'all', "∩i", "∩f", "∪i", "∪f"])
        # check if any value is above 1
        if np.all(np.array(metric_values) <= 1.0):
            axs[row, col].set_ylim([0, 1])
        axs[row, col].set_title(metric)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    experiments = Experiments("rex_generated_linear_*.csv", verbose=False)
    experiments.load("rex_generated_linear_*_gbt.pickle")
    main_metrics = experiments.metrics()
    print(main_metrics)
