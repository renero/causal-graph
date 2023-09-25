"""
A module to run experiments with the causalgraph package, and simplify the
process of loading and saving experiments in notebooks.

Example:
    >>> from causalgraph.common.experiment import init_experiment, run_experiment
    >>> experiment = init_experiment("RC3")
    >>> rex = run_experiment(experiment)
    
(C) 2023 J. Renero
"""

import warnings
warnings.filterwarnings('ignore')

from os import path
import os
import networkx as nx
import numpy as np
import pandas as pd
import shap
from causalgraph.common.utils import (graph_from_dot_file, load_experiment,
                                      save_experiment)
from causalgraph.estimators.rex import Rex
# from causalgraph.models import GBTRegressor, NNRegressor
from sklearn.preprocessing import StandardScaler
import glob


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

    def prepare_experiment_input(self, experiment_filename):
        """
        - Loads the data and 
        - splits it into train and test, 
        - scales it
        - loads the reference graph from the dot file, which has to be named
          as the experiment file, with the .dot extension
        """
        self.experiment_name = path.basename(experiment_filename)
        self.data = pd.read_csv(
            f"{path.join(self.input_path, self.experiment_name)}.csv")
        self.data = self.data.apply(pd.to_numeric, downcast='float')
        scaler = StandardScaler()
        self.data = pd.DataFrame(
            scaler.fit_transform(self.data), columns=self.data.columns)
        self.train_data = self.data.sample(frac=self.train_size, random_state=42)
        self.test_data = self.data.drop(self.train_data.index)

        self.ref_graph = graph_from_dot_file(
            f"{path.join(self.input_path, self.experiment_name)}.dot")

        if self.verbose:
            print(
                f"Data for {self.experiment_name}\n"
                f"+-> Train....: {self.data.shape[0]} rows, "
                f"{self.data.shape[1]} cols\n"
                f"+-> Test.....: {self.test_data.shape[0]} rows, "
                f"{self.data.shape[1]} cols\n"
                f"+-> Ref.graph: {self.experiment_name}.dot")

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
                print(f"    +-> Experiment '{self.experiment_name}' will be LOADED")
            else:
                print(f"    +-> Experiment '{self.experiment_name}' will be TRAINED")

        self.save_experiment = True
        if self.save_experiment and not experiment_exists:
            print(f"    +-> Experiment will be saved.") if self.verbose else None
        elif self.save_experiment and experiment_exists:
            print(f"    +-> Experiment exists and will be overwritten.") \
                if self.verbose else None
        else:
            print(f"    +-> Experiment will NOT be saved.") if self.verbose else None
            self.save_experiment = False

    def list_files(self) -> list:
        """
        List all the files in the input path matching the input pattern
        """
        input_files = glob.glob(os.path.join(
            self.input_path, self.input_pattern))
        input_files = sorted([os.path.splitext(f)[0] for f in input_files])

        assert len(self.input_files) > 0, \
            f"No files found in {self.input_path} matching <{self.input_pattern}>"

        return input_files


class Experiment(BaseExperiment):

    def __init__(
        self,
        experiment_name,
        input_path="/Users/renero/phd/data/RC3/",
        output_path="/Users/renero/phd/output/RC3/",
        train_size: float = 0.9,
        random_state: int = 42,
        verbose=False
    ):

        super().__init__(
            input_path, output_path, train_size=train_size,
            random_state=random_state, verbose=verbose)

        self.experiment_name = experiment_name

        # Prepare the input
        self.prepare_experiment_input(experiment_name)

    def load(self) -> Rex:
        rex = load_experiment(self.experiment_name, self.output_path)
        print(f"Loaded '{self.experiment_name}' from '{self.output_path}'")
        
        return rex
            
    def train(self, **kwargs) -> Rex:
        rex = Rex(**kwargs)
        rex.fit_predict(self.train_data, self.test_data, self.ref_graph)

        return rex

    def save(self, rex: Rex):
        if self.save_experiment:
            where_to = save_experiment(
                self.experiment_name, self.output_path, rex)
            print(f"Saved '{self.experiment_name}' to '{where_to}'")
        else:
            print(f"Experiment '{self.experiment_name}' was not saved")


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
            verbose=False):

        super().__init__(
            input_path, output_path, train_anyway, save_anyway, train_size,
            random_state, verbose)
        self.input_pattern = input_pattern
        self.input_files = self.list_files()
        self.experiment_name = None
        
        if self.verbose:
            print(
                f"Found {len(self.input_files)} files matching <{self.input_pattern}>")

    def load(self) -> dict:
        """
        Loads all the experiments matching the input pattern
        """
        exp = {}
        for filename in self.input_files:
            self.experiment_name = path.basename(filename)
            self.decide_what_to_do()

            if self.load_experiment:
                exp[self.experiment_name] = load_experiment(
                    self.experiment_name, self.output_path)
                exp[self.experiment_name].ref_graph = graph_from_dot_file(
                    f"{path.join(self.input_path, self.experiment_name)}.dot")
                if self.verbose:
                    print(f"        +-> Loaded {self.experiment_name} "
                        f"({type(exp[self.experiment_name])})")
            else:
                print(f"No trained experiment for {filename}...") if self.verbose else None
        
        return exp

    def train(self) -> list:
        exps = {}
        for filename in self.input_files:
            self.prepare_experiment_input(filename)
            self.decide_what_to_do()

            if self.load_experiment:
                exps[self.experiment_name] = load_experiment(
                    self.experiment_name, self.output_path)
                print(f"        +-> Loaded {self.experiment_name} ({type(rex)})")
            else:
                print(f"Training Rex on {filename}...")
                # rex = Rex(
                #     model_type=NNRegressor, explainer=shap.GradientExplainer,
                #     tune_model=True, hpo_n_trials=50, hpo_study_name=f"{experiment}",
                #     silent=True)
                # rex.fit_predict(self.train, self.test, self.ref_graph)

            if self.save_experiment:
                saved_to = "dummy"
                # saved_to = save_experiment(
                #     f'{self.experiment_name}', self.output_path, rex)
                # output_files.append(saved_to)
                print(f"\rSaved to: {saved_to}")


if __name__ == "__main__":
    experiments = Experiments("rex_generated_linear_*.csv", verbose=False)
    experiments.load()
    