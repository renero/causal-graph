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

from sklearn.preprocessing import StandardScaler
from causalgraph.models import GBTRegressor, NNRegressor
from causalgraph.estimators.rex import Rex
from causalgraph.common.utils import (graph_from_dot_file, load_experiment,
                                      save_experiment)
import shap
import pandas as pd
import numpy as np
import networkx as nx
import os



class Experiment:
    
    name: str
    input_path: str
    output_path: str
    train: pd.DataFrame
    test: pd.DataFrame
    ref_graph: nx.DiGraph
    rex: Rex
    load: bool
    save_experiment: bool
        
    def __init__(
            self,
            experiment_name,
            input_path="/Users/renero/phd/data/RC3/",
            output_path="/Users/renero/phd/output/RC3/",
            train_anyway = False,
            save_anyway = False):

        # Display Options
        np.set_printoptions(precision=4, linewidth=100)
        pd.set_option('display.precision', 4)
        pd.set_option('display.float_format', '{:.4f}'.format)

        self.name = experiment_name
        self.input_path = input_path
        self.output_path = output_path

        # Decide whether to load or train the model
        experiment_exists = os.path.exists(
            os.path.join(output_path, f"{self.name}.pickle"))
        if experiment_exists:
            self.load = True and not train_anyway
        else:
            self.load = False and not train_anyway
        self.save_experiment = (True if self.load is False else False) or save_anyway

        if self.load:
            print(f"Experiment '{self.name}' will be LOADED")
        else:
            print(f"Experiment '{self.name}' will be TRAINED")

        if self.save_experiment and not experiment_exists:
            print(f"Experiment will be saved.")
        elif self.save_experiment and experiment_exists:
            print(f"Experiment exists and will be overwritten.")
        else:
            print(f"Experiment will NOT be saved.")

        # Read the data
        self.ref_graph = graph_from_dot_file(f"{input_path}{self.name}.dot")
        data = pd.read_csv(f"{input_path}{self.name}.csv")
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        # Split the dataframe into train and test
        self.train = data.sample(frac=0.9, random_state=42)
        self.test = data.drop(self.train.index)
        

    def run(self, **kwargs) -> Rex:
        if self.load:
            rex = load_experiment(self.name, self.output_path)
            print(f"Loaded '{self.name}' from '{self.output_path}'")
        else:
            rex = Rex(model_type=NNRegressor, explainer=shap.GradientExplainer, **kwargs)
            rex.fit_predict(self.train, self.test, self.ref_graph)

        return rex

    def save_experiment(self, rex: Rex):
        if self.save_experiment:
            where_to = save_experiment(self.name, self.output_path, rex)
            print(f"Saved '{self.name}' to '{where_to}'")
        else:
            print(f"Experiment '{self.name}' was not saved")
