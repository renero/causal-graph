"""
This module contains the GraphDiscovery class which is responsible for
creating, fitting, and evaluating causal discovery experiments.
"""

import pickle
import numpy as np

from causalexplain.common import utils
from causalexplain.common.notebook import Experiment
from causalexplain.metrics.compare_graphs import evaluate_graph


class GraphDiscovery:
    def __init__(
        self,
        experiment_name: str,
        model_type: str,
        csv_filename: str,
        true_dag_filename: str,
        verbose: bool = False,
        seed: int = 42
    ) -> None:
        self.dataset_name = experiment_name
        self.estimator = model_type
        self.csv_filename = csv_filename
        self.dot_filename = true_dag_filename
        self.verbose = verbose
        self.seed = seed

    def create_experiments(
        self,
        regressors: list,
        dataset_path: str,
        output_path: str,
    ) -> dict:
        """
        Create an Experiment object for each regressor.

        Args:
            dataset_name (str): Name of the dataset
            true_dag (str): Path to the true DAG DOT file
            regressors (list): List of regressor types to create experiments for
            dataset_path (str): Path to the input dataset
            output_path (str): Path for output files

        Returns:
            dict: A dictionary of Experiment objects
        """
        self.trainer = {}
        for model_type in regressors:
            trainer_name = f"{self.dataset_name}_{model_type}"
            self.trainer[trainer_name] = Experiment(
                experiment_name=self.dataset_name,
                csv_filename=self.csv_filename,
                dot_filename=self.dot_filename,
                model_type=model_type,
                input_path=dataset_path,
                output_path=output_path,
                verbose=False)

        return self.trainer

    def fit_experiments(
        self,
        hpo_iterations: int = None,
        bootstrap_iterations: int = None
    ) -> None:
        """
        Fit the Experiment objects.

        Args:
            trainer (dict): A dictionary of Experiment objects
            estimator (str): The estimator to use ('rex' or other)
            verbose (bool, optional): Whether to print verbose output. 
                Defaults to False.
            hpo_iterations (int, optional): Number of HPO trials for REX. 
                Defaults to None.
            bootstrap_iterations (int, optional): Number of bootstrap trials 
                for REX. Defaults to None.
        """
        if self.estimator == 'rex':
            xargs = {
                'verbose': self.verbose,
                'hpo_n_trials': hpo_iterations,
                'bootstrap_trials': bootstrap_iterations
            }
        else:
            xargs = {
                'verbose': self.verbose
            }

        for trainer_name, experiment in self.trainer.items():
            if not trainer_name.endswith("_rex"):
                experiment.fit_predict(estimator=self.estimator, **xargs)

    def combine_and_evaluate_dags(
        self,
        dataset_path: str,
        output_path: str,
        ref_graph: np.ndarray = None,
        data_columns: list = None,
    ) -> Experiment:
        """
        Retrieve the DAG from the Experiment objects.

        Args:
            trainer (dict): A dictionary of Experiment objects
            dataset_name (str): Name of the dataset
            estimator (str): The estimator type ('rex' or other)
            dataset_path (str): Path to the input dataset
            output_path (str): Path for output files
            ref_graph (np.ndarray, optional): Reference graph for evaluation. 
                Defaults to None.
            data_columns (list, optional): List of column names from the data. 
                Defaults to None.

        Returns:
            Experiment: The experiment object with the final DAG
        """
        if self.estimator != 'rex':
            trainer_key = f"{self.dataset_name}_{self.estimator}"
            estimator_obj = getattr(self.trainer[trainer_key], self.estimator)
            self.trainer[trainer_key].dag = estimator_obj.dag
            if ref_graph is not None and data_columns is not None:
                self.trainer[trainer_key].metrics = evaluate_graph(
                    ref_graph, estimator_obj.dag, data_columns)
            else:
                self.trainer[trainer_key].metrics = None

            return self.trainer[trainer_key]

        # For ReX, we need to combine the DAGs. Hardcode for now to combine
        # the first and second DAGs
        estimator1 = getattr(self.trainer[list(self.trainer.keys())[0]], 'rex')
        estimator2 = getattr(self.trainer[list(self.trainer.keys())[1]], 'rex')
        _, _, dag, _ = utils.combine_dags(estimator1.dag, estimator2.dag,
                                        estimator1.shaps.shap_discrepancies)
        
        # Create a new Experiment object for the combined DAG
        new_trainer = f"{self.dataset_name}_rex"
        if new_trainer in self.trainer:
            del self.trainer[new_trainer]
        self.trainer[new_trainer] = Experiment(
            experiment_name=self.dataset_name,
            model_type='rex',
            input_path=dataset_path,
            output_path=output_path,
            verbose=False)

        # Set the DAG and evaluate
        self.trainer[new_trainer].ref_graph = ref_graph
        self.trainer[new_trainer].dag = dag
        if ref_graph is not None and data_columns is not None:
            self.trainer[new_trainer].metrics = evaluate_graph(
                ref_graph, dag, data_columns)
        else:
            self.trainer[new_trainer].metrics = None

        return self.trainer[new_trainer]

    def save_model(
        self,
        output_path: str,
    ) -> None:
        """
        Save the model as an Experiment object.

        Args:
            trainer (dict): A dictionary of Experiment objects
            output_path (str): Directory path where to save the model
            dataset_name (str): Name of the dataset
            estimator (str): The estimator type ('rex' or other)
        """
        if self.estimator == 'rex':
            # Save trainer with the name of tha last experiment_name in dictionary
            trainer_name = list(self.trainer.keys())[-1]
        else:
            trainer_name = f"{self.dataset_name}_{self.estimator}"
        
        saved_as = utils.save_experiment(
            trainer_name, output_path, self.trainer, overwrite=False)
        print(f"Saved model as: {saved_as}", flush=True)

    def load_models(self, model_path: str) -> Experiment:
        """
        Load the model from a pickle file.

        Args:
            model_path (str): Path to the pickle file containing the model

        Returns:
            Experiment: The loaded Experiment object
        """
        with open(model_path, 'rb') as f:
            self.trainer = pickle.load(f)
            print(f"Loaded model from: {model_path}", flush=True)

        return self.trainer

    def printout_results(self, graph, metrics):
        """
        This method prints the DAG to stdout in hierarchical order.

        Parameters:
        -----------
        dag : nx.DiGraph
            The DAG to be printed.
        """
        if len(graph.edges()) == 0:
            print("Empty graph")
            return

        print("Resulting Graph:\n---------------")

        def dfs(node, visited, indent=""):
            if node in visited:
                return  # Avoid revisiting nodes
            visited.add(node)
            
            # Print edges for this node
            for neighbor in graph.successors(node):
                print(f"{indent}{node} -> {neighbor}")
                dfs(neighbor, visited, indent + "  ")
            
        visited = set()
        
        # Start traversal from all nodes without predecessors (roots)
        for node in graph.nodes:
            if graph.in_degree(node) == 0:
                dfs(node, visited)

        # Handle disconnected components (not reachable from any "root")
        for node in graph.nodes:
            if node not in visited:
                dfs(node, visited)

        if metrics is not None:
            print("\nGraph Metrics:\n-------------")
            print(metrics)
