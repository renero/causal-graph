#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# causalexplain/__main__.py
#
# (C) 2024 J. Renero
#
# This file is part of causalexplain
#
# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init
# pylint: disable=C0103:invalid-name, C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, W0511:fixme
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches
#

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from causalexplain.common import (
    DEFAULT_BOOTSTRAP_TOLERANCE,
    DEFAULT_BOOTSTRAP_TRIALS,
    DEFAULT_HPO_TRIALS,
    DEFAULT_REGRESSORS,
    DEFAULT_SEED,
    SUPPORTED_METHODS,
    utils,
)
from causalexplain.common.notebook import Experiment
from causalexplain.metrics.compare_graphs import evaluate_graph


class GraphDiscovery:
    def __init__(self) -> None:
        pass

    def create_experiments(
        self,
        dataset_name: str,
        regressors: list,
        dataset_file: str,
        true_dag_file: str,
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
            trainer_name = f"{dataset_name}_{model_type}"
            self.trainer[trainer_name] = Experiment(
                experiment_name=dataset_name,
                csv_filename=dataset_file,
                dot_filename=true_dag_file,
                model_type=model_type,
                input_path=dataset_path,
                output_path=output_path,
                verbose=False)

        return self.trainer

    def fit_experiments(
        self,
        estimator: str,
        verbose: bool = False,
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
        if estimator == 'rex':
            xargs = {
                'verbose': verbose,
                'hpo_n_trials': hpo_iterations,
                'bootstrap_trials': bootstrap_iterations
            }
        else:
            xargs = {
                'verbose': verbose
            }

        for trainer_name, experiment in self.trainer.items():
            experiment.fit_predict(estimator=estimator, **xargs)


    def combine_and_evaluate_dags(
        self,
        dataset_name: str,
        estimator: str,
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
        if estimator != 'rex':
            trainer_key = f"{dataset_name}_{estimator}"
            estimator_obj = getattr(self.trainer[trainer_key], estimator)
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
        new_trainer = f"{dataset_name}_rex"
        self.trainer[new_trainer] = Experiment(
            experiment_name=dataset_name,
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
        dataset_name: str,
        estimator: str,
        save_model: bool = True,
    ) -> None:
        """
        Save the model as an Experiment object.

        Args:
            trainer (dict): A dictionary of Experiment objects
            output_path (str): Directory path where to save the model
            dataset_name (str): Name of the dataset
            estimator (str): The estimator type ('rex' or other)
            save_model (bool, optional): Whether to save the model. 
                Defaults to True.
        """
        if not save_model:
            print("Not saving model.", flush=True)
            return

        if estimator == 'rex':
            # Save trainer with the name of tha last experiment_name in dictionary
            trainer_name = list(self.trainer.keys())[-1]
        else:
            trainer_name = f"{dataset_name}_{estimator}"
        
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


def parse_args():
    class SplitArgs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            """Split a comma-separated list of values."""
            if values is None:
                setattr(namespace, self.dest, [])
            elif isinstance(values, str):
                setattr(namespace, self.dest, values.split(','))
            else:
                setattr(namespace, self.dest, list(values))

    parser = argparse.ArgumentParser(
        description="Causal Graph Learning with ReX and other compared methods.",
    )
    parser.add_argument(
        '-d', '--dataset', type=str, required=True,
        help='Dataset name. Must be CSV file with headers and comma separated columns')
    parser.add_argument(
        '-m', '--method', type=str, required=False,
        choices=['rex', 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears'],
        help="Method to used. If not specified, the method will be ReX.\n" +
        "Other options are: 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears'.")
    parser.add_argument(
        '-t', '--true_dag', type=str, required=False,
        help='True DAG file name. The file must be in .dot format')
    parser.add_argument(
        '-l', '--load_model', type=str, required=False,
        help='Model name (pickle) to load. If not specified, the model will be trained')
    parser.add_argument(
        '-n', '--no-train', action='store_true', required=False,
        help='Do not train the model. If not specified, the model will be trained')
    parser.add_argument(
        '-T', '--threshold', type=float, required=False,
        help='Threshold to apply to the bootstrapped adjacency matrix.')
    parser.add_argument(
        '-u', '--union', type=str, action=SplitArgs, required=False,
        help='List of comma-separated DAGs from regressor to combine. Default is all of them.')
    parser.add_argument(
        '-i', '--iterations', type=int, required=False,
        help='Hyper-parameter tuning max. iterations')
    parser.add_argument(
        '-b', '--bootstrap', type=int, required=False,
        help='Bootstrap iterations')
    parser.add_argument(
        '-r', '--regressor', type=str, required=False, action=SplitArgs,
        help='Regressor list')
    parser.add_argument(
        '-S', '--seed', type=int, required=False, help='Random seed')
    parser.add_argument(
        '-s', '--save_model', type=str, required=False, nargs='?', const='',
        help='Save model as specified name. If not specified the model will be saved' +
        ' with the same name as the dataset, but with pickle extension.')
    parser.add_argument(
        '-v', '--verbose', action='store_true', required=False, help='Verbose')
    parser.add_argument(
        '-q', '--quiet', action='store_true', required=False, help='Quiet')
    parser.add_argument(
        '-o', '--output', type=str, required=False,
        help='Output file where saving the final DAG (dot format). If not specified, ' +
        'the final DAG will be printed to stdout.')

    args = parser.parse_args()
    return args


def check_args_validity(args):
    """
    Check the validity of the arguments.

    Returns:
        dict: A dictionary of run values
    """
    run_values = {}

    # Set model type (estimator)
    if args.method is None:
        args.method = 'rex'
        run_values['estimator'] = 'rex'
    else:
        assert args.method in SUPPORTED_METHODS, \
            "Method must be one of: rex, pc, fci, ges, lingam, cam, notears"
        run_values['estimator'] = str(args.method)

    # Check that the dataset file exist, and load it (data)
    assert args.dataset is not None, "Dataset file must be specified"
    assert os.path.isfile(args.dataset), \
        f"Dataset file '{args.dataset}' does not exist"
    run_values['data'] = pd.read_csv(args.dataset)
    run_values['dataset_filepath'] = args.dataset

    # Extract the path from where the dataset is, dataset basename
    run_values['dataset_path'] = os.path.dirname(args.dataset)
    dataset_name = os.path.basename(args.dataset)
    run_values['dataset_name'] = dataset_name.replace('.csv', '')

    # Load true DAG, if specified (true_dag)
    run_values['true_dag'] = None
    if args.true_dag is not None:
        assert '.dot' in args.true_dag, "True DAG must be in .dot format"
        assert os.path.isfile(args.true_dag), "True DAG file does not exist"
        run_values['ref_graph'] = utils.graph_from_dot_file(args.true_dag)

    if args.load_model and not os.path.isfile(args.load_model):
        raise FileNotFoundError("Model file does not exist")
    run_values['load_model'] = args.load_model

    if args.no_train:
        run_values['no_train'] = True
    else:
        run_values['no_train'] = False

    # Determine where to save the model pickle.
    if args.save_model == '':
        save_model = f"{args.dataset.replace('.csv', '')}"
        save_model = f"{save_model}_{args.method}.pickle"
        run_values['save_model'] = os.path.basename(save_model)
        # Output_path is the current directory
        run_values['output_path'] = os.getcwd()
        run_values['model_filename'] = utils.valid_output_name(
            filename=save_model, path=run_values['output_path'])
    elif args.save_model is not None:
        run_values['save_model'] = args.save_model
        run_values['output_path'] = os.path.dirname(args.save_model)
        run_values['model_filename'] = args.save_model
    else:
        run_values['save_model'] = None
        run_values['output_path'] = None

    # Set default regressors in case ReX is called.
    if args.method == 'rex' and args.regressor is None:
        run_values['regressors'] = DEFAULT_REGRESSORS
        run_values['seed'] = args.seed if args.seed is not None else DEFAULT_SEED
        run_values['hpo_iterations'] = args.iterations \
            if args.iterations is not None else DEFAULT_HPO_TRIALS
        run_values['bootstrap_iterations'] = args.bootstrap \
            if args.bootstrap is not None else DEFAULT_BOOTSTRAP_TRIALS
        run_values['bootstrap_tolerance'] = args.threshold \
            if args.threshold is not None else DEFAULT_BOOTSTRAP_TOLERANCE
        run_values['quiet'] = True if args.quiet else False
    else:
        run_values['regressors'] = [args.method]

    run_values['verbose'] = True if args.verbose else False
    run_values['output_dag_file'] = args.output

    # show_run_values(run_values)

    # return a dictionary with all the new variables created
    return run_values


def header_():
    """
    Done with "Ogre" font from https://patorjk.com/software/taag/
    """
    print(
"""   ___                      _                 _       _       
  / __\\__ _ _   _ ___  __ _| | _____  ___ __ | | __ _(_)_ __  
 / /  / _` | | | / __|/ _` | |/ _ \\ \\/ / '_ \\| |/ _` | | '_ \\ 
/ /__| (_| | |_| \\__ \\ (_| | |  __/>  <| |_) | | (_| | | | | |
\\____/\\__,_|\\__,_|___/\\__,_|_|\\___/_/\\_\\ .__/|_|\\__,_|_|_| |_|
                                       |_|""")


def show_run_values(run_values):
    """
    Print the run values.

    Args:
        run_values (dict): A dictionary of run values
    """
    print("-----")
    print("Run values:")
    for k, v in run_values.items():
        if isinstance(v, pd.DataFrame):
            print(f"- {k}: {v.shape[0]}x{v.shape[1]} DataFrame")
            continue
        print(f"- {k}: {v}")

    print("-----")


def main():
    header_()
    args = parse_args()
    run_values = check_args_validity(args)
    discoverer = GraphDiscovery()

    if run_values['load_model'] is not None:
        discoverer.load_models(run_values['load_model'])
        # In case of REX, trainer contains multiple entries in the dictionary
        # and we need to retrieve the last one, but in the case of others, we
        # only need to retrieve the only entry.
        result = next(reversed(discoverer.trainer.values()))
    else:
        discoverer.create_experiments(
            run_values['dataset_name'],
            run_values['regressors'],
            run_values['dataset_filepath'],
            run_values['true_dag'],
            run_values['dataset_path'],
            run_values['output_path']
        )

    if not run_values['no_train']:
        discoverer.fit_experiments(
            run_values['estimator'], 
            run_values['verbose'], 
            run_values['hpo_iterations'], 
            run_values['bootstrap_iterations']
        )
        result = discoverer.combine_and_evaluate_dags(
            run_values['dataset_name'],
            run_values['estimator'],
            run_values['dataset_path'],
            run_values['output_path'],
            run_values.get('ref_graph'),
            list(run_values['data'].columns) if 'data' in run_values else None
        )

    discoverer.printout_results(result.dag, result.metrics)
    discoverer.save_model(
        run_values['output_path'],
        run_values['dataset_name'],
        run_values['estimator'],
        run_values.get('save_model', True)
    )

    if run_values['output_dag_file'] is not None:
        utils.graph_to_dot_file(result.dag, run_values['output_dag_file'])
        print(f"Saved DAG to {run_values['output_dag_file']}")


if __name__ == "__main__":
    main()
