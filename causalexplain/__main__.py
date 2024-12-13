#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# causalexplain/causalexplain.py
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
import pandas as pd

from causalexplain.common import (DEFAULT_BOOTSTRAP_TOLERANCE,
                                DEFAULT_BOOTSTRAP_TRIALS, DEFAULT_HPO_TRIALS,
                                DEFAULT_REGRESSORS, DEFAULT_SEED, utils)
from causalexplain.common.notebook import Experiment
from causalexplain.metrics.compare_graphs import evaluate_graph

SUPPORTED_METHODS = ['rex', 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears']


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
        '-s', '--save_model', type=str, required=False, nargs='?',
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
        run_values['true_dag'] = utils.graph_from_dot_file(args.true_dag)

    # Determine where to save the model pickle.
    if args.save_model is None or args.save_model == '':
        save_model = f"{args.dataset.replace('.csv', '')}.pickle"
        run_values['save_model'] = os.path.basename(save_model)
        # Output_path is the current directory
        run_values['output_path'] = os.getcwd()
        run_values['model_filename'] = utils.valid_output_name(
            filename=save_model, path=run_values['output_path'])
    else:
        run_values['save_model'] = args.save_model
        run_values['output_path'] = os.path.dirname(args.save_model)
        run_values['model_filename'] = args.save_model

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
    if args.output is None:
        run_values['output_dag_file'] = utils.valid_output_name(
            filename=run_values['dataset_name'], 
            path=run_values['output_path'], 
            extension="dot")
    else:
        run_values['output_dag_file'] = args.output

    # show_run_values(run_values)

    # return a dictionary with all the new variables created
    return run_values


def create_experiments(**args):
    """
    Create an Experiment object for each regressor.

    Returns:
        dict: A dictionary of Experiment objects
    """
    trainer = {}
    for model_type in args['regressors']:
        trainer_name = f"{args['dataset_name']}_{model_type}"
        trainer[trainer_name] = Experiment(
            experiment_name=f"{args['dataset_name']}",
            model_type=model_type,
            input_path=args['dataset_path'],
            output_path=args['output_path'],
            verbose=False)
        trainer[trainer_name].ref_graph = args['true_dag']
        # print(f"Created experiment named '{trainer_name}'", flush=True)

    return trainer


def fit_experiments(trainer, run_values):
    """
    Fit the Experiment objects.

    Args:
        trainer (dict): A dictionary of Experiment objects
        run_values (dict): A dictionary of run values
    """
    if run_values['estimator'] == 'rex':
        xargs = {
            'verbose': run_values['verbose'],
            'hpo_n_trials': run_values['hpo_iterations'],
            'bootstrap_trials': run_values['bootstrap_iterations']
        }
    else:
        xargs = {
            'verbose': run_values['verbose']
        }

    for trainer_name, experiment in trainer.items():
        experiment.fit_predict(estimator=run_values['estimator'], **xargs)


def combine_and_evaluate_dags(trainer, run_values):
    """
    Retrieve the DAG from the Experiment objects.

    Args:
        trainer (dict): A dictionary of Experiment objects
        run_values (dict): A dictionary of run values

    Returns:
        nx.DiGraph: The combined DAG
    """
    if run_values['estimator'] != 'rex':
        trainer_key = f"{run_values['dataset_name']}_{run_values['estimator']}"
        estimator = getattr(trainer[trainer_key], run_values['estimator'])
        trainer[trainer_key].dag = estimator.dag
        if run_values['true_dag'] is not None:
            trainer[trainer_key].metrics = evaluate_graph(
                run_values['true_dag'], estimator.dag, 
                list(run_values['data'].columns))
        else:
            trainer[trainer_key].metrics = None

        return trainer[trainer_key]

    estimator1 = getattr(trainer[list(trainer.keys())[0]], 'rex')
    estimator2 = getattr(trainer[list(trainer.keys())[1]], 'rex')
    _, _, dag, _ = utils.combine_dags(estimator1.dag, estimator2.dag,
                                      estimator1.shaps.shap_discrepancies)
    trainer[f"{run_values['dataset_name']}_rex"] = Experiment(
        experiment_name=f"{run_values['dataset_name']}",
        model_type='rex',
        input_path=run_values['dataset_path'],
        output_path=run_values['output_path'],
        verbose=False)

    # Create a new trainer with the combined DAG.
    new_trainer = f"{run_values['dataset_name']}_rex"
    trainer[new_trainer].ref_graph = run_values['true_dag']
    trainer[new_trainer].dag = dag
    if run_values['true_dag'] is not None:
        trainer[new_trainer].metrics = evaluate_graph(
            run_values['true_dag'], dag, list(run_values['data'].columns))
    else:
        trainer[new_trainer].metrics = None

    return trainer[new_trainer]


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


def printout_results(graph, metrics):
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


def main():
    header_()
    args = parse_args()
    run_values = check_args_validity(args)

    trainer = create_experiments(**run_values)
    fit_experiments(trainer, run_values)
    result = combine_and_evaluate_dags(trainer, run_values)
    printout_results(result.dag, result.metrics)


if __name__ == "__main__":
    main()
