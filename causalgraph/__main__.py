#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# causalgraph/causalgraph.py
#
# (C) 2024 J. Renero
#
# This file is part of causalgraph
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

from causalgraph.common import (DEFAULT_BOOTSTRAP_TOLERANCE,
                                DEFAULT_BOOTSTRAP_TRIALS, DEFAULT_HPO_TRIALS,
                                DEFAULT_REGRESSORS, DEFAULT_SEED, utils)
from causalgraph.common.notebook import Experiment


def parse_args():
    class SplitArgs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values.split(','))

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
    # Set model type (estimator)
    if args.method is None:
        args.method = 'rex'
        estimator = 'rex'
    else:
        assert args.method in ['rex', 'pc', 'fci',
                               'ges', 'lingam', 'cam', 'notears']
        "Method must be one of: 'rex', 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears'"
        estimator = str(args.method)

    # Check that the dataset file exist, and load it (data)
    assert args.dataset is not None, "Dataset file must be specified"
    assert os.path.isfile(args.dataset), "Dataset file does not exist"
    data = pd.read_csv(args.dataset)
    dataset_filepath = args.dataset

    # Extract the path from where the dataset is, dataset basename
    dataset_path = os.path.dirname(args.dataset)
    dataset_name = os.path.basename(args.dataset)
    dataset_name = dataset_name.replace('.csv', '')

    # Load true DAG, if specified (true_dag)
    true_dag = None
    if args.true_dag is not None:
        assert '.dot' in args.true_dag, "True DAG must be in .dot format"
        assert os.path.isfile(args.true_dag), "True DAG file does not exist"
        true_dag = utils.graph_from_dot_file(args.true_dag)

    # Determine where to save the model pickle.
    if args.save_model is None or args.save_model == '':
        save_model = f"{args.dataset.replace('.csv', '')}.pickle"
        save_model = os.path.basename(save_model)
        # Output_path is the current directory
        output_path = os.getcwd()
        model_filename = utils.valid_output_name(
            filename=save_model, path=output_path)
    else:
        save_model = args.save_model
        output_path = os.path.dirname(save_model)
        model_filename = args.save_model

    # Set default regressors in case ReX is called.
    if args.method == 'rex' and args.regressor is None:
        regressors = DEFAULT_REGRESSORS
    else:
        regressors = [args.method]

    seed = args.seed if args.seed is not None else DEFAULT_SEED

    hpo_iterations = args.iterations if args.iterations is not None \
        else DEFAULT_HPO_TRIALS
    bootstrap_iterations = args.bootstrap if args.bootstrap is not None \
        else DEFAULT_BOOTSTRAP_TRIALS
    bootstrap_tolerance = args.threshold if args.threshold is not None \
        else DEFAULT_BOOTSTRAP_TOLERANCE

    verbose = True if args.verbose else False
    quiet = True if args.quiet else False
    if args.output is None:
        output_dag_file = utils.valid_output_name(
            filename=dataset_name, path=output_path, extension="dot")
    else:
        output_dag_file = args.output

    # return a dictionary with all the new variables created
    return {
        'estimator': estimator,
        'regressors': regressors,
        'data': data,
        'dataset_filepath': dataset_filepath,
        'dataset_name': dataset_name,
        'dataset_path': dataset_path,
        'true_dag': true_dag,
        'model_filename': model_filename,
        'output_path': output_path,
        'seed': seed,
        'hpo_iterations': hpo_iterations,
        'bootstrap_iterations': bootstrap_iterations,
        'bootstrap_tolerance': bootstrap_tolerance,
        'verbose': verbose,
        'quiet': quiet,
        'output_dag_file': output_dag_file
    }


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
        print(
            f"Created experiment named '{trainer_name}'", flush=True)

    return trainer


def fit_experiments(trainer, run_values):
    """
    Fit the Experiment objects.

    Args:
        trainer (dict): A dictionary of Experiment objects
        run_values (dict): A dictionary of run values
    """
    xargs = {
        'verbose': run_values['verbose'],
        'hpo_n_trials': run_values['hpo_iterations'],
        'bootstrap_trials': run_values['bootstrap_iterations']
    }

    for trainer_name, experiment in trainer.items():
        experiment.fit_predict(estimator=run_values['estimator'], **xargs)


def retrieve_dag(trainer, run_values):
    """
    Retrieve the DAG from the Experiment objects.

    Args:
        trainer (dict): A dictionary of Experiment objects
        run_values (dict): A dictionary of run values

    Returns:
        nx.DiGraph: The combined DAG
    """
    if run_values['estimator'] != 'rex':
        estimator = getattr(trainer, run_values['estimator'])
        return estimator.dag

    estimator1 = getattr(trainer[list(trainer.keys())[0]], 'rex')
    estimator2 = getattr(trainer[list(trainer.keys())[1]], 'rex')
    _, _, dag, _ = utils.combine_dags(estimator1.dag, estimator2.dag,
                                      estimator1.shaps.shap_discrepancies)

    return dag


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
            print(f"{k}: {v.shape[0]}x{v.shape[1]} DataFrame")
            continue
        print(f"{k}: {v}")

    print("-----")


def header_():
    print(
        """
   ____                      _  ____                 _
  / ___|__ _ _   _ ___  __ _| |/ ___|_ __ __ _ _ __ | |__
 | |   / _` | | | / __|/ _` | | |  _| '__/ _` | '_ \| '_ \\
 | |__| (_| | |_| \__ \ (_| | | |_| | | | (_| | |_) | | | |
  \____\__,_|\__,_|___/\__,_|_|\____|_|  \__,_| .__/|_| |_|
                                              |_|
""")


def main():
    header_()
    args = parse_args()
    run_values = check_args_validity(args)
    show_run_values(run_values)

    trainer = create_experiments(**run_values)
    fit_experiments(trainer, run_values)
    dag = retrieve_dag(trainer, run_values)
    print(dag)


if __name__ == "__main__":
    main()
