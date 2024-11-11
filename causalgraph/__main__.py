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

from causalgraph.common import utils
from causalgraph.common import (
    DEFAULT_BOOTSTRAP_TOLERANCE, DEFAULT_SEED, DEFAULT_REGRESSORS
)
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
        help='List of DAGs from regressor to combine. Default is all of them.')
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
        '-s', '--save_model', type=str, required=False,
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
        estimator = args.method

    # Check that the dataset file exist, and load it (data)
    assert args.dataset is not None, "Dataset file must be specified"
    assert os.path.isfile(args.dataset), "Dataset file does not exist"
    data = pd.read_csv(args.dataset)

    # Extract the path from where the dataset is, dataset basename
    dataset_path = os.path.dirname(args.dataset)
    dataset_name = os.path.basename(args.dataset)

    # Load true DAG, if specified (true_dag)
    true_dag = None
    if args.true_dag is not None:
        assert '.dot' in args.true_dag, "True DAG must be in .dot format"
        assert os.path.isfile(args.true_dag), "True DAG file does not exist"
        true_dag = utils.graph_from_dot_file(args.true_dag)

    # Determine where to save the model pickle.
    if args.save_model is None:
        save_model = f"{args.dataset.replace('.csv', '')}.pickle"
        save_model = os.path.basename(save_model)
        # Output_path is the current directory
        output_path = os.getcwd()
    else:
        save_model = args.save_model
        output_path = os.path.dirname(save_model)

    # Set default regressors in case ReX is called.
    if args.method == 'rex' and args.regressor is None:
        regressors = DEFAULT_REGRESSORS

    seed = args.seed if args.seed is not None else DEFAULT_SEED
    bootstrap_tolerance = args.threshold if args.threshold is not None \
        else DEFAULT_BOOTSTRAP_TOLERANCE

    verbose = True if args.verbose else False
    quiet = True if args.quiet else False
    output_dag_file = args.output

    # return a dictionary with all the new variables created
    return {
        'estimator': estimator,
        'regressors': regressors,
        'data': data,
        'dataset_name': dataset_name,
        'dataset_path': dataset_path,
        'true_dag': true_dag,
        'save_model': save_model,
        'output_path': output_path,
        'seed': seed,
        'bootstrap_tolerance': bootstrap_tolerance,
        'verbose': verbose,
        'quiet': quiet,
        'output_dag_file': output_dag_file
    }


def train_rex(**args):
    for model_type in args['regressors']:
        trainer = Experiment(
            experiment_name=args['dataset_name'],
            model_type=model_type,
            input_path=args['dataset_path'],
            output_path=args['output_path'],
            verbose=False)
        trainer.ref_graph = args.true_dag
        print(f"Created experiment named '{args['dataset_name']}'", flush=True)
        # fit(experiment, model_type)
        # predict(experiment)
        # save_experiment(experiment)


def main():
    args = parse_args()
    run_values = check_args_validity(args)
    # for k,v in run_values.items():
    #     print(f"{k}={v}")


if __name__ == "__main__":
    main()
