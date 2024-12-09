<img src="./_static/logo-light.png" alt="logo" width="300">

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-31012/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey.svg)](#)
[![Build Status](https://github.com/renero/causalgraph/actions/workflows/codecov.yaml/badge.svg)](https://github.com/renero/causalgraph/actions/workflows/codecov.yaml)
[![codecov](https://codecov.io/gh/renero/causalgraph/graph/badge.svg?token=HCV0IJDFLQ)](https://codecov.io/gh/renero/causalgraph)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://renero.github.io/causalgraph/)


# causalgraph - A library to infer causal-effect relationships from tabular data

'**causalgraph**' is a library that implements methods to extract the causal graph,
from tabular data, specifically the **ReX** method, and other compared methods
like GES, PC, FCI, LiNGAM, CAM, and NOTEARS.

**ReX** is a causal discovery method that leverages machine learning (ML) models 
coupled with explainability techniques, specifically Shapley values, to 
identify and interpret significant causal relationships among variables. 
Comparative evaluations on synthetic datasets comprising tabular data reveal that 
**ReX** outperforms state-of-the-art causal discovery methods across diverse data 
generation processes, including non-linear and additive noise models. Moreover, 
**ReX** was tested on the Sachs single-cell protein-signaling dataset, achieving a 
precision of 0.952 and recovering 
key causal relationships with no incorrect edges. Taking together, these 
results showcase **ReX**’s effectiveness in accurately recovering true causal 
structures while minimizing false positive pre- dictions, its robustness 
across diverse datasets, and its applicability to real-world problems. 
By combining ML and explainability techniques with causal discovery, **ReX** 
bridges the gap between predictive modeling and causal inference, offering an 
effective tool for understanding complex causal structures.

![ReX Schema](./_static/REX.png)

It is built using SKLearn estimators, so that it can be used in scikit-learn 
pipelines and (hyper)parameter search, while facilitating testing (including 
some API compliance), documentation, open source development, packaging, 
and continuous integration.

The datasets used in the examples can be generated using the `generators` 
module, which is also part of this library. But in case you want to 
reproduce results from the articles that we used as reference, you can find 
the datasets in the `data` folder.

## Prerequisites without Docker

- Operating System: Linux or macOS
- Environment Manager: PyEnv or Conda
- Programming Language: Python 3.10.12 or higher
- Hardware: CPU

## Installation

In the comming days the library will be made available in PyPI. 
In the meantime, you can install it from the source code:

```bash
$ git clone
$ cd causalgraph
$ pip install .
```

## Data

The datasets used to reproduce the results presented in the manuscript are 
available under the `data` folder. The datasets were generated using the
`generators` module.

## Executing `causalgraph`

To run `causalgraph` on your data, you can use the `causalgraph` command:

```
$ python -m causalgraph
   ____                      _  ____                 _
  / ___|__ _ _   _ ___  __ _| |/ ___|_ __ __ _ _ __ | |__
 | |   / _` | | | / __|/ _` | | |  _| '__/ _` | '_ \| '_ \
 | |__| (_| | |_| \__ \ (_| | | |_| | | | (_| | |_) | | | |
  \____\__,_|\__,_|___/\__,_|_|\____|_|  \__,_| .__/|_| |_|
                                              |_|
usage: causalgraph [-h] -d DATASET [-m {rex,pc,fci,ges,lingam,cam,notears}] 
                   [-t TRUE_DAG] [-l LOAD_MODEL] [-T THRESHOLD] [-u UNION] 
                   [-i ITERATIONS] [-b BOOTSTRAP] [-r REGRESSOR] [-S SEED] 
                   [-s [SAVE_MODEL]] [-v] [-q] [-o OUTPUT]
```

that will present you with a menu to choose the dataset you want to use, the 
method you want to use to infer the causal graph, and the hyperparameters you
want to use.

The minimum required to run `causalgraph` is a dataset file in CSV format,
with the first row containing the names of the variables, and the rest of
the rows containing the values of the variables. The method selected by default
is ReX, but you can also choose between PC, FCI, GES, LiNGAM, CAM, NOTEARS. 
At the end of the execution, the edges of the plausible causal graph will be 
displayed along with the metrics obtained, if the true dag is provided 
(argument `-t`).


## Example commands

The following command illustrates how to run `causalgraph` on the toy dataset
using the ReX method:

```bash
$ python -m causalgraph -d /path/to/toy_dataset.csv -t /path/to/toy_dataset.dot
```

The same command can be used to run `causalgraph` on the toy dataset using the
CAM method:

```bash
$ python -m causalgraph -d /path/to/toy_dataset.csv -m cam -t /path/to/toy_dataset.dot
```

## Input Arguments Information

The basic arguments are:

- `-d` or `--dataset`: The path to the dataset file in CSV format.
- `-t` or `--true_dag`: The path to the true DAG file in DOT format.
- `-m` or `--method`: The method to use to infer the causal graph.

These options allow you to specify the dataset, true DAG, and method to be used. 
In case you don't have a true DAG, the result is the plausible causal graph, 
which is the causal graph that is inferred by the method without taking into
account the true DAG.

Regarding the output of the `causalgraph` command, the following information is 
provided:

- The plausible causal graph, which is the causal graph that is inferred by
the method without taking into account the true DAG.
- The metrics obtained from the evaluation of the causal graph against the true
DAG.

In those cases where training or running a method takes a long time, `causalgraph` 
allows you to save the model (`-s` or `--save_model`) trained in a file and load it later. To load the model, use the `-l` or `--load_model` option.

The option `-b` or `--bootstrap` allows you to specify the number of iterations
for bootstrap in the ReX method. 
The default value is 20, but you can change it to a different
value, to test the effect of the number of iterations on the performance of the
method. This option is linked to the next one, `-T`.

The option `-T` or `--threshold` allows you to specify a threshold for the 
bootstrapped adjacency matrix computed for the ReX method. The default value is 
0.3, but you can change it to a different value, to test the effect of the 
threshold on the performance of the method. Lower values in the adjacency matrix
represent edges that appear less frequently in the bootstrap samples, while higher
values represent edges that appear more frequently. So, a higher threshold
represents a more conservative approach to the inference of the causal graph.

The optin `-r` or `--regressor` allows you to specify a list of comma-separated
names of the regressors to be used. The default value is `dnn,gbt`, but you can
change it to a different list of regressors. Current implementation only supports
DNN and GBT regressors, but they can be extended in the future.

The option `-u` or `--union` allows you to specify a list of comma-separated
names of the DAGs to be unioned. This option is only valid for the ReX method, 
and it is used to combine the causal graphs inferred by the method with different
hyperparameters. By default, ReX combines the DAGs inferred with the DNN and 
GBT regressors, but you can extend ReX with more regressors and combine them 
with different hyperparameters.

The option `-i` or `--iterations` allows you to specify the number of iterations
that the hyper-parameter optimization will perform in the ReX method. The default
value is 100, but you can change it to a different value, to test the effect of
the number of iterations on the performance of the method.

The option `-S` or `--seed` allows you to specify a seed for the random number
generator. The default value is 1234, but you can change it to a different value,
to test the effect of the seed on the performance of the method.

The option `-o` or `--output` allows you to specify the path to the output file
where the resulting DAG will be saved in DOT format. The default value is
`./output.dot`, but you can change it to a different value, to save the DAG in a
different file.

## Additional Information

WIP