![logo](https://raw.githubusercontent.com/renero/causalgraph/main/docs/_static/logo-light.png)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-31012/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey.svg)](#)
[![PyPI version](https://badge.fury.io/py/causalexplain.svg)](https://badge.fury.io/py/causalexplain)
[![Build Status](https://github.com/renero/causalgraph/actions/workflows/build.yaml/badge.svg)](https://github.com/renero/causalgraph/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/renero/causalgraph/graph/badge.svg?token=HCV0IJDFLQ)](https://codecov.io/gh/renero/causalgraph)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://renero.github.io/causalgraph/)


# CausalExplain - A library to infer causal-effect relationships from tabular data

'**CausalExplain**' is a library that implements methods to extract the causal 
graph, from tabular data, specifically the **ReX** method, and other compared 
methods like GES, PC, FCI, LiNGAM, CAM, and NOTEARS.

This repository contains the implementation of **ReX** and all necessary tools 
to reproduce the results presented in our accompanying paper. **ReX** supports 
diverse data generation processes, including non-linear and additive noise 
models, and has demonstrated robust performance on synthetic and real-world 
datasets.

## About **ReX**

**ReX** is a causal discovery method that leverages machine learning (ML) models 
coupled with explainability techniques, specifically Shapley values, to 
identify and interpret significant causal relationships among variables. 
Comparative evaluations on synthetic datasets comprising tabular data reveal that 
**ReX** outperforms state-of-the-art causal discovery methods across diverse data 
generation processes, including non-linear and additive noise models. Moreover, 
**ReX** was tested on the Sachs single-cell protein-signaling dataset, achieving a 
precision of 0.952 and recovering key causal relationships with no incorrect 
edges. Taking together, these results showcase **ReX**'s effectiveness in 
accurately recovering true causal structures while minimizing false positive 
predictions, its robustness across diverse datasets, and its applicability to 
real-world problems. By combining ML and explainability techniques with causal 
discovery, **ReX** bridges the gap between predictive modeling and causal 
inference, offering an effective tool for understanding complex causal 
structures.

![ReX Schema](https://raw.githubusercontent.com/renero/causalgraph/main/docs/_static/REX.png)

Our experimental results, conducted on five families of synthetic datasets with 
varying complexity, demonstrate that REX consistently recovers true causal 
relationships with high precision while minimizing false positives and orientation
errors, comparing favorably to existing methods. Additionally, REX was tested on 
the Sachs single-cell protein-signaling dataset (Sachs et al., 2005), achieving 
a competitive performance with no false positives and recovering important causal 
relationships. This further validates the applicability of REX to real-world 
datasets, highlighting its robustness across different types of data.

## Prerequisites without Docker

- Operating System: Linux or macOS
- Environment Manager: PyEnv or Conda
- Programming Language: Python 3.10.12 or higher
- Hardware: CPU

## Installation

The project can be installed using pip:

```bash
$ pip install causalexplain
```

## Data

The datasets used in the paper and the examples can be generated using the 
`generators` module, which is also part of this library. In case you want to 
reproduce results from the articles that we used as reference, you can find 
the datasets in the `data` folder.

## Executing `causalexplain`

### Option 1: Command Line

To run `causalexplain` on your data, you can use the `causalexplain` command:

```
$ python -m causalexplain
   ___                      _                 _       _       
  / __\__ _ _   _ ___  __ _| | _____  ___ __ | | __ _(_)_ __  
 / /  / _` | | | / __|/ _` | |/ _ \ \/ / '_ \| |/ _` | | '_ \ 
/ /__| (_| | |_| \__ \ (_| | |  __/>  <| |_) | | (_| | | | | |
\____/\__,_|\__,_|___/\__,_|_|\___/_/\_\ .__/|_|\__,_|_|_| |_|
                                       |_|                                        
usage: causalexplain [-h] -d DATASET [-m {rex,pc,fci,ges,lingam,cam,notears}] 
                   [-t TRUE_DAG] [-l LOAD_MODEL] [-T THRESHOLD] [-u UNION] 
                   [-i ITERATIONS] [-b BOOTSTRAP] [-r REGRESSOR] [-S SEED] 
                   [-s [SAVE_MODEL]] [-n] [-v] [-q] [-o OUTPUT]
```

that will present you with a menu to choose the dataset you want to use, the 
method you want to use to infer the causal graph, and the hyperparameters you
want to use.

The minimum required to run `causalexplain` is a dataset file in CSV format,
with the first row containing the names of the variables, and the rest of
the rows containing the values of the variables. The method selected by default
is ReX, but you can also choose between PC, FCI, GES, LiNGAM, CAM, NOTEARS. 
At the end of the execution, the edges of the plausible causal graph will be 
displayed along with the metrics obtained, if the true dag is provided 
(argument `-t`).

### Option 2: Notebook

In case you want to run `causalexplain` from your code in a notebook, you can
use the `GraphDiscovery` class. The following example shows how to use 
the `GraphDiscovery` class to train a model on a dataset using **ReX** method:

```python
from causalexplain import GraphDiscovery

experiment = GraphDiscovery(
   experiment_name='my_experiment',
   model_type='rex',
   csv_filename='data.csv',
   dot_filename='true_graph.dot')

# Run the experiments
experiment.run()

# Plot the resulting DAG
experiment.plot()

# Save the trained model to a file
experiment.save("/path/to/model.pkl")
```

To load a model from a file, you can use the `load` method of the 
`GraphDiscovery` class:

```python
from causalexplain import GraphDiscovery

experiment = GraphDiscovery()
experiment.load("/path/to/model.pkl")
```

This can be useful if you want to train a model on a dataset and then use it 
to predict causal graphs on other datasets, or train a model on different 
batches.

To export the predicted causal graph to a DOT file, you can use the `export` 
method of the `GraphDiscovery` class:

```python
experiment.export("/path/to/my_predicted_graph.dot")
```

### Output

The output of `causalexplain` is typically a graph with the edges of the 
plausible causal graph and the metrics obtained from the evaluation of the 
causal graph against the true DAG. These results are printed to the console, 
unless the '-o' option is specified, in which case the DAG is saved to a 
file in DOT format. Metrics are printed only if the true DAG is provided.

## Example commands

The following command illustrates how to run `causalexplain` on the toy dataset
using the ReX method:

```bash
$ python -m causalexplain -d /path/to/toy_dataset.csv -t /path/to/toy_dataset.dot
```

The same command can be used to run `causalexplain` on the toy dataset using the
CAM method:

```bash
$ python -m causalexplain -d /path/to/toy_dataset.csv -m cam -t /path/to/toy_dataset.dot
```

For more information on command line options, run `causalexplain -h` or go to 
the [Quickstart](https://renero.github.io/causalgraph/quickstart.html) 
section in the documentation.
