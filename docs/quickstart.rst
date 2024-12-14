Quickstart
==========

This guide will help you get started with CausalExplain.

Basic Usage
----------

Here's a simple example of how to get CausalExplain help from the command line:

.. code-block:: bash

   python -m causalexplain --help

In order to run CausalExplain from the command line, you need to have Python 3.10
or later installed on your system. To install CausalExplain, run the following
command:

.. code-block:: bash

   pip install causalexplain

Once CausalExplain is installed, you can run it from the command line by typing
``python -m causalexplain``.

To run a simple case with a ``toy_dataset.csv`` file using ReX model, you can 
use the following command, assuming default parameters:

.. code-block:: bash

   python -m causalexplain -d /path/to/toy_dataset.csv

That will generate the ReX model and run the model on the dataset, and print
the results to the terminal, like this:

.. code-block:: bash

   Resulting Graph:
   ---------------
   X1 -> X2
     X2 -> X4
     X2 -> X3
   X1 -> X4

which is the true graph expected.


Input Arguments Information
--------------------------

The basic arguments are:

* ``-d`` or ``--dataset``: The path to the dataset file in CSV format.
* ``-t`` or ``--true_dag``: The path to the true DAG file in DOT format.
* ``-m`` or ``--method``: The method to use to infer the causal graph.

These options allow you to specify the dataset, true DAG, and method to be used. 
In case you don't have a true DAG, the result is the plausible causal graph, 
which is the causal graph that is inferred by the method without taking into
account the true DAG.

Regarding the output of the ``causalexplain`` command, the following information is 
provided:

- The plausible causal graph, which is the causal graph that is inferred by
the method without taking into account the true DAG.
- The metrics obtained from the evaluation of the causal graph against the true
DAG.

In those cases where training or running a method takes a long time, ``causalexplain`` 
allows you to save the model (``-s`` or ```--save_model```) trained in a file and 
load it later. To load the model, use the ``-l`` or ``--load_model`` option.

The option ``-b`` or ``--bootstrap`` allows you to specify the number of iterations
for bootstrap in the ReX method. 
The default value is 20, but you can change it to a different
value, to test the effect of the number of iterations on the performance of the
method. This option is linked to the next one, ``-T``.

The option ``-T`` or ``--threshold`` allows you to specify a threshold for the 
bootstrapped adjacency matrix computed for the ReX method. The default value is 
0.3, but you can change it to a different value, to test the effect of the 
threshold on the performance of the method. Lower values in the adjacency matrix
represent edges that appear less frequently in the bootstrap samples, while higher
values represent edges that appear more frequently. So, a higher threshold
represents a more conservative approach to the inference of the causal graph.

The option ``-r`` or ``--regressor`` allows you to specify a list of comma-separated
names of the regressors to be used. The default value is ``dnn,gbt``, but you can
change it to a different list of regressors. Current implementation only supports
DNN and GBT regressors, but they can be extended in the future.

The option ``-u`` or ``--union`` allows you to specify a list of comma-separated
names of the DAGs to be unioned. This option is only valid for the ReX method, 
and it is used to combine the causal graphs inferred by the method with different
hyperparameters. By default, ReX combines the DAGs inferred with the DNN and 
GBT regressors, but you can extend ReX with more regressors and combine them 
with different hyperparameters.

The option ``-i`` or ``--iterations`` allows you to specify the number of iterations
that the hyper-parameter optimization will perform in the ReX method. The default
value is 100, but you can change it to a different value, to test the effect of
the number of iterations on the performance of the method.

The option ``-S`` or ``--seed`` allows you to specify a seed for the random number
generator. The default value is 1234, but you can change it to a different value,
to test the effect of the seed on the performance of the method.

The option ``-o`` or ``--output`` allows you to specify the path to the output file
where the resulting DAG will be saved in DOT format. The default value is
``./output.dot``, but you can change it to a different value, to save the DAG in a
different file.
