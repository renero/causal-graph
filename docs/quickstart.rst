Quickstart
==========

This guide will help you get started with CausalGraph.

Basic Usage
----------

Here's a simple example of how to get CausalGraph help from the command line:

.. code-block:: bash

   python -m causalgraph --help

In order to run CausalGraph from the command line, you need to have Python 3.10
or later installed on your system. To install CausalGraph, run the following
command:

.. code-block:: bash

   pip install causalgraph

Once CausalGraph is installed, you can run it from the command line by typing
``python -m causalgraph``.

To run a simple case with a ``toy_dataset.csv`` file using ReX model, you can 
use the following command, assuming default parameters:

.. code-block:: bash

   python -m causalgraph -d /path/to/toy_dataset.csv

That will generate the ReX model and run the model on the dataset, and print
the results to the terminal, like this:

.. code-block:: bash
   