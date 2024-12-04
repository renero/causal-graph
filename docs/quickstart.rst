Quickstart
==========

This guide will help you get started with CausalGraph.

Basic Usage
----------

Here's a simple example of how to use CausalGraph:

.. code-block:: python

   import causalgraph as cg
   
   # Load your dataset
   data = cg.load_dataset('toy_dataset.csv')
   
   # Create a causal graph
   graph = cg.CausalGraph(data)
   
   # Fit the model
   graph.fit()
   
   # Plot the causal graph
   graph.plot()

For more examples and detailed usage, please refer to the API documentation.
