#!/usr/bin/env python3

import os
import pkgutil

def generate_rst_file(module_name, output_dir):
    """Generate an RST file for the given module."""
    output_file = os.path.join(output_dir, f"{module_name}.rst")
    
    content = f"""{module_name}
{'=' * len(module_name)}

.. automodule:: {module_name}
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
"""
    
    with open(output_file, 'w') as f:
        f.write(content)

def main():
    # Create api directory if it doesn't exist
    api_dir = "api"
    os.makedirs(api_dir, exist_ok=True)
    
    # Generate the main modules.rst file
    with open(os.path.join(api_dir, "modules.rst"), "w") as f:
        f.write("""Modules
=======

.. toctree::
   :maxdepth: 4

   causalgraph
""")
    
    # Generate the main package RST
    with open(os.path.join(api_dir, "causalgraph.rst"), "w") as f:
        f.write("""causalgraph
==========

.. automodule:: causalgraph
   :members:
   :undoc-members:
   :show-inheritance:

Submodules
----------

.. toctree::
   :maxdepth: 4

   causalgraph.common
   causalgraph.estimators
   causalgraph.explainability
   causalgraph.generators
   causalgraph.independence
   causalgraph.metrics
   causalgraph.models
""")
    
    # Generate RST files for each submodule
    submodules = [
        "causalgraph.common",
        "causalgraph.estimators",
        "causalgraph.explainability",
        "causalgraph.generators",
        "causalgraph.independence",
        "causalgraph.metrics",
        "causalgraph.models"
    ]
    
    for submodule in submodules:
        generate_rst_file(submodule, api_dir)

if __name__ == "__main__":
    main()
