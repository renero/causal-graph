#!/usr/bin/env python3

import os

MODULES = [
    'common',
    'estimators',
    'explainability',
    'generators',
    'independence',
    'metrics',
    'models'
]

TEMPLATE = """causalgraph.{module}
==================

.. automodule:: causalgraph.{module}
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :recursive:

   causalgraph.{module}
"""

def main():
    for module in MODULES:
        filename = f'causalgraph.{module}.rst'
        with open(filename, 'w') as f:
            f.write(TEMPLATE.format(module=module))
        print(f'Created {filename}')

if __name__ == '__main__':
    main()
