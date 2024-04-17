[![Travis](https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master)](https://travis-ci.org/scikit-learn-contrib/project-template)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true)](https://ci.appveyor.com/project/glemaitre/project-template)
[![Codecov](https://codecov.io/gh/renero/causal-graph/graph/badge.svg?token=HCV0IJDFLQ)](https://codecov.io/gh/renero/causal-graph)
[![ReadTheDocs](https://readthedocs.org/projects/sklearn-template/badge/?version=latest)](https://sklearn-template.readthedocs.io/en/latest/?badge=latest)

# causal-graph - A library to infer causal-effect relationships from continuous tabular data

[scikit-learn](https://scikit-learn.org)

**causal-graph** is a library that implements methods to extract the causal graph,
from continuous tabular data.

It is built using SKLearn estimators, so that it can be used in scikit-learn pipelines
and (hyper)parameter search, while facilitating testing (including some API
compliance), documentation, open source development, packaging, and continuous
integration.

**NOTICE**: This library is still in development and should be used with caution. Feel free to contact me to provide more details on how to use it.

The datasets used in the examples can be generated using the `generators` module,
which is also part of this library. But in case you want to reproduce results from the
articles that we used as reference, you can find the datasets in the `data` folder.

## Installation

At the moment, the library is not available in PyPI. You can install it from the source code:

```bash
git clone
cd causal-graph
pip install .
```

