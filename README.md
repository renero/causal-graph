[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-31012/)
[![Travis](https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master)](https://travis-ci.org/scikit-learn-contrib/project-template)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true)](https://ci.appveyor.com/project/glemaitre/project-template)
[![Codecov](https://codecov.io/gh/renero/causalgraph/graph/badge.svg?token=HCV0IJDFLQ)](https://codecov.io/gh/renero/causalgraph)
[![ReadTheDocs](https://readthedocs.org/projects/sklearn-template/badge/?version=latest)](https://sklearn-template.readthedocs.io/en/latest/?badge=latest)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# causalgraph - A library to infer causal-effect relationships from continuous tabular data

**causalgraph** is a library that implements methods to extract the causal graph,
from continuous tabular data, specifically the **ReX** method, and other compared methods
like GES, PC, FCI, LiNGAM, CAM, and NOTEARS.

ReX isa causal discovery method that leverages machine learning (ML) models 
coupled with explainability techniques, specifically Shapley values, to 
identify and interpret significant causal relationships among variables. 
Comparative evaluations on synthetic datasets comprising continuous tabular 
data reveal that **ReX** outperforms state-of-the-art causal discovery methods 
across diverse data generation processes, including non-linear and additive 
noise models. Moreover, ReX was tested on the Sachs single-cell 
protein-signaling dataset, achieving a precision of 0.952 and recovering 
key causal relationships with no incorrect edges. Taking together, these 
results showcase ReX’s effectiveness in accurately recovering true causal 
structures while minimizing false positive pre- dictions, its robustness 
across diverse datasets, and its applicability to real-world problems. 
By combining ML and explainability techniques with causal discovery, **ReX** 
bridges the gap between predictive modeling and causal inference, offering an 
effective tool for understanding complex causal structures.

![ReX](https://raw.githubusercontent.com/renero/causalgraph/main/docs/_static/rex.png)

It is built using SKLearn estimators, so that it can be used in scikit-learn 
pipelines and (hyper)parameter search, while facilitating testing (including 
some API compliance), documentation, open source development, packaging, 
and continuous integration.

The datasets used in the examples can be generated using the `generators` 
module, which is also part of this library. But in case you want to 
reproduce results from the articles that we used as reference, you can find 
the datasets in the `data` folder.

## Installation

At the moment, the library is not available in PyPI. You can install it from the source code:

```bash
git clone
cd causalgraph
pip install .
```

