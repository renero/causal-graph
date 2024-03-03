.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn-contrib/project-template

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/glemaitre/project-template

.. |Codecov| image:: https://codecov.io/gh/renero/causal-graph/graph/badge.svg?token=HCV0IJDFLQ 
.. _Codecov: https://codecov.io/gh/renero/causal-graph

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/sklearn-template/badge/?version=latest
.. _ReadTheDocs: https://sklearn-template.readthedocs.io/en/latest/?badge=latest

causal-graph - A library to infer causal-effect relationships from tabular data
==================================================================================

.. _scikit-learn: https://scikit-learn.org

**causal-graph** is a library that implements methods to extract the causal graph, 
from continuous tabular data.

It is build using SKLearn estimators, so that it can be used in scikit-learn pipelines
and (hyper)parameter search, while facilitating testing (including some API
compliance), documentation, open source development, packaging, and continuous
integration.

**NOTICE**: This library is still in development and should be used with caution.

The datasets used in the examples can be generated using the `generators` module, 
which is also part of this library. But in case you want to reproduce results from the 
articles that we used as reference, you can find the datasets in the `data` folder.