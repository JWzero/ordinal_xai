ordinal\_xai.utils package
==========================

The utils package provides various utility functions for ordinal regression and explainable AI tasks. This package includes tools for data handling, model evaluation, and partial dependence plot modifications.

Submodules
----------

ordinal\_xai.utils.data\_utils module
-------------------------------------

.. automodule:: ordinal_xai.utils.data_utils
   :members:
   :undoc-members:
   :show-inheritance:

This module provides functions for data loading and preprocessing:
- ``load_data``: Load and prepare datasets for ordinal regression
- ``transform_features``: Transform features for model training

ordinal\_xai.utils.evaluation\_metrics module
---------------------------------------------

.. automodule:: ordinal_xai.utils.evaluation_metrics
   :members:
   :undoc-members:
   :show-inheritance:

This module contains functions for evaluating ordinal regression models:
- ``evaluate_ordinal_model``: Comprehensive evaluation of ordinal models
- ``print_evaluation_results``: Display evaluation metrics in a readable format

ordinal\_xai.utils.pdp\_modified module
---------------------------------------

.. automodule:: ordinal_xai.utils.pdp_modified
   :members:
   :undoc-members:
   :show-inheritance:

This module provides modified partial dependence plot functionality for ordinal regression:
- Custom implementations of PDP for ordinal models
- Visualization tools for feature effects

ordinal\_xai.utils._response\_modified module
--------------------------------------------

.. automodule:: ordinal_xai.utils._response_modified
   :members:
   :undoc-members:
   :show-inheritance:

This module contains internal utilities for response modification:
- Helper functions for response transformation
- Support for ordinal model predictions

Module contents
---------------

.. automodule:: ordinal_xai.utils
   :members:
   :undoc-members:
   :show-inheritance:

The main utils package exports the following functions:
- ``load_data``: Load and prepare datasets
- ``transform_features``: Transform features for model training
- ``evaluate_ordinal_model``: Evaluate ordinal regression models
- ``print_evaluation_results``: Display evaluation metrics
