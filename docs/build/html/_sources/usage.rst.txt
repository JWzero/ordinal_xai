Usage Guide
===========

Basic Usage
----------

The package provides a command-line interface for running ordinal regression models with interpretability methods. Here's a basic example:

.. code-block:: bash

   python main.py --dataset dummy.csv --model CLM --interpretation LIME --observation_idx 0

Command Line Arguments
--------------------

- ``--dataset``: Dataset filename in 'data/' folder (default: 'dummy.csv')
- ``--model``: Model filename (without .py) in 'models/' folder (default: 'CLM')
- ``--interpretation``: Interpretability method filename (without .py) in 'interpretation/' folder
- ``--link``: Link function for CLM model (default: 'logit'). Options: 'logit', 'probit'
- ``--observation_idx``: Index of the observation to interpret (only for local explanations)
- ``--features``: Comma-separated list of feature indices to include in the explanation
- ``--metrics``: Comma-separated list of metrics to use for LOCO interpretation
- ``--sampling``: Sampling method for LIME (default: None). Options: None, 'uniform', 'grid'

Examples
--------

1. Using LIME with training set samples:
   .. code-block:: bash

      python main.py --dataset dummy.csv --model CLM --interpretation LIME --observation_idx 0 --sampling none

2. Using LIME with uniform sampling:
   .. code-block:: bash

      python main.py --dataset dummy.csv --model CLM --interpretation LIME --observation_idx 0 --sampling uniform

3. Using LIME with grid sampling:
   .. code-block:: bash

      python main.py --dataset dummy.csv --model CLM --interpretation LIME --observation_idx 0 --sampling grid 