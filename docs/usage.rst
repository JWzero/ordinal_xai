Usage
=====

This page provides detailed usage examples for the main models and interpretation methods in Ordinal XAI.

Quick Start
-----------

.. code-block:: python

    from ordinal_xai.models import CLM, ONN, OBD
    from ordinal_xai.interpretation import LIME, LOCO, ICE, ICEProb, PDP, PDPProb, PFI
    import pandas as pd
    import numpy as np

    # Create sample data
    X = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series(np.random.randint(0, 3, 100))

    # Initialize and train model
    model = CLM(link='logit')
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Generate explanations
    pdp = PDP(model, X, y)
    pdp_effects = pdp.explain(features=['feature1', 'feature2'], plot=True)

    pdp_prob = PDPProb(model, X, y)
    pdp_prob_effects = pdp_prob.explain(features=['feature1', 'feature2'], plot=True)

    ice = ICE(model, X, y)
    ice_effects = ice.explain(features=['feature1', 'feature2'], plot=True)

    ice_prob = ICEProb(model, X, y)
    ice_prob_effects = ice_prob.explain(features=['feature1', 'feature2'], plot=True)

    pfi = PFI(model, X, y)
    pfi_importance = pfi.explain(plot=True)

    loco = LOCO(model, X, y)
    loco_importance = loco.explain(plot=True)

    lime = LIME(model, X, y)
    lime_explanation = lime.explain(observation_idx=0, plot=True)


Models
------

The following models are currently implemented:

- Cumulative Link Model (CLM):
    The Cumulative Link Model (CLM) is a popular model for ordinal regression. It models the cumulative probability of the response variable being less than or equal to a certain threshold as a function of the predictors. It uses the proportional odds assumption, which means that the odds of being in any category are the same for all categories.
- Ordinal Neural Network (ONN):
    The Ordinal Neural Network (ONN) is a fully connected neural network that uses a softmax activation function in the output layer to ensure that the output is a valid probability distribution over the categories.
- Ordinal Binary Decomposition (OBD):
    The Ordinal Binary Decomposition (OBD) is a model for ordinal regression that decomposes the ordinal response into a set of binary responses. It uses a series of binary classifiers to model the relationship between the predictors and the binary responses.


Interpretation Methods
----------------------

Ordinal XAI provides a variety of interpretation methods to help understand the predictions of ordinal regression models.

- Feature Effect Plots:
    - Partial Dependence Plot (PDP):
        The Partial Dependence Plot (PDP) shows the marginal effect of a feature on the predicted probability of the response variable. It is a global method that can be used to understand the relationship between a feature and the response variable.
    - Individual Conditional Expectation (ICE):
        The Individual Conditional Expectation (ICE) plot shows the effect of a feature on the predicted probability of the response variable for a specific observation. It is a local method that can be used to understand the relationship between a feature and the response variable for a specific observation.
    - Partial Dependence Plot with Probabilities (PDPProb):
        The Partial Dependence Plot with Probabilities (PDPProb) shows the marginal effect of a feature on the predicted probability of the response variable. It is a global method that can be used to understand the relationship between a feature and the response variable.
    - Individual Conditional Expectation with Probabilities (ICEProb):
        The Individual Conditional Expectation with Probabilities (ICEProb) plot shows the effect of a feature on the predicted probability of the response variable for a specific observation. It is a local method that can be used to understand the relationship between a feature and the response variable for a specific observation.
- Local Interpretable Model-agnostic Explanations (LIME):
    The Local Interpretable Model-agnostic Explanations (LIME) method is a local method that can be used to explain the predictions of a model for a specific observation. It is a model-agnostic method that can be used to explain the predictions of any model.
- Feature Importance Methods:
    - Permutation Feature Importance (PFI):
        The Permutation Feature Importance (PFI) method is a global method that can be used to understand the importance of each feature in the model. It is a model-agnostic method that can be used to understand the importance of each feature in any model.
    - Leave-One-Covariate-Out (LOCO):
        The Leave-One-Covariate-Out (LOCO) method is a global method that can be used to understand the importance of each feature in the model. It is a model-agnostic method that can be used to understand the importance of each feature in any model. A local version of this method is also implemented.

Command-Line Usage
------------------

You can also use Ordinal XAI directly from the command line to run models and generate explanations. This is useful for batch processing, automation, or quick experimentation.

Basic usage:

.. code-block:: bash

    python -m ordinal_xai --dataset wine.csv --model CLM --interpretation PDP

Or, if installed as a CLI entry point (if available):

.. code-block:: bash

    ordinal_xai --dataset wine.csv --model CLM --interpretation PDP

You can specify various arguments to control the dataset, model, interpretation method, and their parameters.

**Command-Line Arguments:**

``--dataset``
    Dataset filename in 'data/' folder (default: 'dummy.csv')

``--model``
    Model to use: CLM, ONN, or OBD (default: 'CLM')

``--interpretation``
    Interpretation method: PDP, ICE, LIME, LOCO, PFI, etc. (default: 'PDP')

``--model_params``
    JSON string of model parameters (e.g., '{"link": "probit"}')

``--interpretation_params``
    JSON string of interpretation parameters (e.g., '{"sampling": "uniform", "model_type": "decision_tree"}')

``--observation_idx``
    Index of the observation to interpret (for local explanations)

``--features``
    Comma-separated list of feature indices or names to include in the explanation (optional)

**Examples:**

Run CLM with default settings on the wine dataset and generate a PDP:

.. code-block:: bash

    python -m ordinal_xai --dataset wine.csv --model CLM --interpretation PDP

Run OBD with SVM base classifier and LIME interpretation:

.. code-block:: bash

    python -m ordinal_xai --dataset wine.csv --model OBD --model_params '{"base_classifier": "svm", "decomposition_type": "one-vs-next"}' --interpretation LIME --interpretation_params '{"sampling": "uniform", "model_type": "decision_tree"}' --observation_idx 0

For more advanced usage, see the API reference and the documentation for each class and method. 