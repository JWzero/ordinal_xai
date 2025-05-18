Interpretation Methods
====================

This section documents the interpretation methods available in the package.

LIME (Local Interpretable Model-agnostic Explanations)
---------------------------------------------------

.. automodule:: interpretation.LIME
   :members:
   :undoc-members:
   :show-inheritance:

The LIME class provides local explanations for ordinal regression models by approximating the model's behavior around a specific observation. It supports different sampling methods:

- None: Use training set samples
- 'uniform': Sample uniformly from feature ranges
- 'grid': Create equidistant grid over feature ranges

LOCO (Leave-One-Covariate-Out)
-----------------------------

.. automodule:: interpretation.LOCO
   :members:
   :undoc-members:
   :show-inheritance:

The LOCO class implements the Leave-One-Covariate-Out method for feature importance estimation in ordinal regression models.

Base Interpretation
-----------------

.. automodule:: interpretation.base_interpretation
   :members:
   :undoc-members:
   :show-inheritance:

The BaseInterpretation class serves as the base class for all interpretation methods in the package. 