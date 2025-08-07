Installation
============

This page explains how to install Ordinal XAI and its dependencies.

Prerequisites
-------------
- Python 3.8 or newer is recommended
- pip (Python package manager)

Install from PyPI
-----------------
The easiest way to install Ordinal XAI is from the Python Package Index (PyPI):

.. code-block:: bash

    pip install ordinal-xai

Install from Source
-------------------
If you want the latest development version or want to contribute, clone the repository and install with pip:

.. code-block:: bash

    git clone https://github.com/JWZero/ordinal-xai.git
    cd ordinal-xai
    pip install -e .

Optional: Build Documentation Locally
-------------------------------------
To build the documentation locally, install the documentation dependencies:

.. code-block:: bash

    pip install -r docs/requirements.txt
    sphinx-build -b html docs/ docs/_build/html

Optional Dependencies
---------------------
Some features (such as neural network models) require additional packages like `torch` and `skorch`. These are included in the default requirements, but if you encounter issues, install them manually:

.. code-block:: bash

    pip install torch skorch

Troubleshooting
---------------
- If you encounter issues with missing packages, try upgrading pip:

  .. code-block:: bash

      pip install --upgrade pip

- For GPU support with neural networks, ensure you have the correct version of PyTorch installed for your hardware.
- If you have issues with plotting, ensure `matplotlib` is installed and up to date.

For more help, see the README or open an issue on the project's GitHub page. 