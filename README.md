# Ordinal XAI

Explainable AI for Ordinal Regression Models.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ordinal_xai.git
cd ordinal_xai

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e ".[dev]"
```

## Usage

```python
from ordinal_xai import LIME
import pandas as pd
import numpy as np
from ordinal_xai.models import CLM

# Load your data
X = pd.DataFrame(...)  # Your features
y = np.array(...)      # Your ordinal target

# Train your model
model = CLM()
model.fit(X, y)

# Create LIME explainer
explainer = LIME(model, X, y)

# Generate explanations
explanation = explainer.explain(observation_idx=0, plot=True)
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=ordinal_xai

# Run specific test file
pytest tests/test_lime.py

# Run tests matching a pattern
pytest -k "test_lime_initialization"
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

```bash
# Format code
black ordinal_xai tests

# Sort imports
isort ordinal_xai tests

# Run linter
flake8 ordinal_xai tests

# Run type checker
mypy ordinal_xai tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
