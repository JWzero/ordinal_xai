from abc import ABC, abstractmethod
import numpy as np

class BaseOrdinalModel(ABC):
    """Abstract base class for ordinal regression models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ordinal labels."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability distributions over ordinal classes."""
        pass

    @abstractmethod
    def get_params(self, deep=True):
        """Return model parameters for scikit-learn compatibility."""
        pass

