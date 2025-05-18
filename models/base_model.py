from abc import ABC, abstractmethod
import numpy as np

class BaseOrdinalModel(ABC):
    """Abstract base class for ordinal regression models."""

    def __init__(self):
        """Initialize the base model."""
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        self.is_fitted_ = True
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ordinal labels."""
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet. Call fit() before predict().")
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability distributions over ordinal classes."""
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet. Call fit() before predict_proba().")
        pass

    @abstractmethod
    def get_params(self, deep=True):
        """Return model parameters for scikit-learn compatibility."""
        pass

