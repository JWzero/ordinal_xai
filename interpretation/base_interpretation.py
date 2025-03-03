from abc import ABC, abstractmethod
import numpy as np

class BaseInterpretation(ABC):
    """Base class for all interpretation methods."""
    
    def __init__(self, model, X: np.ndarray, y: np.ndarray = None):
        """
        Initialize the interpretation method with the trained model and dataset.
        
        Parameters:
        - model: The trained ordinal regression model.
        - X: NumPy array containing the dataset used for interpretation.
        - y: (Optional) NumPy array containing target labels.
        """
        self.model = model
        self.X = X
        self.y = y

    @abstractmethod
    def explain(self, observation_idx=None, feature_subset=None):
        """
        Generate explanations.
        
        Parameters:
        - observation_idx: (Optional) Index of the specific instance to explain.
        - feature_subset: (Optional) List of feature indices to include in the explanation.
        """
        pass
