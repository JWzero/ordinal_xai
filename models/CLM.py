import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import expit  # Sigmoid function
from .base_model import BaseOrdinalModel

class CLM(BaseOrdinalModel):
    """Cumulative Logit Model for Ordinal Regression."""
    
    def __init__(self):
        self.models = []  # One logistic regression per threshold
        self.thresholds = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the cumulative logit model."""
        X = self.scaler.fit_transform(X)
        unique_classes = np.sort(np.unique(y))
        self.thresholds = unique_classes[:-1]  # Thresholds for cumulative logits

        for threshold in self.thresholds:
            binary_y = (y > threshold).astype(int)  # Convert to binary task
            model = LogisticRegression()
            model.fit(X, binary_y)
            self.models.append(model)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability distribution over ordinal classes."""
        X = self.scaler.transform(X)
        cumulative_probs = np.array([model.predict_proba(X)[:, 1] for model in self.models]).T
        cumulative_probs = np.hstack([np.zeros((X.shape[0], 1)), cumulative_probs, np.ones((X.shape[0], 1))])
        
        # Convert cumulative probabilities to class probabilities
        class_probs = np.diff(cumulative_probs, axis=1)
        return class_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ordinal class labels."""
        class_probs = self.predict_proba(X)
        return np.argmax(class_probs, axis=1)
