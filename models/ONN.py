import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from utils.data_utils import transform_features
from sklearn.utils.validation import check_X_y, check_is_fitted
from skorch import NeuralNet
from skorch.callbacks import EarlyStopping
from dlordinal.losses import TriangularLoss
from models.base_model import BaseOrdinalModel

class OrdinalNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))  # One output per class
        self.network = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.network(X)

class ONN(BaseEstimator, BaseOrdinalModel):
    def __init__(self, hidden_layers=[64,64], dropout=0.2, max_epochs=10000, batch_size=32, lr=0.001,
                 patience=10, min_delta=0.0001, verbose=2):
        super().__init__()  # Initialize base class
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self._model = None
        self._encoder = None
        self._scaler = None
        self.is_fitted_ = False

    def get_params(self, deep=True):
        return {
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "patience": self.patience,
            "min_delta": self.min_delta
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ONN":
        """Fit the neural network ordinal regression model."""
        
        # Store feature names and ranks
        self.feature_names_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        self.ranks_ = np.unique(y)
        
        # Transform input data
        X_transformed = self.transform(X, fit=True)
        
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X_transformed.values)
        y_tensor = torch.LongTensor(y.values)
        
        # Initialize model
        input_dim = X_transformed.shape[1]
        num_classes = len(self.ranks_)
        net = OrdinalNet(input_dim, num_classes, self.hidden_layers, self.dropout)
        
        # Initialize early stopping callback
        early_stopping = EarlyStopping(
            monitor='valid_loss',
            patience=self.patience,
            threshold=self.min_delta,
            threshold_mode='rel',
            lower_is_better=True
        )
        
        # Initialize skorch neural network
        self._model = NeuralNet(
            module=net,
            criterion=TriangularLoss(base_loss=nn.CrossEntropyLoss(), num_classes=num_classes),
            optimizer=torch.optim.Adam,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            callbacks=[early_stopping],
            verbose=self.verbose  # Suppress training progress output
        )
        
        # Fit the model
        self._model.fit(X_tensor, y_tensor)
        
        # Set fitted flag
        self.is_fitted_ = True
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the ordinal class labels."""
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self)
        
        X_transformed = self.transform(X, fit=False)
        X_tensor = torch.FloatTensor(X_transformed.values)
        
        # Get raw predictions
        with torch.no_grad():
            raw_preds = self._model.predict(X_tensor)
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(torch.tensor(raw_preds), dim=1).numpy()
        
        return probs

    def transform(self, X: pd.DataFrame, fit=False, no_scaling=False) -> pd.DataFrame:
        """Transform input data into the format expected by the model."""
        X_transformed, encoder, scaler = transform_features(
            X,
            fit=fit,
            encoder=self._encoder,
            scaler=self._scaler,
            no_scaling=no_scaling
        )
        if fit:
            self._encoder = encoder
            self._scaler = scaler
        return X_transformed 