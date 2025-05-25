"""
Ordinal Neural Network (ONN) for ordinal regression.

This module implements an Ordinal Neural Network model for ordinal regression,
using a fully connected neural network based on the skorch library with a triangular loss function from the dlordinal library (Berchez et al., 2025). The model
is designed to handle ordinal data by incorporating the ordinal nature of the
target variable into the loss function.

The model is implemented as a scikit-learn compatible estimator, allowing it to be
used with scikit-learn's pipeline and cross-validation tools.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from utils.data_utils import transform_features
from sklearn.utils.validation import check_X_y, check_is_fitted
from skorch import NeuralNet
from skorch.callbacks import EarlyStopping
from dlordinal.losses import CDWCELoss
from dlordinal.output_layers import StickBreakingLayer, CLM
from models.base_model import BaseOrdinalModel
from typing import List, Optional, Union, Dict, Any

class FCNet(nn.Module):
    """
    Fully Connected Neural Network for ordinal regression.
    
    This class implements a feed-forward neural network with configurable
    hidden layers, dropout, and ReLU activation functions. The network outputs
    logits that can be used with various ordinal loss functions.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    num_classes : int
        Number of ordinal classes
    hidden_layers : List[int]
        List of hidden layer sizes
    dropout : float
        Dropout rate for regularization
        
    Attributes
    ----------
    network : nn.Sequential
        The neural network architecture
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_layers: List[int], dropout: float, output_layer: nn.Module):
        """
        Initialize the neural network architecture.
        
        Parameters
        ----------
        input_dim : int
            Number of input features
        num_classes : int
            Number of ordinal classes
        hidden_layers : List[int]
            List of hidden layer sizes
        dropout : float
            Dropout rate for regularization
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(output_layer)
        self.network = nn.Sequential(*layers)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes) containing logits
        """
        return self.network(X)

class ONN(BaseEstimator, BaseOrdinalModel):
    """
    Ordinal Neural Network for ordinal regression.
    
    This class implements an ordinal regression model using a neural network
    with a triangular loss function. The model uses a fully connected network
    with configurable architecture and training parameters.
    
    Parameters
    ----------
    hidden_layers : List[int], default=[64, 64]
        List of hidden layer sizes
    dropout : float, default=0.2
        Dropout rate for regularization
    max_epochs : int, default=10000
        Maximum number of training epochs
    batch_size : int, default=32
        Batch size for training
    lr : float, default=0.001
        Learning rate for optimization
    patience : int, default=10
        Number of epochs to wait for improvement before early stopping
    min_delta : float, default=0.0001
        Minimum change in validation loss to be considered as improvement
    verbose : int, default=2
        Verbosity level for training progress
        
    Attributes
    ----------
    feature_names_ : list
        Names of features used during training
    n_features_in_ : int
        Number of features seen during training
    ranks_ : ndarray
        Unique ordinal class labels
    _model : NeuralNet
        The fitted skorch neural network
    _encoder : OneHotEncoder
        Encoder for categorical features
    _scaler : StandardScaler
        Scaler for numerical features
    is_fitted_ : bool
        Whether the model has been fitted
        
    Notes
    -----
    - The model handles both categorical and numerical features automatically
    - Categorical features are one-hot encoded
    - Numerical features are standardized
    - Uses early stopping to prevent overfitting
    - Automatically uses GPU if available
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [64, 64],
        output_layer: nn.Module = StickBreakingLayer,
        loss_function: nn.Module = CDWCELoss,
        dropout: float = 0.2,
        max_epochs: int = 10000,
        batch_size: int = 32,
        lr: float = 0.001,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        patience: int = 10,
        min_delta: float = 0.0001,
        verbose: int = 2
    ):
        """
        Initialize the Ordinal Neural Network.
        
        Parameters
        ----------
        hidden_layers : List[int], default=[64, 64]
            List of hidden layer sizes
        dropout : float, default=0.2
            Dropout rate for regularization
        max_epochs : int, default=10000
            Maximum number of training epochs
        batch_size : int, default=32
            Batch size for training
        lr : float, default=0.001
            Learning rate for optimization
        optimizer: torch.optim.Optimizer, default=torch.optim.Adam
            Optimizer for optimization
        patience : int, default=10
            Number of epochs to wait for improvement before early stopping
        min_delta : float, default=0.0001
            Minimum change in validation loss to be considered as improvement
        verbose : int, default=2
            Verbosity level for training progress
        """
        super().__init__()
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.loss_function = loss_function
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self._model = None
        self._encoder = None
        self._scaler = None
        self.is_fitted_ = False

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        dict
            Parameter names mapped to their values
        """
        return {
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "verbose": self.verbose
        }

    def set_params(self, **params: Any) -> "ONN":
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters
            
        Returns
        -------
        self : ONN
            The estimator instance
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ONN":
        """
        Fit the Ordinal Neural Network model.
        
        This method fits the model to the training data, handling both categorical
        and numerical features appropriately. It uses early stopping to prevent
        overfitting and automatically selects the best device (CPU/GPU) for training.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data
        y : pd.Series of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : ONN
            The fitted model
            
        Raises
        ------
        ValueError
            If the input data contains invalid values
        RuntimeError
            If there are issues with the neural network training
        """
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
        self.output_layer = self.output_layer(num_classes=num_classes,input_shape=self.hidden_layers[-1])
        net = FCNet(input_dim, num_classes, self.hidden_layers, self.dropout, self.output_layer)
        
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
            criterion=self.loss_function(num_classes=num_classes),
            optimizer=self.optimizer,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            callbacks=[early_stopping],
            verbose=self.verbose
        )
        
        # Fit the model
        self._model.fit(X_tensor, y_tensor)
        
        # Set fitted flag
        self.is_fitted_ = True
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ordinal class labels.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted ordinal class labels
            
        Raises
        ------
        NotFittedError
            If the model has not been fitted
        """
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Samples to predict probabilities for
            
        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Predicted class probabilities
            
        Raises
        ------
        NotFittedError
            If the model has not been fitted
        """
        check_is_fitted(self)
        
        X_transformed = self.transform(X, fit=False)
        X_tensor = torch.FloatTensor(X_transformed.values)
        
        # Get logits and convert to probabilities
        with torch.no_grad():
            logits = self._model.predict(X_tensor)
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        
        return np.array(probs)

    def transform(self, X: pd.DataFrame, fit: bool = False, no_scaling: bool = False) -> pd.DataFrame:
        """
        Transform input data into the format expected by the model.
        
        This method handles both categorical and numerical features:
        - Categorical features are one-hot encoded
        - Numerical features are standardized (unless no_scaling=True)
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data to transform
        fit : bool, default=False
            Whether to fit new encoder/scaler or use existing ones
        no_scaling : bool, default=False
            Whether to skip scaling of numerical features
            
        Returns
        -------
        pd.DataFrame
            Transformed data
            
        Raises
        ------
        ValueError
            If the input data has different features than training data
        """
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