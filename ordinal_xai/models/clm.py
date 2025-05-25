"""
Cumulative Link Model (CLM) for ordinal regression.

This module implements a Cumulative Link Model, also known as an Ordered Logit/Probit model,
for ordinal regression. The model uses a link function (logit or probit) to model the
cumulative probabilities of ordinal outcomes.

The model is implemented as a scikit-learn compatible estimator, allowing it to be used
with scikit-learn's pipeline and cross-validation tools.
"""

from typing import Optional, List, Dict, Union, Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from ..utils.data_utils import transform_features
from sklearn.utils.validation import check_X_y, check_is_fitted, validate_data
from statsmodels.miscmodels.ordinal_model import OrderedModel
from .base_model import BaseOrdinalModel

class CLM(BaseEstimator, BaseOrdinalModel):
    """
    Cumulative Link Model for ordinal regression.
    
    This class implements a Cumulative Link Model (CLM) for ordinal regression,
    which models the cumulative probabilities of ordinal outcomes using either
    a logit or probit link function. The model is particularly suitable for
    ordinal data where the response variable has a natural ordering.
    
    Parameters
    ----------
    link : {'logit', 'probit'}, default='logit'
        The link function to use:
        - 'logit': Logistic link function (default)
        - 'probit': Probit link function
        
    Attributes
    ----------
    feature_names_ : list
        Names of features used during training
    n_features_in_ : int
        Number of features seen during training
    ranks_ : ndarray
        Unique ordinal class labels
    _encoder : OneHotEncoder
        Encoder for categorical features
    _scaler : StandardScaler
        Scaler for numerical features
    _model : OrderedModel
        The fitted statsmodels OrderedModel
    _result : OrderedModelResults
        Results from fitting the model
    params_ : ndarray
        Model parameters
    is_fitted_ : bool
        Whether the model has been fitted
        
    Notes
    -----
    - The model handles both categorical and numerical features automatically
    - Categorical features are one-hot encoded
    - Numerical features are standardized
    - The model assumes ordinal classes are consecutive integers
    """
    
    def __init__(self, link: str = "logit"):
        """
        Initialize the Cumulative Link Model.
        
        Parameters
        ----------
        link : {'logit', 'probit'}, default='logit'
            The link function to use for modeling cumulative probabilities
        """
        super().__init__()
        self.link = link
        self._encoder = None
        self._scaler = None
        self.is_fitted_ = False

    def get_params(self, deep: bool = True) -> Dict[str, any]:
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
        return {"link": self.link}

    def set_params(self, **params: any) -> "CLM":
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters
            
        Returns
        -------
        self : CLM
            The estimator instance
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CLM":
        """
        Fit the Cumulative Link Model.
        
        This method fits the model to the training data, handling both categorical
        and numerical features appropriately.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data
        y : pd.Series of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : CLM
            The fitted model
            
        Raises
        ------
        ValueError
            If the link function is invalid
            If the input data contains invalid values
        """
        # Store feature names and metadata
        self.feature_names_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        self.ranks_ = np.unique(y)

        # Transform features
        X_transformed = self.transform(X, fit=True)

        # Validate input
        X, y = check_X_y(X_transformed, y, ensure_2d=True)

        # Validate link function
        link_functions = {"logit": "logit", "probit": "probit"}
        if self.link not in link_functions:
            raise ValueError(
                f"Invalid link function '{self.link}'. "
                f"Choose from {list(link_functions.keys())}."
            )

        # Fit the model
        self._model = OrderedModel(y, X, distr=link_functions[self.link])
        self._result = self._model.fit(method='bfgs', disp=False)
        self.params_ = self._result.params

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
        # Check if model is fitted
        check_is_fitted(self)
        
        # Transform features
        X_transformed = self.transform(X, fit=False)

        # Compute probabilities
        return self._result.predict(X_transformed.values)

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


