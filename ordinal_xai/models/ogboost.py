"""
Ordinal Gradient Boosting Model for ordinal regression.

This module implements a wrapper around the GradientBoostingOrdinal model from the
ogboost package for ordinal regression. The model uses gradient boosting to learn
ordinal relationships while maintaining compatibility with the ordinal_xai framework.

The model is implemented as a scikit-learn compatible estimator, allowing it to be used
with scikit-learn's pipeline and cross-validation tools.
"""

from typing import Optional, List, Dict, Union, Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from ..utils.data_utils import transform_features
from sklearn.utils.validation import check_X_y, check_is_fitted, validate_data
from .base_model import BaseOrdinalModel

try:
    from ogboost import GradientBoostingOrdinal
    from sklearn.tree import DecisionTreeRegressor
except ImportError:
    raise ImportError(
        "ogboost package is required for OGBoost model. "
        "Please install it with: pip install ogboost"
    )


class OGBoost(BaseEstimator, BaseOrdinalModel):
    """
    Ordinal Gradient Boosting Model for ordinal regression.
    
    This class implements a wrapper around the GradientBoostingOrdinal model
    from the ogboost package. The model uses gradient boosting to learn ordinal
    relationships and is particularly effective for complex non-linear patterns
    in ordinal data.
    
    Parameters
    ----------
    base_learner : estimator, default=DecisionTreeRegressor(max_depth=3)
        The base learner used to update the latent function
    n_estimators : int, default=100
        Maximum number of boosting iterations
    learning_rate : float, default=0.1
        Learning rate for the latent function updates
    learning_rate_thresh : float, default=0.001
        Learning rate for the threshold updates
    validation_fraction : float, default=0.1
        Fraction of data to use as a holdout set for early stopping
    n_iter_no_change : int or None, default=None
        Number of iterations with no improvement to wait before stopping early
    tol : float, default=1e-4
        Tolerance for measuring improvement in early stopping
    link_function : {'probit', 'logit', 'loglog', 'cloglog', 'cauchit'}, default='probit'
        Link function used to transform latent scores to probabilities
    subsample : float, default=1.0
        Fraction of samples used to fit each base learner
    verbose : int, default=0
        Verbosity level
    random_state : int, RandomState instance or None, default=None
        Seed or random state for reproducibility
    cv_early_stopping_splits : int or None, default=None
        If an integer > 1, uses K-fold cross-validation for early stopping
        
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
    _model : GradientBoostingOrdinal
        The fitted ogboost GradientBoostingOrdinal model
    is_fitted_ : bool
        Whether the model has been fitted
        
    Notes
    -----
    - The model handles both categorical and numerical features automatically
    - Categorical features are one-hot encoded
    - Numerical features are standardized
    - The model assumes ordinal classes are consecutive integers starting from 0
    """
    
    def __init__(
        self, 
        base_learner = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        learning_rate_thresh: float = 0.001,
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: float = 1e-4,
        link_function: str = 'probit',
        subsample: float = 1.0,
        verbose: int = 0,
        random_state: Optional[int] = None,
        cv_early_stopping_splits: Optional[int] = None
    ):
        """
        Initialize the Ordinal Gradient Boosting Model.
        
        Parameters
        ----------
        base_learner : estimator, default=None
            The base learner used to update the latent function.
            If None, uses DecisionTreeRegressor(max_depth=3)
        n_estimators : int, default=100
            Maximum number of boosting iterations
        learning_rate : float, default=0.1
            Learning rate for the latent function updates
        learning_rate_thresh : float, default=0.001
            Learning rate for the threshold updates
        validation_fraction : float, default=0.1
            Fraction of data to use as a holdout set for early stopping
        n_iter_no_change : int or None, default=None
            Number of iterations with no improvement to wait before stopping early
        tol : float, default=1e-4
            Tolerance for measuring improvement in early stopping
        link_function : str, default='probit'
            Link function used to transform latent scores to probabilities
        subsample : float, default=1.0
            Fraction of samples used to fit each base learner
        verbose : int, default=0
            Verbosity level
        random_state : int, RandomState instance or None, default=None
            Seed or random state for reproducibility
        cv_early_stopping_splits : int or None, default=None
            If an integer > 1, uses K-fold cross-validation for early stopping
        """
        super().__init__()
        self.base_learner = base_learner if base_learner is not None else DecisionTreeRegressor(max_depth=3)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.learning_rate_thresh = learning_rate_thresh
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.link_function = link_function
        self.subsample = subsample
        self.verbose = verbose
        self.random_state = random_state
        self.cv_early_stopping_splits = cv_early_stopping_splits
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
        return {
            "base_learner": self.base_learner,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "learning_rate_thresh": self.learning_rate_thresh,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "tol": self.tol,
            "link_function": self.link_function,
            "subsample": self.subsample,
            "verbose": self.verbose,
            "random_state": self.random_state,
            "cv_early_stopping_splits": self.cv_early_stopping_splits,
        }

    def set_params(self, **params: any) -> "OGBoost":
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters
            
        Returns
        -------
        self : OGBoost
            The estimator instance
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OGBoost":
        """
        Fit the Ordinal Gradient Boosting Model.
        
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
        self : OGBoost
            The fitted model
            
        Raises
        ------
        ValueError
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

        # Initialize the GradientBoostingOrdinal model
        self._model = GradientBoostingOrdinal(
            base_learner=self.base_learner,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            learning_rate_thresh=self.learning_rate_thresh,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            link_function=self.link_function,
            subsample=self.subsample,
            verbose=self.verbose,
            random_state=self.random_state,
            cv_early_stopping_splits=self.cv_early_stopping_splits
        )

        # Fit the model
        self._model.fit(X, y)

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
        # Check if model is fitted
        check_is_fitted(self)
        
        # Transform features
        X_transformed = self.transform(X, fit=False)

        # Make predictions
        return self._model.predict(X_transformed.values)

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
        return self._model.predict_proba(X_transformed.values)

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

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the latent function values for input samples.
        
        This method returns the scalar value of the latent function for each
        observation, which can be used as a high-resolution alternative to
        class labels for comparing and ranking observations.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Samples to compute decision function for
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Latent function values
            
        Raises
        ------
        NotFittedError
            If the model has not been fitted
        """
        # Check if model is fitted
        check_is_fitted(self)
        
        # Transform features
        X_transformed = self.transform(X, fit=False)

        # Compute decision function
        return self._model.decision_function(X_transformed.values)

    def feature_importances_(self) -> np.ndarray:
        """
        Get feature importances from the fitted model.
        
        Note: This method may not be available for all base learners.
        
        Returns
        -------
        ndarray of shape (n_features,)
            Feature importances if available
            
        Raises
        ------
        NotFittedError
            If the model has not been fitted
        AttributeError
            If the base learner doesn't support feature importances
        """
        check_is_fitted(self)
        if hasattr(self._model, 'feature_importances_'):
            return self._model.feature_importances_
        else:
            raise AttributeError(
                "Feature importances are not available for the current base learner. "
                "Use a base learner that supports feature importances (e.g., DecisionTreeRegressor)."
            )

    def get_booster_params(self) -> Dict[str, any]:
        """
        Get parameters of the underlying boosting model.
        
        Returns
        -------
        dict
            Parameters of the underlying GradientBoostingOrdinal model
            
        Raises
        ------
        NotFittedError
            If the model has not been fitted
        """
        check_is_fitted(self)
        return self._model.get_params()
