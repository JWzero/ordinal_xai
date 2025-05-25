"""
Ordinal Binary Decomposition (OBD) Model.

This module implements the Ordinal Binary Decomposition (OBD) model, which is a framework
for ordinal classification that decomposes the ordinal problem into a series of binary
classification tasks. The model supports two decomposition strategies:

1. "one-vs-following" (default):
   - Classifier 1: Class 0 vs. Classes 1,2,...,K-1
   - Classifier 2: Class 1 vs. Classes 2,3,...,K-1
   - ...and so on

2. "one-vs-next":
   - Classifier 1: Class 0 vs. Class 1
   - Classifier 2: Class 1 vs. Class 2
   - ...and so on

The model can use various base classifiers (logistic regression, SVM, random forest, XGBoost)
to solve each binary classification task. The final prediction is obtained by combining
the probabilities from all binary classifiers in a way that respects the ordinal nature
of the problem.

Example:
    >>> from models.obd import OBD
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample data
    >>> X = pd.DataFrame(np.random.randn(100, 5))
    >>> y = pd.Series(np.random.randint(0, 3, 100))
    >>> 
    >>> # Initialize and train model
    >>> model = OBD(base_classifier='svm', decomposition_type='one-vs-next')
    >>> model.fit(X, y)
    >>> 
    >>> # Make predictions
    >>> predictions = model.predict(X)
    >>> probabilities = model.predict_proba(X)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from utils.data_utils import transform_features
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
import sys
import os

# Add the root directory to the path to make imports work when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_model import BaseOrdinalModel

class OBD(BaseEstimator, BaseOrdinalModel):
    """
    Ordinal Binary Decomposition (OBD) Model.
    
    This model implements ordinal classification by decomposing the problem into
    a series of binary classification tasks. It supports two decomposition strategies
    and various base classifiers.
    
    Parameters
    ----------
    base_classifier : str, default='logistic'
        The base classifier to use for each binary classification task. Options are:
        - 'logistic': LogisticRegression with default parameters
        - 'svm': SVC with probability=True and balanced class weights
        - 'rf': RandomForestClassifier with conservative defaults
        - 'xgb': XGBClassifier with conservative defaults
    decomposition_type : str, default='one-vs-following'
        The type of binary decomposition to use. Options are:
        - 'one-vs-following': Each class is compared against all following classes
        - 'one-vs-next': Each class is compared only against the next class
    **kwargs : dict
        Additional parameters to pass to the base classifier. These will override
        the default parameters set for each classifier type.
        
    Attributes
    ----------
    feature_names_ : list
        Names of the features used during training
    n_features_in_ : int
        Number of features seen during training
    ranks_ : ndarray
        Unique ordinal class labels in ascending order
    _models : list
        List of trained binary classifiers
    _encoder : object
        Feature encoder used for categorical variables
    _scaler : object
        Feature scaler used for numerical variables
    is_fitted_ : bool
        Flag indicating whether the model has been fitted
        
    Notes
    -----
    - The model automatically handles categorical and numerical features through
      the transform_features utility
    - For each binary classification task, if only one class is present in the
      training data, a DummyClassifier is used instead of the specified base classifier
    - The predict_proba method returns class probabilities that sum to 1 for each sample
    - The model is compatible with scikit-learn's cross-validation and grid search
      
    Examples
    --------
    >>> from models.obd import OBD
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample data
    >>> X = pd.DataFrame(np.random.randn(100, 5))
    >>> y = pd.Series(np.random.randint(0, 3, 100))
    >>> 
    >>> # Initialize model with SVM base classifier
    >>> model = OBD(base_classifier='svm', decomposition_type='one-vs-next')
    >>> 
    >>> # Train the model
    >>> model.fit(X, y)
    >>> 
    >>> # Make predictions
    >>> predictions = model.predict(X)
    >>> probabilities = model.predict_proba(X)
    """
    
    def __init__(self, base_classifier='logistic', decomposition_type='one-vs-following', **kwargs):
        """
        Initialize the OBD model.
        
        Parameters
        ----------
        base_classifier : str, default='logistic'
            The base classifier to use. Options are:
            - 'logistic': LogisticRegression
            - 'svm': SVC with probability=True
            - 'rf': RandomForestClassifier
            - 'xgb': XGBClassifier
        decomposition_type : str, default='one-vs-following'
            The type of binary decomposition to use. Options are:
            - 'one-vs-following': Each class is compared against all following classes
            - 'one-vs-next': Each class is compared only against the next class
        **kwargs : dict
            Additional parameters to pass to the base classifier.
        """
        super().__init__()  # Initialize base class
        self.base_classifier = base_classifier
        self.decomposition_type = decomposition_type
        self.kwargs = kwargs
        self._models = None
        self._encoder = None
        self._scaler = None

    def _get_base_classifier(self):
        """
        Get the appropriate base classifier instance with sensible defaults.
        
        Returns
        -------
        estimator : object
            An instance of the specified base classifier with appropriate default parameters.
            
        Raises
        ------
        ValueError
            If an unknown base classifier is specified.
        """
        if self.base_classifier == 'logistic':
            return LogisticRegression(**self.kwargs)
        elif self.base_classifier == 'svm':
            # Set sensible defaults for SVM if not provided
            svm_params = {
                'C': 1.0,                # Regularization parameter
                'kernel': 'rbf',         # Radial basis function kernel
                'gamma': 'scale',        # Automatic gamma scaling
                'probability': True,      # Enable probability estimates
                'class_weight': 'balanced',  # Handle class imbalance
                'random_state': 42,
                'cache_size': 1000,      # Increase cache size for better performance
                'tol': 1e-3              # Tolerance for stopping criterion
            }
            svm_params.update(self.kwargs)
            return SVC(**svm_params)
        elif self.base_classifier == 'rf':
            # Set conservative defaults if not provided
            rf_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_leaf': 5,
                'random_state': 42
            }
            rf_params.update(self.kwargs)
            return RandomForestClassifier(**rf_params)
        elif self.base_classifier == 'xgb':
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
                'random_state': 42
            }
            xgb_params.update(self.kwargs)
            return XGBClassifier(**xgb_params)
        else:
            raise ValueError(f"Unknown base classifier: {self.base_classifier}. Use 'logistic', 'svm', 'rf', or 'xgb'.")

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            "base_classifier": self.base_classifier,
            "decomposition_type": self.decomposition_type
        }
        params.update(self.kwargs)
        return params
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Estimator instance.
        """
        if 'base_classifier' in params:
            self.base_classifier = params.pop('base_classifier')
        if 'decomposition_type' in params:
            self.decomposition_type = params.pop('decomposition_type')
        self.kwargs.update(params)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OBD":
        """
        Fit the ordinal binary decomposition model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data of shape (n_samples, n_features)
        y : pd.Series
            Target values of shape (n_samples,)
            
        Returns
        -------
        self : object
            Returns self.
            
        Raises
        ------
        TypeError
            If X is not a DataFrame or y is not a Series
        ValueError
            If X and y have different number of samples
            If X or y contains missing values
            If decomposition_type is not 'one-vs-following' or 'one-vs-next'
        """
        # Validate input types
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
            
        # Validate input shapes
        if len(X) != len(y):
            raise ValueError(f"X and y have different number of samples: X has {len(X)} samples, y has {len(y)} samples")
            
        # Validate missing values
        if X.isnull().any().any():
            missing_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(f"X contains missing values in columns: {missing_cols}")
        if y.isnull().any():
            raise ValueError("y contains missing values")
            
        # Validate decomposition type
        valid_types = ['one-vs-following', 'one-vs-next']
        if self.decomposition_type not in valid_types:
            raise ValueError(f"decomposition_type must be one of {valid_types}, got {self.decomposition_type}")
        
        # Store feature names and ranks
        self.feature_names_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        self.ranks_ = np.unique(y)
        
        # Transform input data
        X_transformed = self.transform(X, fit=True)
        
        # Initialize models for each binary classification task
        self._models = []
        for i in range(len(self.ranks_) - 1):
            # Create binary labels based on decomposition type
            if self.decomposition_type == 'one-vs-following':
                y_binary = (y > self.ranks_[i]).astype(int)
            elif self.decomposition_type == 'one-vs-next':
                # Only compare current class with next class
                mask = (y == self.ranks_[i]) | (y == self.ranks_[i + 1])
                y_binary = (y[mask] > self.ranks_[i]).astype(int)
                X_binary = X_transformed[mask]
            else:
                raise ValueError(f"Unknown decomposition type: {self.decomposition_type}. Use 'one-vs-following' or 'one-vs-next'.")
            
            # Check if we have samples from both classes
            unique_classes = np.unique(y_binary)
            if len(unique_classes) < 2:
                # If only one class, create a dummy model that always predicts that class
                class_value = unique_classes[0]
                model = DummyClassifier(strategy='constant', constant=class_value)
                print(f"Warning: Using DummyClassifier for threshold {i} as only class {class_value} is present")
            else:
                # Get and fit base classifier
                model = self._get_base_classifier()
            
            # Fit the model with appropriate data
            if self.decomposition_type == 'one-vs-following':
                model.fit(X_transformed, y_binary)
            else:  # one-vs-next
                model.fit(X_binary, y_binary)
            
            self._models.append(model)
        
        # Set fitted flag
        self.is_fitted_ = True
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ordinal class labels.
        
        Parameters
        ----------
        X : pd.DataFrame
            Samples of shape (n_samples, n_features)
            
        Returns
        -------
        y_pred : ndarray
            Predicted class labels of shape (n_samples,)
        """
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : pd.DataFrame
            Samples of shape (n_samples, n_features)
            
        Returns
        -------
        proba : ndarray
            Class probabilities of shape (n_samples, n_classes)
            
        Raises
        ------
        TypeError
            If X is not a DataFrame
        ValueError
            If X contains missing values
            If X has different number of features than training data
            
        Notes
        -----
        The probabilities are computed differently based on the decomposition type:
        
        For 'one-vs-following':
        - P(class 0) = 1 - P(class > 0)
        - P(class i) = P(class > i-1) - P(class > i)
        - P(class K-1) = P(class > K-2)
        
        For 'one-vs-next':
        - P(class 0) = 1 - P(class > 0)
        - P(class i) = P(class > i-1) * (1 - P(class > i))
        - P(class K-1) = P(class > K-2)
        """
        check_is_fitted(self)
        
        # Validate input type
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
            
        # Validate missing values
        if X.isnull().any().any():
            missing_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(f"X contains missing values in columns: {missing_cols}")
            
        # Validate number of features
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but model was trained on {self.n_features_in_} features")
        
        X_transformed = self.transform(X, fit=False)
        
        # Get probabilities for each binary classifier
        binary_probs = []
        for model in self._models:
            if isinstance(model, DummyClassifier):
                # DummyClassifier returns single column, convert to two columns
                prob = model.predict_proba(X_transformed)
                if prob.shape[1] == 1:
                    # If only one class, create two columns
                    prob = np.column_stack([1 - prob, prob])
            else:
                prob = model.predict_proba(X_transformed)
            binary_probs.append(prob[:, 1])  # Get probability of positive class
        
        binary_probs = np.array(binary_probs)
        
        # Convert to ordinal probabilities based on decomposition type
        n_samples = len(X)
        n_classes = len(self.ranks_)
        probs = np.zeros((n_samples, n_classes))
        
        if self.decomposition_type == 'one-vs-following':
            # First class probability
            probs[:, 0] = 1 - binary_probs[0]
            
            # Middle class probabilities
            for i in range(1, n_classes - 1):
                probs[:, i] = binary_probs[i-1] - binary_probs[i]
            
            # Last class probability
            probs[:, -1] = binary_probs[-1]
        else:  # one-vs-next
            # First class probability
            probs[:, 0] = 1 - binary_probs[0]
            
            # Middle class probabilities
            for i in range(1, n_classes - 1):
                probs[:, i] = binary_probs[i-1] * (1 - binary_probs[i])
            
            # Last class probability
            probs[:, -1] = binary_probs[-1]
        
        # Normalize probabilities to sum to 1
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs

    def transform(self, X: pd.DataFrame, fit=False, no_scaling=False) -> pd.DataFrame:
        """
        Transform input data into the format expected by the model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data of shape (n_samples, n_features)
        fit : bool, default=False
            Whether this is being called during fit or predict
        no_scaling : bool, default=False
            Whether to skip feature scaling
            
        Returns
        -------
        X_transformed : pd.DataFrame
            Transformed data ready for model input
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