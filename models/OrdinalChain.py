import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

class OrdinalChain(BaseEstimator, BaseOrdinalModel):
    """
    One-vs-Following Binary Decomposition Ordinal Regression Model.
    
    For K ordinal classes (0, 1, 2, ..., K-1), it trains K-1 binary classifiers:
    - Classifier 1: Class 0 vs. Classes 1,2,...,K-1
    - Classifier 2: Class 1 vs. Classes 2,3,...,K-1
    - ...and so on
    
    This approach captures the ordinal relationship between classes by comparing
    each class against all following classes, which better represents the ordinal
    nature of the problem compared to standard one-vs-all approaches.
    
    Parameters
    ----------
    base_classifier : str, default='logistic'
        The base classifier to use. Options are:
        - 'logistic': LogisticRegression
        - 'svm': SVC with probability=True
        - 'rf': RandomForestClassifier
        - 'xgb': XGBClassifier
    **kwargs : dict
        Additional parameters to pass to the base classifier.
    """
    
    def __init__(self, base_classifier='logistic', **kwargs):
        super().__init__()  # Initialize base class
        self.base_classifier = base_classifier
        self.kwargs = kwargs
        self._models = None
        self._encoder = None
        self._scaler = None

    def _get_base_classifier(self):
        """Get the appropriate base classifier instance with sensible defaults."""
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
        """Return model parameters for scikit-learn compatibility."""
        params = {"base_classifier": self.base_classifier}
        params.update(self.kwargs)
        return params
    
    def set_params(self, **params):
        """Set parameters for scikit-learn compatibility."""
        if 'base_classifier' in params:
            self.base_classifier = params.pop('base_classifier')
        self.kwargs.update(params)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OrdinalChain":
        """Fit the ordinal chain model."""
        
        # Store feature names and ranks
        self.feature_names_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        self.ranks_ = np.unique(y)
        
        # Transform input data
        X_transformed = self.transform(X, fit=True)
        
        # Initialize models for each binary classification task
        self._models = []
        for i in range(len(self.ranks_) - 1):
            # Create binary labels for this threshold (current vs all following)
            y_binary = (y > self.ranks_[i]).astype(int)
            
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
            
            model.fit(X_transformed, y_binary)
            self._models.append(model)
        
        # Set fitted flag
        self.is_fitted_ = True
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ordinal class labels."""
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self)
        
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
        
        # Convert to ordinal probabilities
        n_samples = len(X)
        n_classes = len(self.ranks_)
        probs = np.zeros((n_samples, n_classes))
        
        # First class probability
        probs[:, 0] = 1 - binary_probs[0]
        
        # Middle class probabilities
        for i in range(1, n_classes - 1):
            probs[:, i] = binary_probs[i-1] - binary_probs[i]
        
        # Last class probability
        probs[:, -1] = binary_probs[-1]
        
        # Normalize probabilities to sum to 1
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs

    def transform(self, X: pd.DataFrame, fit=False) -> pd.DataFrame:
        """Transform input data into the format expected by the model."""
        
        # Identify categorical columns
        categorical_columns = X.select_dtypes(include=["object"]).columns
        
        if fit:
            # Initialize and fit the encoder on categorical data
            self._encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            categorical_features = self._encoder.fit_transform(X[categorical_columns])
        else:
            # Transform categorical data using stored encoder
            categorical_features = self._encoder.transform(X[categorical_columns])

        # Convert to DataFrame with proper column names
        categorical_features = pd.DataFrame(categorical_features, 
                                          columns=self._encoder.get_feature_names_out(categorical_columns),
                                          index=X.index)

        # Handle numerical features if they exist
        numerical_columns = X.drop(columns=categorical_columns, axis=1).columns
        if len(numerical_columns) > 0:
            if fit:
                self._scaler = StandardScaler()
                numerical_features = self._scaler.fit_transform(X[numerical_columns])
            else:
                numerical_features = self._scaler.transform(X[numerical_columns])

            # Combine numerical and categorical features
            numerical_df = pd.DataFrame(numerical_features, 
                                      columns=numerical_columns,
                                      index=X.index)
            X_transformed = pd.concat([numerical_df, categorical_features], axis=1)
        else:
            # If no numerical columns, just use categorical features
            X_transformed = categorical_features

        return X_transformed

