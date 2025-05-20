import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from utils.data_utils import transform_features
from sklearn.utils.validation import check_X_y, check_is_fitted, validate_data
from statsmodels.miscmodels.ordinal_model import OrderedModel
from models.base_model import BaseOrdinalModel

class CLM(BaseEstimator, BaseOrdinalModel):
    def __init__(self, link="logit"):
        super().__init__()  # Initialize base class
        self.link = link  # Hyperparameter

    def get_params(self, deep=True):
        return {"link": self.link}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self  

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CLM":
        """Fit the ordered logistic regression model, handling categorical variables."""

        # Store feature names for consistency during prediction
        self.feature_names_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        self.ranks_ = np.unique(y)

        X_transformed = self.transform(X, fit = True)

        # Run scikit-learn's check
        X, y = check_X_y(X_transformed, y, ensure_2d=True)

        # Fit Ordered Model
        link_functions = {"logit": "logit", "probit": "probit"}
        if self.link not in link_functions:
            raise ValueError(f"Invalid link function '{self.link}'. Choose from {list(link_functions.keys())}.")

        self._model = OrderedModel(y, X, distr=link_functions[self.link])
        self._result = self._model.fit(method='bfgs', disp=False)
        self.params_ = self._result.params

        # Set fitted flag
        self.is_fitted_ = True

        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the ordinal class labels."""
        return self.predict_proba(X).argmax(axis=1)


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        
        # Check if fit has been called
        check_is_fitted(self)
        
        X_transformed = self.transform(X, fit = False)

        # Compute probabilities
    
        return self._result.predict(X_transformed.values)

        
    def transform(self, X: pd.DataFrame, fit=False, no_scaling=False) -> pd.DataFrame:
        """Transform input data into the format expected by the model."""
        if fit:
            self._encoder = None
            self._scaler = None
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


