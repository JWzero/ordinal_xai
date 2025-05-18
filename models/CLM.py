import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_is_fitted, validate_data
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import OneHotEncoder
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


