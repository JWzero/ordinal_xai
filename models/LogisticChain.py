import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.validation import check_is_fitted
import sys
import os

# Add the root directory to the path to make imports work when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_model import BaseOrdinalModel

class LogisticChain(BaseEstimator, BaseOrdinalModel):
    """
    One-vs-Next Binary Decomposition Ordinal Regression Model.
    
    For K ordinal classes (0, 1, 2, ..., K-1), it trains K-1 binary classifiers:
    - Classifier 1: Class 0 vs. Class 1
    - Classifier 2: Class 1 vs. Class 2
    - ...and so on
    
    This approach better captures the ordinal relationship between classes
    compared to standard one-vs-all or one-vs-rest approaches.
    
    Parameters
    ----------
    base_estimator : estimator object, default=LogisticRegression()
        The base binary classifier to use.
    """
    
    def __init__(self, base_estimator=None, C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42):
        self.base_estimator = base_estimator
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
    
    def get_params(self, deep=True):
        """Return model parameters for scikit-learn compatibility."""
        return {
            "base_estimator": self.base_estimator,
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state
        }
    
    def set_params(self, **params):
        """Set parameters for scikit-learn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def transform(self, X, fit=False):
        """Transform input data into the format expected by the model."""
        
        # Store feature names for consistency during prediction
        if fit:
            self.feature_names_ = X.columns.tolist()
            self.n_features_in_ = X.shape[1]
            
        # Identify categorical columns
        categorical_columns = X.select_dtypes(include=["object"]).columns
        
        if fit:
            # Initialize and fit the encoder on categorical data
            self._encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            if len(categorical_columns) > 0:
                categorical_features = self._encoder.fit_transform(X[categorical_columns])
        else:
            # Transform categorical data using stored encoder
            if len(categorical_columns) > 0:
                categorical_features = self._encoder.transform(X[categorical_columns])

        # Convert categorical features to DataFrame with proper column names
        if len(categorical_columns) > 0:
            categorical_features = pd.DataFrame(
                categorical_features, 
                columns=self._encoder.get_feature_names_out(categorical_columns),
                index=X.index
            )

        # Handle numerical features if they exist
        numerical_columns = X.drop(columns=categorical_columns, axis=1).columns
        if len(numerical_columns) > 0:
            if fit:
                self._scaler = StandardScaler()
                numerical_features = self._scaler.fit_transform(X[numerical_columns])
            else:
                numerical_features = self._scaler.transform(X[numerical_columns])

            # Combine numerical and categorical features
            numerical_df = pd.DataFrame(
                numerical_features, 
                columns=numerical_columns,
                index=X.index
            )
            
            if len(categorical_columns) > 0:
                X_transformed = pd.concat([numerical_df, categorical_features], axis=1)
            else:
                X_transformed = numerical_df
        else:
            # If no numerical columns, just use categorical features
            X_transformed = categorical_features

        return X_transformed
    
    def fit(self, X, y):
        """
        Train the model using one-vs-next binary decomposition.
        
        Parameters
        ----------
        X : pandas DataFrame, shape (n_samples, n_features)
            Training data.
        y : pandas Series or numpy array, shape (n_samples,)
            Target values (ordinal classes).
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Transform the input data
        X_transformed = self.transform(X, fit=True)
        
        # Store the unique classes and their order
        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ <= 2:
            raise ValueError("LogisticChain requires at least 3 classes for ordinal regression.")
        
        # Prepare default base estimator if none provided
        if self.base_estimator is None:
            self.base_estimator_ = LogisticRegression(
                C=self.C,
                penalty=self.penalty,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
        else:
            self.base_estimator_ = self.base_estimator
        
        # Train K-1 binary classifiers for one-vs-next comparisons
        self.binary_clfs_ = []
        
        for i in range(self.n_classes_ - 1):
            # Create binary problem: class i vs class i+1
            mask = np.isin(y, [self.classes_[i], self.classes_[i+1]])
            if np.sum(mask) == 0:
                # Skip this classifier if no samples for this pair
                self.binary_clfs_.append(None)
                continue
                
            X_binary = X_transformed.iloc[mask]
            y_binary = y.iloc[mask] if isinstance(y, pd.Series) else y[mask]
            
            # Remap labels to 0 and 1
            y_binary = np.array([0 if label == self.classes_[i] else 1 for label in y_binary])
            
            # Clone and fit the classifier
            clf = clone(self.base_estimator_)
            clf.fit(X_binary, y_binary)
            self.binary_clfs_.append(clf)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : pandas DataFrame, shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        P : array, shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model.
        """
        # Check if fit had been called
        check_is_fitted(self, ['binary_clfs_', 'classes_'])
        
        # Transform input data
        X_transformed = self.transform(X, fit=False)
        
        # Initialize probabilities for each class
        n_samples = X_transformed.shape[0]
        probas = np.zeros((n_samples, self.n_classes_))
        
        # Start with all probability mass on the first class
        probas[:, 0] = 1.0
        
        # Redistribute probability mass based on binary classifier predictions
        for i, clf in enumerate(self.binary_clfs_):
            if clf is None:
                continue
                
            # Get probability of being in the higher class
            p_higher = clf.predict_proba(X_transformed)[:, 1]
            
            # Redistribute probability from class i to higher classes
            remaining_mass = probas[:, i].copy()
            probas[:, i] *= (1 - p_higher)
            probas[:, i+1] += remaining_mass * p_higher
        
        return probas
    
    def predict(self, X):
        """
        Predict ordinal classes.
        
        Parameters
        ----------
        X : pandas DataFrame, shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y : array, shape (n_samples,)
            Predicted class labels.
        """
        # Get class probabilities
        probas = self.predict_proba(X)
        
        # Return class with highest probability
        return self.classes_[np.argmax(probas, axis=1)]


if __name__ == "__main__":
    # A simple test case for the LogisticChain model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_absolute_error
    import os
    
    # First make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Load data
    data_path = os.path.join("data", "dummy.csv")
    df = pd.read_csv(data_path, sep=";")
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target (ordinal labels)
    
    print(f"Loaded dataset with shape {X.shape}.")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and train the model
    model = LogisticChain()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"LogisticChain Model Evaluation:")
    print(f" - Accuracy: {accuracy:.4f}")
    print(f" - MAE: {mae:.4f}")
    
    # Show probability distribution for the first few samples
    probas = model.predict_proba(X_test.iloc[:5])
    print("\nProbability distributions for first 5 test samples:")
    for i, proba in enumerate(probas):
        print(f"Sample {i+1}: {dict(zip(model.classes_, proba.round(3)))}") 