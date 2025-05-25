import os
import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(
    data_path: str,
    target: Union[int, str] = -1,
    sep: str = ";",
    label_map: Optional[dict] = None,
    drop: Optional[list] = None,
    handle_nan: str = 'drop'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess a dataset from a file.
    
    Parameters
    ----------
    data_path : str
        Full path to the data file
    target : Union[int, str], default=-1
        Target variable specification. Can be:
        - int: Index of target column (e.g., -1 for last column)
        - str: Name of target column
    sep : str, default=';'
        Delimiter to use when reading the file
    label_map : Optional[dict], default=None
        Optional mapping to convert target labels to numeric values.
        If None, labels will be mapped to 0-based continuous indices.
    drop : Optional[list], default=None
        List of feature indices or names to drop from the features DataFrame.
    handle_nan : str, default='drop'
        How to handle NaN values. Options are:
        - 'drop': Drop rows containing any NaN values
        - 'error': Raise an error if NaN values are found
        - 'warn': Print a warning if NaN values are found but continue
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X: Features DataFrame
        y: Target Series with mapped labels
    
    Raises
    ------
    FileNotFoundError
        If the data file doesn't exist
    ValueError
        If target specification is invalid
        If handle_nan is not one of ['drop', 'error', 'warn']
        If handle_nan='error' and NaN values are found
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    if handle_nan not in ['drop', 'error', 'warn']:
        raise ValueError("handle_nan must be one of ['drop', 'error', 'warn']")

    # Read the data
    df = pd.read_csv(data_path, sep=sep)
    
    # Handle NaN values
    if df.isnull().any().any():
        if handle_nan == 'error':
            raise ValueError("Data contains NaN values")
        elif handle_nan == 'warn':
            print(f"Warning: Data contains NaN values in columns: {df.columns[df.isnull().any()].tolist()}")
        elif handle_nan == 'drop':
            original_len = len(df)
            df = df.dropna()
            if len(df) < original_len:
                print(f"Dropped {original_len - len(df)} rows containing NaN values")
    
    # Handle target specification
    if isinstance(target, int):
        if target < -len(df.columns) or target >= len(df.columns):
            raise ValueError(f"Target index {target} out of range")
        y = df.iloc[:, target]
        X = df.drop(df.columns[target], axis=1)
    elif isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        y = df[target]
        X = df.drop(target, axis=1)
    else:
        raise ValueError("Target must be either an integer index or column name")
    
    # Drop specified features if requested
    if drop is not None:
        if all(isinstance(d, int) for d in drop):
            drop_cols = X.columns[drop]
        else:
            drop_cols = drop
        X = X.drop(columns=drop_cols)
    
    # Map labels if needed
    if label_map is None:
        unique_labels = sorted(y.unique())
        label_map = {old: new for new, old in enumerate(unique_labels)}
    
    y = y.map(label_map)
    
    return X, y 

def transform_features(
    X: pd.DataFrame,
    fit: bool = False,
    no_scaling: bool = False,
    encoder: Optional[OneHotEncoder] = None,
    scaler: Optional[StandardScaler] = None,
    categorical_columns: Optional[list] = None,
    handle_nan: str = 'drop'
) -> Tuple[pd.DataFrame, OneHotEncoder, Optional[StandardScaler]]:
    """
    Transform input data using one-hot encoding for categoricals and scaling for numericals.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input data to transform
    fit : bool, default=False
        Whether to fit new encoder/scaler or use existing ones
    no_scaling : bool, default=False
        Whether to skip scaling of numerical features
    encoder : Optional[OneHotEncoder], default=None
        Existing encoder to use if fit=False
    scaler : Optional[StandardScaler], default=None
        Existing scaler to use if fit=False
    categorical_columns : Optional[list], default=None
        List of categorical column names. If None, inferred from data types
    handle_nan : str, default='drop'
        How to handle NaN values. Options are:
        - 'drop': Drop rows containing any NaN values
        - 'error': Raise an error if NaN values are found
        - 'warn': Print a warning if NaN values are found but continue
    
    Returns
    -------
    Tuple[pd.DataFrame, OneHotEncoder, Optional[StandardScaler]]
        X_transformed: Transformed DataFrame
        encoder: Fitted OneHotEncoder
        scaler: Fitted StandardScaler (None if no_scaling=True)
        
    Raises
    ------
    ValueError
        If handle_nan is not one of ['drop', 'error', 'warn']
        If handle_nan='error' and NaN values are found
    """
    if handle_nan not in ['drop', 'error', 'warn']:
        raise ValueError("handle_nan must be one of ['drop', 'error', 'warn']")

    # Handle NaN values
    if X.isnull().any().any():
        if handle_nan == 'error':
            raise ValueError("Data contains NaN values")
        elif handle_nan == 'warn':
            print(f"Warning: Data contains NaN values in columns: {X.columns[X.isnull().any()].tolist()}")
        elif handle_nan == 'drop':
            original_len = len(X)
            X = X.dropna()
            if len(X) < original_len:
                print(f"Dropped {original_len - len(X)} rows containing NaN values")

    if categorical_columns is None:
        categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

    # Handle categorical features
    if fit:
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        categorical_features = encoder.fit_transform(X[categorical_columns]) if len(categorical_columns) > 0 else np.empty((len(X), 0))
    else:
        categorical_features = encoder.transform(X[categorical_columns]) if len(categorical_columns) > 0 else np.empty((len(X), 0))

    if len(categorical_columns) > 0:
        categorical_features = pd.DataFrame(
            categorical_features,
            columns=encoder.get_feature_names_out(categorical_columns),
            index=X.index
        )
    else:
        categorical_features = pd.DataFrame(index=X.index)

    # Handle numerical features
    numerical_columns = X.drop(columns=categorical_columns, axis=1).columns.tolist()
    if len(numerical_columns) > 0:
        if no_scaling:
            numerical_df = X[numerical_columns].copy()
            scaler = None
        else:
            if fit:
                scaler = StandardScaler()
                numerical_features = scaler.fit_transform(X[numerical_columns])
            else:
                numerical_features = scaler.transform(X[numerical_columns])
            numerical_df = pd.DataFrame(numerical_features, columns=numerical_columns, index=X.index)
        X_transformed = pd.concat([numerical_df, categorical_features], axis=1)
    else:
        X_transformed = categorical_features

    return X_transformed, encoder, scaler 