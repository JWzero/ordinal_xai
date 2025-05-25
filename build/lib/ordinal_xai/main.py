"""
Main module for ordinal regression with interpretability.

This module provides the main entry point for running ordinal regression models
with various interpretability methods. It handles data loading, model training,
evaluation, and interpretation of results.

The module supports multiple models (CLM, ONN, OBD) and various
interpretability methods (PDP, ICE, LIME, LOCO, PFI) for explaining model predictions.

Example:
    # Basic usage
    python main.py --dataset wine.csv --model CLM --interpretation PDP

    # With model parameters
    python main.py --dataset wine.csv --model CLM --model_params '{"link": "probit"}'

    # With interpretation parameters
    python main.py --dataset wine.csv --model CLM --interpretation LIME --interpretation_params '{"sampling": "uniform", "model_type": "decision_tree"}'

    # With both model and interpretation parameters
    python main.py --dataset wine.csv --model OBD --model_params '{"base_classifier": "svm", "decomposition_type": "one-vs-next"}' --interpretation PDP
"""

import argparse
import importlib
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List, Union
from utils.evaluation_metrics import evaluate_ordinal_model, print_evaluation_results
import warnings
import sys
import json

if sys.platform == "win32":
    warnings.filterwarnings(
        "ignore",
        message=".*'super' object has no attribute '__del__'.*",
        category=UserWarning,
        module="joblib.externals.loky.backend.resource_tracker"
    )

def load_data(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess dataset from the data/ folder.
    
    Args:
        dataset_name (str): Name of the dataset file (with or without .csv extension)
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X: Feature matrix
            - y: Target variable with ordinal labels mapped to 0-based indices
            
    Raises:
        FileNotFoundError: If the specified dataset is not found in the data/ folder
    """
    if not dataset_name.endswith(".csv"):
        dataset_name = dataset_name + ".csv"
    data_path = os.path.join("data", dataset_name)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found in 'data/' folder.")

    df = pd.read_csv(data_path, sep=";")
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target (ordinal labels)
    unique_labels = sorted(y.unique())  # Get sorted unique labels
    label_map = {old: new for new, old in enumerate(unique_labels)}  # Create mapping
    y = y.map(label_map)  # Map to 0-based continuous indices

    return X, y

def load_model(model_name: str, **model_params) -> Any:
    """
    Dynamically import and initialize the model with optional parameters.
    
    Args:
        model_name (str): Name of the model to load (e.g., 'CLM', 'ONN', 'OBD')
        **model_params: Additional keyword arguments for model initialization
            Common parameters:
            - link (str): Link function for CLM model ('logit' or 'probit')
            - hidden_layers(List[int]): Hidden layer sizes for ONN
            - base_classifier (str): Base classifier for OBD ('logistic', 'svm', 'rf', 'xgb')
            - decomposition_type (str): Decomposition strategy for OBD ('one-vs-following' or 'one-vs-next')
            
    Returns:
        Any: Initialized model instance
        
    Raises:
        ImportError: If the model module cannot be imported
        AttributeError: If the model class is not found in the module
    """
    # Convert model_name to lowercase for file import
    module_path = f"models.{model_name.lower()}"
    module = importlib.import_module(module_path)
    
    # Map of file names to class names
    class_name_map = {
        'clm': 'CLM',
        'onn': 'ONN',
        'obd': 'OBD'
    }
    
    # Get the correct class name
    class_name = class_name_map.get(model_name.lower(), model_name)
    model_class = getattr(module, class_name)
    
    return model_class(**model_params)

def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Evaluate the model using comprehensive ordinal regression metrics.
    
    Args:
        model: Trained model instance
        X (pd.DataFrame): Feature matrix
        y (pd.Series): True labels
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    predictions = model.predict(X)
    
    # Get probability predictions if the model supports it
    proba_predictions = None
    try:
        proba_predictions = model.predict_proba(X)
        print(f"Probability predictions shape: {proba_predictions.shape}")
    except (AttributeError, NotImplementedError) as e:
        print(f"Model does not support predict_proba: {e}")
    except Exception as e:
        print(f"Error getting probability predictions: {e}")
    
    # Evaluate the model with all available metrics
    results = evaluate_ordinal_model(y, predictions, proba_predictions)
    
    # Print the evaluation results
    print_evaluation_results(results)
    
    return results

def load_interpretation(method_name: str, model: Any, X: pd.DataFrame, y: pd.Series, **kwargs) -> Any:
    """
    Dynamically import and initialize the interpretation method with model and data.
    
    Args:
        method_name (str): Name of the interpretation method
        model: Trained model instance
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        **kwargs: Additional keyword arguments for the interpretation method
            Common parameters:
            - sampling (str): Sampling method for LIME ('uniform' or 'grid')
            - model_type (str): Model type for LIME ('logistic' or 'decision_tree')
            - metrics (List[str]): Metrics for LOCO interpretation
            
    Returns:
        Any: Initialized interpretation method instance
        
    Raises:
        ImportError: If the interpretation module cannot be imported
        AttributeError: If the interpretation class is not found in the module
    """
    # Convert method_name to lowercase for file import
    module_path = f"interpretation.{method_name.lower()}"
    module = importlib.import_module(module_path)
    
    # Map of file names to class names
    class_name_map = {
        'pdp': 'PDP',
        'pdp_prob': 'PDPProb',
        'ice': 'ICE',
        'ice_prob': 'ICEProb',
        'lime': 'LIME',
        'loco': 'LOCO',
        'pfi': 'PFI',
        'dummy_interpretation': 'DummyInterpretation'
    }
    
    # Get the correct class name
    class_name = class_name_map.get(method_name.lower(), method_name)
    interpretation_class = getattr(module, class_name)
    return interpretation_class(model, X, y, **kwargs)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Run ordinal regression with interpretability.")
    
    parser.add_argument("--dataset", type=str, default="dummy.csv",
                        help="Dataset filename in 'data/' folder (default: 'dummy.csv').")
    parser.add_argument("--model", type=str, default="CLM",
                        help="Model filename (without .py) in 'models/' folder (default: 'CLM').")
    parser.add_argument("--interpretation", type=str, default="DummyInterpretation",
                        help="Interpretability method filename (without .py) in 'interpretation/' folder (default: 'DummyInterpretation').")
    parser.add_argument("--model_params", type=str, default=None,
                        help="JSON string of model parameters (e.g., '{\"link\": \"probit\", \"hidden_layer_sizes\": [50, 20]}').")
    parser.add_argument("--interpretation_params", type=str, default=None,
                        help="JSON string of interpretation parameters (e.g., '{\"sampling\": \"uniform\", \"model_type\": \"decision_tree\"}').")
    parser.add_argument("--observation_idx", type=int, default=None,
                        help="Index of the observation to interpret (only for local explanations).")
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated list of feature indices to include in the explanation (optional).")

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    """
    Main function to run the ordinal regression and interpretation pipeline in a simple way.
    
    This function orchestrates the entire process:
    1. Loads and preprocesses the dataset
    2. Initializes and trains the specified model
    3. Evaluates the model performance
    4. Generates and visualizes model interpretations
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Parse model parameters
    model_params = {}
    if args.model_params:
        try:
            model_params = json.loads(args.model_params)
        except json.JSONDecodeError:
            print("Error: --model_params must be a valid JSON string.")
            sys.exit(1)

    # Parse interpretation parameters
    interpretation_params = {}
    if args.interpretation_params:
        try:
            interpretation_params = json.loads(args.interpretation_params)
        except json.JSONDecodeError:
            print("Error: --interpretation_params must be a valid JSON string.")
            sys.exit(1)

    # Load dataset
    X, y = load_data(args.dataset)
    print(f"Loaded dataset '{args.dataset}' with shape {X.shape}.")
    
    # Load and initialize model
    model = load_model(args.model, **model_params)
    print(f"Using model: {args.model}")
    if model_params:
        print(f"Model parameters: {model_params}")

    # Train model
    model.fit(X, y)

    # Evaluate model
    evaluate_model(model, X, y)

    # Load interpretability method
    if args.features:
        feature_subset = [int(i) if i.isdigit() else i for i in args.features.split(",")]
    else:
        feature_subset = None
    
    # Initialize interpretation method
    interpretation = load_interpretation(args.interpretation, model, X, y, **interpretation_params)
    if interpretation_params:
        print(f"Interpretation parameters: {interpretation_params}")

    # Generate explanation
    explanation = interpretation.explain(observation_idx=args.observation_idx, feature_subset=feature_subset, plot=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
