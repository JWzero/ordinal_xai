import argparse
import importlib
import os
import numpy as np
import pandas as pd
from utils.evaluation_metrics import evaluate_ordinal_model, print_evaluation_results
import warnings
import sys

if sys.platform == "win32":
    warnings.filterwarnings(
        "ignore",
        message=".*'super' object has no attribute '__del__'.*",
        category=UserWarning,
        module="joblib.externals.loky.backend.resource_tracker"
    )

def load_data(dataset_name):
    """Load dataset from the data/ folder."""
    if  not dataset_name.endswith(".csv"):
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

def load_model(model_name, link_function="logit"):
    """Dynamically import and initialize the model with optional parameters."""
    module_path = f"models.{model_name}"
    module = importlib.import_module(module_path)
    model_class = getattr(module, model_name)  # Assumes class name matches file name

    if model_name == "CLM":
        return model_class(link=link_function)  # Pass link function
    return model_class()

def evaluate_model(model, X, y):
    """Evaluate the model using comprehensive ordinal regression metrics."""
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

def load_interpretation(method_name, model, X, y, **kwargs):
    """Dynamically import and initialize the interpretation method with model and data."""
    module_path = f"interpretation.{method_name}"
    module = importlib.import_module(module_path)
    
    interpretation_class = getattr(module, method_name)
    return interpretation_class(model, X, y, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description="Run ordinal regression with interpretability.")
    
    parser.add_argument("--dataset", type=str, default="dummy.csv",
                        help="Dataset filename in 'data/' folder (default: 'dummy.csv').")
    parser.add_argument("--model", type=str, default="CLM",
                        help="Model filename (without .py) in 'models/' folder (default: 'CLM').")
    parser.add_argument("--interpretation", type=str, default="DummyInterpretation",
                        help="Interpretability method filename (without .py) in 'interpretation/' folder (default: 'DummyInterpretation').")
    parser.add_argument("--link", type=str, default="logit",
                    help="Link function for CLM model (default: 'logit'). Options: 'logit', 'probit'.")
    parser.add_argument("--observation_idx", type=int, default=None,
                    help="Index of the observation to interpret (only for local explanations).")
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated list of feature indices to include in the explanation (optional).")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Comma-separated list of metrics to use for LOCO interpretation (optional).")
    parser.add_argument("--sampling", type=str, default=None,
                        help="Sampling method for LIME (default: None). Options: None, 'uniform', 'grid'.")
    parser.add_argument(
        "--base",
        type=str,
        default="logistic",
        choices=["logistic", "svm", "rf", "xgb"],
        help="Base classifier for OrdinalChain (logistic, svm, rf, xgb)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic",
        choices=["logistic", "decision_tree"],
        help="Model type for LIME interpretation (default: 'logistic'). Options: 'logistic', 'decision_tree'"
    )

    return parser.parse_args()

def main(args):
    # Load dataset
    X, y = load_data(args.dataset)
    print(f"Loaded dataset '{args.dataset}' with shape {X.shape}.")
    
    # Load model
    if args.model == "OrdinalChain":
        from models.OrdinalChain import OrdinalChain
        model = OrdinalChain(base_classifier=args.base)
    else:
        model = load_model(args.model, args.link)
    print(f"Using model: {args.model}")

    # Train model
    model.fit(X, y)
    print(model.get_params())

    # Evaluate model
    evaluate_model(model, X, y)

    # Load interpretability method
    if args.features:
        feature_subset = [int(i) if i.isdigit() else i for i in args.features.split(",")]
    else:
        feature_subset = None
    
    # Prepare kwargs for interpretation method
    interpretation_kwargs = {}
    # Add sampling parameter for LIME if specified
    if args.interpretation == "LIME":
        if args.sampling:
            interpretation_kwargs["sampling"] = args.sampling
        interpretation_kwargs["model_type"] = args.model_type

    interpretation = load_interpretation(args.interpretation, model, X, y, **interpretation_kwargs)
    # Prepare kwargs for explain
    explain_kwargs = {}
    if args.metrics:
        explain_kwargs["metrics"] = args.metrics.split(",")
    explanation = interpretation.explain(observation_idx=args.observation_idx, feature_subset=feature_subset, plot=True, **explain_kwargs)

if __name__ == "__main__":
    args = parse_args()
    main(args)
