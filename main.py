import argparse
import importlib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

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
    """Evaluate the model using accuracy and MAE."""
    predictions = model.predict(X)
    
    accuracy = accuracy_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    
    print(f"Model Evaluation:\n - Accuracy: {accuracy:.4f}\n - MAE: {mae:.4f}")

def load_interpretation(method_name, model, X, y):
    """Dynamically import and initialize the interpretation method with model and data."""
    module_path = f"interpretation.{method_name}"
    module = importlib.import_module(module_path)
    
    interpretation_class = getattr(module, method_name)
    return interpretation_class(model, X, y)


def main(args):
    # Load dataset
    X, y = load_data(args.dataset)
    print(f"Loaded dataset '{args.dataset}' with shape {X.shape}.")
    
    # Load model
    model = load_model(args.model,args.link)
    print(f"Using model: {args.model}")

    # Train model
    model.fit(X, y)
    print(model.get_params())

    # Evaluate model
    evaluate_model(model, X, y)


    # Load interpretability method
    if args.features:
        #covert numbers to ints, else keep feature names
        feature_subset = [int(i) if i.isdigit() else i for i in args.features.split(",")]
    else:
        feature_subset = None

    interpretation = load_interpretation(args.interpretation, model, X, y)
    explanation = interpretation.explain(observation_idx=args.observation_idx, feature_subset=feature_subset)
    


if __name__ == "__main__":
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



    args = parser.parse_args()
    main(args)
