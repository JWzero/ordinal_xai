import argparse
import importlib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

def load_data(dataset_name):
    """Load dataset from the data/ folder."""
    data_path = os.path.join("data", dataset_name)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found in 'data/' folder.")

    df = pd.read_csv(data_path, sep=";")
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target (ordinal labels)

    return X, y

def load_model(model_name):
    """Dynamically import and initialize model from models/ folder."""
    module_path = f"models.{model_name}"
    module = importlib.import_module(module_path)
    model_class = getattr(module, model_name)  # Assumes class name matches file name
    return model_class()

def evaluate_model(model, X, y):
    """Evaluate the model using accuracy and MAE."""
    predictions = model.predict(X)
    
    accuracy = accuracy_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    
    print(f"Model Evaluation:\n - Accuracy: {accuracy:.4f}\n - MAE: {mae:.4f}")

def load_interpreter(interpreter_name):
    """Dynamically import and initialize interpretability method."""
    module_path = f"interpretability.{interpreter_name}"
    module = importlib.import_module(module_path)
    interpreter_class = getattr(module, interpreter_name)  # Assumes class name matches file name
    return interpreter_class()

def main(args):
    # Load dataset
    X, y = load_data(args.dataset)
    print(f"Loaded dataset '{args.dataset}' with shape {X.shape}.")

    # Load model
    model = load_model(args.model)
    print(f"Using model: {args.model}")

    # Train model
    model.fit(X, y)

    # Evaluate model
    evaluate_model(model, X, y)

    # Load interpretability method
    interpreter = load_interpreter(args.interpreter)
    print(f"Using interpretability method: {args.interpreter}")

    # Explain first instance
    explanation = interpreter.explain_instance(model, X[0])
    print(f"Explanation for first instance: {explanation}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ordinal regression with interpretability.")
    
    parser.add_argument("--dataset", type=str, default="winequality-red.csv",
                        help="Dataset filename in 'data/' folder (default: 'winequality-red.csv').")
    parser.add_argument("--model", type=str, default="CLM",
                        help="Model filename (without .py) in 'models/' folder (default: 'CLM').")
    parser.add_argument("--interpreter", type=str, default="DummyInterpretation",
                        help="Interpretability method filename (without .py) in 'interpretability/' folder (default: 'DummyInterpretation').")

    args = parser.parse_args()
    main(args)
