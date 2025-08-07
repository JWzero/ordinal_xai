"""
Leave-One-Covariate-Out (LOCO) interpretation method for ordinal regression models.

This module implements the LOCO method for feature importance analysis in ordinal regression.
LOCO works by measuring how model performance changes when each feature is removed from the dataset.
The method can be used in both global (dataset-wide) and local (instance-specific) modes.

Key Features:
- Global and local feature importance analysis
- Support for multiple evaluation metrics
- Train-test split option for more robust evaluation
- Comprehensive visualization of feature importance
- Support for both classification and probability-based metrics

The implementation is particularly useful for:
- Understanding feature importance in ordinal regression models
- Identifying key features that influence model predictions
- Analyzing feature importance at both global and local levels
- Comparing feature importance across different evaluation metrics
"""

from typing import Optional, List, Dict, Union, Callable, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from .base_interpretation import BaseInterpretation
from sklearn.metrics import accuracy_score, mean_absolute_error
import copy
from sklearn.model_selection import train_test_split
from ..utils.evaluation_metrics import (
    adjacent_accuracy, mze, mae, mse, weighted_kappa, cem, _get_class_counts,
    spearman_correlation, kendall_tau,
    ranked_probability_score, ordinal_weighted_ce,
    evaluate_ordinal_model
)

class LOCO(BaseInterpretation):
    """
    Leave-One-Covariate-Out (LOCO) interpretation method for feature importance.
    
    This class implements the LOCO method for analyzing feature importance in ordinal regression
    models. It measures the impact of each feature by evaluating how model performance changes
    when the feature is removed from the dataset.
    
    Parameters
    ----------
    model : object
        The trained ordinal regression model. Must implement fit and predict methods.
    X : pd.DataFrame
        Dataset used for interpretation. Should contain the same features used
        during model training.
    y : pd.Series, optional
        Target labels. Required for evaluation but optional for initialization.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split when using
        train-test split evaluation.
    random_state : int, optional
        Random state for reproducibility of train-test splits.
    use_train_test_split : bool, default=True
        Whether to use train-test split for evaluation. If False, the entire
        dataset is used for both training and evaluation.
        
    Attributes
    ----------
    available_metrics : dict
        Dictionary of available evaluation metrics and their corresponding functions.
    test_size : float
        Proportion of data used for testing.
    random_state : int
        Random state for reproducibility.
    use_train_test_split : bool
        Whether to use train-test split.
    X_train, X_test : pd.DataFrame
        Training and test datasets when using train-test split.
    y_train, y_test : pd.Series
        Training and test labels when using train-test split.
    original_predictions : array-like
        Predictions from the original model.
    original_proba_predictions : array-like, optional
        Probability predictions from the original model if available.
    original_results : dict
        Evaluation results from the original model.
    """
    
    def __init__(self, model, X, y=None, test_size=0.2, random_state=None, use_train_test_split=True):
        """
        Initialize the LOCO interpretation method.
        
        Parameters
        ----------
        model : object
            The trained ordinal regression model
        X : pd.DataFrame
            Dataset used for interpretation
        y : pd.Series, optional
            Target labels
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        random_state : int, optional
            Random state for reproducibility
        use_train_test_split : bool, default=True
            Whether to use train-test split for evaluation
        """
        super().__init__(model, X, y)
        self.available_metrics = {
            'mze': mze,
            'mae': mae,
            'mse': mse,
            'adjacent_accuracy': adjacent_accuracy,
            'weighted_kappa_quadratic': lambda y_true, y_pred: weighted_kappa(y_true, y_pred, weights='quadratic'),
            'weighted_kappa_linear': lambda y_true, y_pred: weighted_kappa(y_true, y_pred, weights='linear'),
            'cem': cem,
            'spearman_correlation': spearman_correlation,
            'kendall_tau': kendall_tau,
            'ranked_probability_score': ranked_probability_score,
            'ordinal_weighted_ce_linear': lambda y_true, y_pred_proba: ordinal_weighted_ce(y_true, y_pred_proba, alpha=1),
            'ordinal_weighted_ce_quadratic': lambda y_true, y_pred_proba: ordinal_weighted_ce(y_true, y_pred_proba, alpha=2)
        }
        self.test_size = test_size
        self.random_state = random_state
        self.use_train_test_split = use_train_test_split
        if self.use_train_test_split and self.y is not None:
            self._perform_train_test_split()
        else:
            if self.y is not None:
                self.original_predictions = self.model.predict(self.X)
                try:
                    self.original_proba_predictions = self.model.predict_proba(self.X)
                except (AttributeError, NotImplementedError):
                    self.original_proba_predictions = None
                self.original_results = evaluate_ordinal_model(self.y, self.original_predictions, self.original_proba_predictions)
    
    def _perform_train_test_split(self):
        """
        Perform train-test split on the data and evaluate the original model.
        
        This method:
        1. Splits the data into training and test sets
        2. Creates a copy of the model for training
        3. Fits the model on the training data
        4. Makes predictions on the test set
        5. Evaluates the model using all available metrics
        
        Raises
        ------
        ValueError
            If the model class is not suitable for refitting
        """
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Create a copy of the model for training
        try:
            model_class = type(self.model)
            self.train_model = model_class(**getattr(self.model, 'get_params', lambda: {})())
        except Exception as e:
            raise ValueError("Model class is not suitable for refitting. Error: " + str(e))
        
        # Fit the model on the training data
        self.train_model.fit(self.X_train, self.y_train)
        
        # Make predictions on the test set
        self.original_predictions = self.train_model.predict(self.X_test)
        
        # Get probability predictions if available
        try:
            self.original_proba_predictions = self.train_model.predict_proba(self.X_test)
        except (AttributeError, NotImplementedError):
            self.original_proba_predictions = None
        
        # Evaluate the model on the test set
        self.original_results = evaluate_ordinal_model(
            self.y_test, self.original_predictions, self.original_proba_predictions
        )
    
    def explain(self, observation_idx=None, feature_subset=None, plot=False, metrics=None, title=True):
        """
        Generate LOCO feature importance scores.
        
        This method computes feature importance by measuring how model performance
        changes when each feature is removed. It can operate in two modes:
        1. Global mode (default): Evaluates feature importance across the entire dataset
        2. Local mode (when observation_idx is provided): Evaluates feature importance
        for a specific instance
        
        Parameters
        ----------
        observation_idx : int, optional
            Index of specific instance to analyze (local explanation)
        feature_subset : list, optional
            List of feature names or indices to analyze
        plot : bool, default=False
            Whether to create visualizations
        metrics : list, optional
            List of metrics to use for feature importance calculation
        title : bool, default=True
            Whether to add a suptitle to the visualization
            
        Returns
        -------
        dict
            Dictionary containing feature importance scores for each metric
        """
        _is_local = observation_idx is not None
        if _is_local:
            # Restrict metrics for local explanation
            if metrics is None:
                metrics_to_use = [m for m in self.available_metrics if m not in [
                    'spearman_correlation', 'kendall_tau', 'weighted_kappa_quadratic', 'weighted_kappa_linear']]
            else:
                metrics_to_use = [m for m in metrics if m not in [
                    'spearman_correlation', 'kendall_tau', 'weighted_kappa_quadratic', 'weighted_kappa_linear']]
            metric_funcs = self.available_metrics
            X_test = self.X.iloc[[observation_idx]]
            y_test = self.y.iloc[[observation_idx]] if hasattr(self.y, 'iloc') else np.array([self.y[observation_idx]])
            X_train = self.X
            y_train = self.y
            model_to_use = self.model
            original_predictions = model_to_use.predict(X_test)
            try:
                original_proba_predictions = model_to_use.predict_proba(X_test)
            except (AttributeError, NotImplementedError):
                original_proba_predictions = None
            whole_dataset_class_counts = _get_class_counts(self.y)
            original_results = evaluate_ordinal_model(y_test, original_predictions, original_proba_predictions, metrics=metrics_to_use, class_counts=whole_dataset_class_counts,zero_indexed=True)
        else:
            if metrics is None:
                metrics_to_use = list(self.available_metrics.keys())
            else:
                metrics_to_use = metrics
            metric_funcs = self.available_metrics
            if self.use_train_test_split and self.y is not None:
                X_train = self.X_train
                X_test = self.X_test
                y_train = self.y_train
                y_test = self.y_test
                model_to_use = self.train_model
            else:
                X_train = self.X
                X_test = self.X
                y_train = self.y
                y_test = self.y
                model_to_use = self.model
            original_results = self.original_results
        if feature_subset is None:
            feature_subset = X_train.columns.tolist()
        else:
            feature_subset = [X_train.columns[i] if isinstance(i, int) else i for i in feature_subset]
        results = {
            'feature_importance': {},
            'metric_drops': {}
        }
        for feature in feature_subset:
            X_train_without_feature = X_train.drop(columns=[feature])
            X_test_without_feature = X_test.drop(columns=[feature])
            model_class = type(model_to_use)
            try:
                model_copy = model_class(**getattr(model_to_use, 'get_params', lambda: {})())
            except Exception as e:
                raise ValueError("Model class is not suitable for refitting. Error: " + str(e))
            model_copy.fit(X_train_without_feature, y_train)
            predictions = model_copy.predict(X_test_without_feature)
            try:
                proba_predictions = model_copy.predict_proba(X_test_without_feature)
            except (AttributeError, NotImplementedError):
                proba_predictions = None
            whole_dataset_class_counts = _get_class_counts(self.y)
            feature_results = evaluate_ordinal_model(y_test, predictions, proba_predictions, metrics=metrics_to_use, class_counts=whole_dataset_class_counts,zero_indexed=True)
            metric_drops = {}
            for metric in metrics_to_use:
                if metric in original_results and metric in feature_results:
                    if metric in ['mae', 'mse', 'mze', 'ranked_probability_score', 
                                 'ordinal_weighted_ce_linear', 'ordinal_weighted_ce_quadratic']:
                        metric_drops[metric] = feature_results[metric] - original_results[metric]
                    else:
                        metric_drops[metric] = original_results[metric] - feature_results[metric]
            results['feature_importance'][feature] = metric_drops
            for metric, drop in metric_drops.items():
                if metric not in results['metric_drops']:
                    results['metric_drops'][metric] = {}
                results['metric_drops'][metric][feature] = drop
        if plot:
            self._plot_feature_importance(results, metrics=metrics_to_use, title=title)
        return results
    
    def _plot_feature_importance(self, results, metrics=None, title=True):
        """
        Create visualizations for feature importance.
        
        This method creates bar plots showing feature importance scores for each metric.
        The plots are arranged in a grid, with one subplot per metric.
        
        Parameters
        ----------
        results : dict
            Dictionary containing all results from the LOCO analysis
        metrics : list, optional
            List of metrics to plot (defaults to all in results)
        title : bool, default=True
            Whether to add a suptitle to the visualization
            
        Notes
        -----
        - Features are sorted by importance score
        - Error metrics are shown in red with "Increase" in the title
        - Other metrics are shown in green with "Drop" in the title
        - Feature names are rotated 90 degrees for better readability
        - Score values are displayed in the middle of each bar
        """
        import math
        import matplotlib.pyplot as plt

        metric_abbr = {
            'adjacent_accuracy': 'AA',
            'weighted_kappa_linear': 'LWK',
            'weighted_kappa_quadratic': 'QWK',
            'spearman_correlation': 'Rho',
            'kendall_tau': 'Tau',
            'ranked_probability_score': 'RPS',
            'ordinal_weighted_ce_linear': 'LW-OCE',
            'ordinal_weighted_ce_quadratic': 'QW-OCE',
            'mae': 'MAE',
            'mse': 'MSE',
            'mze': 'MZE',
            'cem': 'CEM',
        }
        if metrics is None:
            metrics = list(results['metric_drops'].keys())
        n_metrics = len(metrics)
        n_cols = min(4, n_metrics) if n_metrics > 1 else 1
        n_rows = math.ceil(n_metrics / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, metric in enumerate(metrics):
            feature_scores = results['metric_drops'][metric]
            ax = axes[i]
            sorted_features = sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            features = [str(f[0]) for f in sorted_features]
            scores = [f[1] for f in sorted_features]
            if metric in ['mae', 'mse', 'mze', 'ranked_probability_score',
                         'ordinal_weighted_ce_linear', 'ordinal_weighted_ce_quadratic']:
                color = 'red'
                title_suffix = "Increase"
            else:
                color = 'green'
                title_suffix = "Drop"
            abbr = metric_abbr.get(metric, metric)
            bars = ax.bar(features, scores, color=color, alpha=0.85)
            ax.set_ylabel(f'{abbr} {title_suffix}', fontsize=8, labelpad=3)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            if len(features) > 12:
                feature_fontsize = 5
            else:
                feature_fontsize = 8
            plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=feature_fontsize)
            ylim = ax.get_ylim()
            y_mid = (ylim[0] + ylim[1]) / 2
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, y_mid,
                        f'{score:.3f}', ha='center', va='center', fontsize=feature_fontsize, rotation=90, clip_on=True)

        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        if title:
            fig.suptitle('LOCO Feature Importance Across Metrics', fontsize=18, y=0.995)
        plt.tight_layout(h_pad=5,w_pad=5)
        plt.subplots_adjust(top=0.95,left=0.05)
        plt.show()