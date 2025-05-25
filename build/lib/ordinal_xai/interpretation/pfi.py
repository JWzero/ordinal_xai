"""
Permutation Feature Importance (PFI) interpretation method for ordinal regression models.

This module implements the Permutation Feature Importance method for analyzing feature
importance in ordinal regression models. PFI works by measuring how model performance
changes when feature values are randomly permuted, breaking the relationship between
the feature and the target variable.

Key Features:
- Global feature importance analysis through feature permutation
- Support for multiple evaluation metrics
- Robust importance estimation through multiple permutations
- Comprehensive visualization of feature importance
- Support for both classification and probability-based metrics

The implementation is particularly useful for:
- Understanding feature importance in ordinal regression models
- Identifying key features that influence model predictions
- Analyzing feature importance across different evaluation metrics
- Comparing feature importance between different models
"""

from typing import Optional, List, Dict, Union, Callable, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from .base_interpretation import BaseInterpretation
from sklearn.inspection import permutation_importance
from ..utils.evaluation_metrics import (
    mze, mae, mse, adjacent_accuracy, weighted_kappa,
    cem, spearman_correlation, kendall_tau,
    ranked_probability_score, ordinal_weighted_ce,
    evaluate_ordinal_model
)

class PFI(BaseInterpretation):
    """
    Permutation Feature Importance (PFI) interpretation method for feature importance.
    
    This class implements the PFI method for analyzing feature importance in ordinal regression
    models. It measures the impact of each feature by evaluating how model performance changes
    when the feature values are randomly permuted.
    
    Parameters
    ----------
    model : object
        The trained ordinal regression model. Must implement fit and predict methods.
    X : pd.DataFrame
        Dataset used for interpretation. Should contain the same features used
        during model training.
    y : np.ndarray, optional
        Target labels. Required for evaluation but optional for initialization.
    n_repeats : int, default=5
        Number of times to permute each feature. Higher values provide more
        robust importance estimates but increase computation time.
    random_state : int, default=42
        Random seed for reproducibility of permutations.
        
    Attributes
    ----------
    available_metrics : dict
        Dictionary of available evaluation metrics and their corresponding functions.
    n_repeats : int
        Number of permutation repeats.
    random_state : int
        Random state for reproducibility.
    original_predictions : array-like
        Predictions from the original model.
    original_results : dict
        Evaluation results from the original model.
        
    Raises
    ------
    ValueError
        If X is not a pandas DataFrame or if model predictions fail.
    """
    
    def __init__(self, model, X: pd.DataFrame, y: np.ndarray = None, n_repeats=5, random_state=42):
        """
        Initialize the Permutation Feature Importance interpretation method.
        
        Parameters
        ----------
        model : object
            The trained ordinal regression model
        X : pd.DataFrame
            Dataset used for interpretation
        y : np.ndarray, optional
            Target labels
        n_repeats : int, default=5
            Number of times to permute each feature
        random_state : int, default=42
            Random seed for reproducibility
            
        Raises
        ------
        ValueError
            If X is not a pandas DataFrame or if model predictions fail
        """
        super().__init__(model, X, y)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        self.available_metrics = {
            'mze': mze,
            'mae': mae,
            'mse': mse,
            'adjacent_accuracy': adjacent_accuracy,
            'quadratic_weighted_kappa': lambda yt, yp: weighted_kappa(yt, yp, weights='quadratic'),
            'linear_weighted_kappa': lambda yt, yp: weighted_kappa(yt, yp, weights='linear'),
            'cem': cem,
            'spearman_correlation': spearman_correlation,
            'kendall_tau': kendall_tau,
            'ranked_probability_score': ranked_probability_score,
            'ordinal_weighted_ce_linear': lambda yt, yp: ordinal_weighted_ce(yt, yp, alpha=1),
            'ordinal_weighted_ce_quadratic': lambda yt, yp: ordinal_weighted_ce(yt, yp, alpha=2),
        }
        self.n_repeats = n_repeats
        self.random_state = random_state
        if self.y is not None:
            try:
                self.original_predictions = self.model.predict(self.X)
                self.original_results = evaluate_ordinal_model(self.y, self.original_predictions)
            except Exception as e:
                raise ValueError(f"Failed to get original model predictions: {str(e)}")
    
    def _create_scoring_func(self, metric_name, metric_func):
        """
        Create a scoring function for a specific metric.
        
        This method creates a scoring function that can be used with sklearn's
        permutation_importance. The function handles both class predictions and
        probability predictions, and ensures proper sign for different metric types.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to create scoring function for
        metric_func : callable
            The metric function to use for scoring
            
        Returns
        -------
        callable
            A scoring function that takes (estimator, X, y) as arguments and
            returns a score
            
        Raises
        ------
        ValueError
            If score calculation fails for the metric
        """
        def scoring_func(estimator, X, y):
            try:
                # Get class predictions
                y_pred = estimator.predict(X)
                
                # Try to get probability predictions if available
                try:
                    y_pred_proba = estimator.predict_proba(X)
                    # Use probability predictions for metrics that support them
                    if metric_name in ['ranked_probability_score', 'ordinal_weighted_ce_linear', 'ordinal_weighted_ce_quadratic']:
                        score = metric_func(y, y_pred_proba)
                    else:
                        # Use class predictions for other metrics
                        score = metric_func(y, y_pred)
                except (AttributeError, NotImplementedError):
                    # If predict_proba is not available, use class predictions
                    score = metric_func(y, y_pred)
                
                # For metrics where lower is better, return negative score
                if metric_name in ['mae', 'mse', 'mze', 'ranked_probability_score', 
                                 'ordinal_weighted_ce_linear', 'ordinal_weighted_ce_quadratic']:
                    return -score
                return score
            except Exception as e:
                raise ValueError(f"Failed to calculate score for metric {metric_name}: {str(e)}")
        return scoring_func
    
    def explain(self, observation_idx=None, feature_subset=None, plot=False, metrics=None):
        """
        Generate Permutation Feature Importance explanations.
        
        This method computes feature importance by measuring how model performance
        changes when feature values are randomly permuted. The importance score
        represents the average change in model performance across multiple
        permutations.
        
        Parameters
        ----------
        observation_idx : int, optional
            Ignored (Permutation Importance is a global method)
        feature_subset : list, optional
            List of feature names or indices to consider (for permutation and output only)
        plot : bool, default=False
            Whether to create visualizations
        metrics : list, optional
            List of metrics to use for feature importance calculation
            
        Returns
        -------
        dict
            Dictionary containing feature importance scores for each metric:
            - 'features': List of feature names
            - 'importances_mean': Mean importance scores
            - 'importances_std': Standard deviation of importance scores
            - 'importances': Raw importance scores for each permutation
            
        Raises
        ------
        ValueError
            If invalid metrics are specified
        """
        if not self.model.is_fitted_:
            self.model.fit(self.X, self.y) # Ensure model is fitted
        all_features = self.X.columns.tolist()
        if feature_subset is None:
            permute_features = all_features
            feature_idxs = list(range(len(all_features)))
        else:
            if all(isinstance(f, int) for f in feature_subset):
                permute_features = [all_features[i] for i in feature_subset]
                feature_idxs = feature_subset
            else:
                permute_features = feature_subset
                feature_idxs = [all_features.index(f) for f in permute_features]
        # Set metrics to use
        if metrics is None:
            metrics_to_use = list(self.available_metrics.keys())
        else:
            invalid_metrics = [m for m in metrics if m not in self.available_metrics]
            if invalid_metrics:
                raise ValueError(f"Invalid metrics: {invalid_metrics}. Available metrics: {list(self.available_metrics.keys())}")
            metrics_to_use = metrics
        results = {}
        for metric_name in metrics_to_use:
            metric_func = self.available_metrics[metric_name]
            scoring_func = self._create_scoring_func(metric_name, metric_func)
            result = permutation_importance(
                self.model, self.X, self.y,
                n_repeats=self.n_repeats,
                n_jobs=-1, # Use all available CPU cores
                random_state=self.random_state,
                scoring=scoring_func
            )
            importances_mean = result.importances_mean[feature_idxs]
            importances_std = result.importances_std[feature_idxs]
            importances = result.importances[feature_idxs]
            results[metric_name] = {
                'features': permute_features,
                'importances_mean': importances_mean,
                'importances_std': importances_std,
                'importances': importances
            }
        if plot:
            self._plot_feature_importance(results)
        return results
    
    def _plot_feature_importance(self, results, metrics=None):
        """
        Plot feature importance scores for each metric.
        
        This method creates bar plots showing feature importance scores for each metric.
        The plots are arranged in a grid, with one subplot per metric. Features are
        sorted by importance score, and error bars show the standard deviation across
        permutations.
        
        Parameters
        ----------
        results : dict
            Dictionary containing all results from the PFI analysis
        metrics : list, optional
            List of metrics to plot (defaults to all in results)
            
        Notes
        -----
        - Features are sorted by importance score
        - Error metrics are shown in red with "Increase" in the title
        - Other metrics are shown in green with "Drop" in the title
        - Feature names are rotated 90 degrees for better readability
        - Score values are displayed in the middle of each bar
        """
        import math

        metric_abbr = {
            'adjacent_accuracy': 'AA',
            'linear_weighted_kappa': 'LWK',
            'quadratic_weighted_kappa': 'QWK',
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
            metrics = list(results.keys())
        n_metrics = len(metrics)
        n_cols = min(4, n_metrics) if n_metrics > 1 else 1
        n_rows = math.ceil(n_metrics / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, metric in enumerate(metrics):
            metric_result = results[metric]
            features = metric_result['features']
            means = metric_result['importances_mean']
            stds = metric_result['importances_std']

            abbr = metric_abbr.get(metric, metric)
            color = 'red' if metric in ['mae', 'mse', 'mze', 'ranked_probability_score',
                                       'ordinal_weighted_ce_linear', 'ordinal_weighted_ce_quadratic'] else 'green'
            title_suffix = "Increase" if color == 'red' else "Drop"

            ax = axes[i]
            bars = ax.bar(features, means, color=color, alpha=0.85)
            ax.set_ylabel(f'{abbr} {title_suffix}', fontsize=8, labelpad=10)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            if len(features) > 10:
                feature_fontsize = 5
            else:
                feature_fontsize = 8
            plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=feature_fontsize)
            ylim = ax.get_ylim()
            y_mid = (ylim[0] + ylim[1]) / 2
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, y_mid,
                        f'{mean:.3f}', ha='center', va='center', fontsize=feature_fontsize, rotation=90, clip_on=True)
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        fig.suptitle('Permutation Feature Importance Across Metrics', fontsize=18, y=0.995)
        plt.tight_layout(h_pad=5,w_pad=5)
        plt.subplots_adjust(top=0.95,left=0.05)
        plt.show() 