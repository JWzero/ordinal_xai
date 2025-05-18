import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interpretation.base_interpretation import BaseInterpretation
from sklearn.inspection import permutation_importance
from utils.evaluation_metrics import (
    adjacent_accuracy, mze, mae, mse, weighted_kappa, cem, 
    spearman_correlation, kendall_tau,
    ranked_probability_score, ordinal_weighted_ce,
    evaluate_ordinal_model
)
from interpretation.LOCO import LOCO

class PermutationImportance(BaseInterpretation):
    """Permutation Importance interpretation method for feature importance."""
    
    def __init__(self, model, X: pd.DataFrame, y: np.ndarray = None, metrics=None, n_repeats=10, random_state=42):
        """
        Initialize the Permutation Importance interpretation method.
        
        Parameters:
        - model: The trained ordinal regression model.
        - X: DataFrame containing the dataset used for interpretation.
        - y: (Optional) Series containing target labels.
        - metrics: (Optional) List of metrics to use for feature importance calculation.
                  If None, all available metrics will be used.
        - n_repeats: Number of times to permute each feature.
        - random_state: Random seed for reproducibility.
        """
        super().__init__(model, X, y)
        
        # Validate input data
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        # Define available metrics
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
        
        # Set metrics to use
        if metrics is None:
            self.metrics = list(self.available_metrics.keys())
        else:
            # Validate metrics
            invalid_metrics = [m for m in metrics if m not in self.available_metrics]
            if invalid_metrics:
                raise ValueError(f"Invalid metrics: {invalid_metrics}. Available metrics: {list(self.available_metrics.keys())}")
            self.metrics = metrics
        
        self.n_repeats = n_repeats
        self.random_state = random_state
        
        # Store the original model's performance metrics
        if self.y is not None:
            try:
                self.original_predictions = self.model.predict(self.X)
                self.original_results = evaluate_ordinal_model(self.y, self.original_predictions)
            except Exception as e:
                raise ValueError(f"Failed to get original model predictions: {str(e)}")
    
    def _create_scoring_func(self, metric_name, metric_func):
        """Create a scoring function for a specific metric."""
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
    
    def explain(self, observation_idx=None, feature_subset=None, plot=False):
        """
        Generate Permutation Importance explanations.
        
        Parameters:
        - observation_idx: Ignored (Permutation Importance is a global method).
        - feature_subset: (Optional) List of feature names to consider.
        - plot: (Optional) Whether to create visualizations. Default is False.
        
        Returns:
        - Dictionary containing feature importance scores for each metric.
        """
        if not self.model.is_fitted_:
            self.model.fit(self.X, self.y) # Ensure model is fitted
        
        # Determine which features to analyze
        if feature_subset is None:
            feature_subset = self.X.columns.tolist()
        else:
            feature_subset = [self.X.columns[i] if isinstance(i, int) else i for i in feature_subset]
        
        # Store results for each metric
        results = {}
        
        # Calculate permutation importance for each metric
        for metric_name in self.metrics:
            metric_func = self.available_metrics[metric_name]
            scoring_func = self._create_scoring_func(metric_name, metric_func)
            
            # Calculate permutation importance
            result = permutation_importance(
                self.model, self.X, self.y,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
                scoring=scoring_func
            )
            
            # Store results
            results[metric_name] = {
                'importances_mean': result.importances_mean,
                'importances_std': result.importances_std,
                'importances': result.importances
            }
        
        # Plot if requested
        if plot:
            self._plot_feature_importance(results)
        
        return results
    
    def _plot_feature_importance(self, results, metrics=None):
        """
        Plot feature importance scores for each metric using the LOCO plot style,
        but adapted for the permutation importance results structure.
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
        n_cols = 2 if n_metrics > 1 else 1
        n_rows = math.ceil(n_metrics / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, metric in enumerate(metrics):
            metric_result = results[metric]
            features = list(self.X.columns)
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
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
            # Add value labels at the same height in the middle of the subplot
            ylim = ax.get_ylim()
            y_mid = (ylim[0] + ylim[1]) / 2
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, y_mid,
                        f'{mean:.3f}', ha='center', va='center', fontsize=8, rotation=90, clip_on=True)
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        fig.suptitle('Permutation Feature Importance Across Metrics', fontsize=18, y=0.995)
        plt.tight_layout(h_pad=8)
        plt.show() 