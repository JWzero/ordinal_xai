"""
Individual Conditional Expectation (ICE) Plot implementation for ordinal regression models.

This module implements ICE plots, a model-agnostic interpretation method that shows how
a model's prediction changes as a feature value changes, while keeping other features
constant. For ordinal regression, it shows how the predicted ordinal classes changes across 
individual observations with feature variations.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from interpretation.base_interpretation import BaseInterpretation
from utils import pdp_modified

class ICE(BaseInterpretation):
    """
    Individual Conditional Expectation (ICE) Plot interpretation method.
    
    ICE plots show how a model's prediction changes as a feature value changes,
    while keeping other features constant. For ordinal regression, it shows how
    the probability distribution across ordinal classes changes with feature variations.
    
    Parameters
    ----------
    model : object
        The trained ordinal regression model. Must implement predict_proba method.
    X : pd.DataFrame
        Dataset used for interpretation. Should contain the same features used
        during model training.
    y : pd.Series, optional
        Target labels. Not required for interpretation but useful for reference.
        
    Attributes
    ----------
    model : object
        The trained ordinal regression model
    X : pd.DataFrame
        Dataset used for interpretation
    y : pd.Series
        Target labels (if provided)
    """
    
    def __init__(self, model, X, y=None):
        """
        Initialize the ICE Plot interpretation method.
        
        Parameters
        ----------
        model : object
            The trained ordinal regression model
        X : pd.DataFrame
            Dataset used for interpretation
        y : pd.Series, optional
            Target labels
        """
        super().__init__(model, X, y)
    
    def explain(self, observation_idx=None, feature_subset=None, plot=False):
        """
        Generate Individual Conditional Expectation Plots.
        
        This method computes and optionally visualizes how the model's predictions
        change as feature values change. For ordinal regression, it shows how the
        probability distribution across classes changes with feature variations.
        
        Parameters
        ----------
        observation_idx : int, optional
            Index of specific instance to highlight in the plot. If provided,
            only this instance's ICE curves will be shown along with the average (PDP).
        feature_subset : list, optional
            List of feature names or indices to plot. If None, all features are used.
        plot : bool, default=False
            Whether to create visualizations of the ICE plots.
            
        Returns
        -------
        dict
            Dictionary containing ICE results for each feature:
            - 'grid_values': Feature values used for prediction
            - 'average': Average predictions (PDP) for each class
            - 'individual': Individual predictions for each instance and class
            
        Notes
        -----
        - For ordinal regression, the plots show probability changes for each class
        - The average curve (PDP) shows the overall effect of the feature
        - Individual curves show instance-specific effects
        - For categorical features, exact feature values are used
        - For numerical features, a grid of values is used
        """
        if feature_subset is None:
            feature_subset = self.X.columns.tolist()
        else:
            feature_subset = [self.X.columns[i] if isinstance(i, int) else i for i in feature_subset]
        
        num_features = len(feature_subset)
        num_cols = min(num_features, 4)  # Max 4 plots per row
        num_rows = int(np.ceil(num_features / num_cols))  # Compute required rows

        if not self.model.is_fitted_:
            self.model.fit(self.X, self.y) # Ensure model is fitted
            
        results = {}

        # Compute ICE curves for each feature
        for idx, feature in enumerate(feature_subset):
            feature_idx = [self.X.columns.get_loc(feature)]
            ice_result = pdp_modified.partial_dependence(
                self.model, self.X, features=feature_idx, 
                kind="both"
            )
            results[feature] = ice_result

        # Create visualizations if requested
        if plot:
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 4 * num_rows))
            if num_features == 1:
                axes = np.array([[axes]])
            elif num_features <= num_cols:
                axes = axes.reshape(1, -1)
            
            for idx, feature in enumerate(feature_subset):
                row, col = divmod(idx, num_cols)
                ax = axes[row, col]
                
                ice_result = results[feature]
                x_values = ice_result['grid_values'][0]
                averaged_predictions = ice_result['average']  # Shape: (n_classes, n_grid_points)
                individual_predictions = ice_result['individual']  # Shape: (n_classes, n_instances, n_grid_points)
                num_ranks = averaged_predictions.shape[0]
                
                # Plot curves based on whether observation_idx is specified
                if observation_idx is not None:
                    # Only plot the specified instance and average
                    for rank in range(num_ranks):
                        # Plot the specified instance
                        ax.plot(x_values, individual_predictions[rank, observation_idx, :], 
                               color=f'C{rank}', linewidth=2, 
                               label=f'Instance {observation_idx} Rank')
                        # Plot the average
                        ax.plot(x_values, averaged_predictions[rank], 
                               color=f'C{rank}', linestyle='--', linewidth=2, 
                               label=f'Average Rank (PDP)')
                    
                    # Add marker for original feature value
                    original_value = self.X.iloc[observation_idx][feature]
                    for rank in range(num_ranks):
                        # Find the closest grid point to the original value
                        if isinstance(original_value, (int, float)):
                            closest_idx = np.argmin(np.abs(x_values - original_value))
                        else:
                            # For categorical features, find the exact match
                            closest_idx = np.where(x_values == original_value)[0][0]
                        ax.scatter(original_value, individual_predictions[rank, observation_idx, closest_idx],
                                 color=f'C{rank}', s=100, zorder=5)
                    
                    # Add vertical line at original feature value
                    ymin, ymax = ax.get_ylim()
                    ax.vlines(x=original_value, ymin=ymin, ymax=ymax, 
                             colors='black', linestyles='dashed', linewidth=1.5, zorder=1)
                else:
                    # Plot all instances and average
                    for i in range(len(self.X)):
                        for rank in range(num_ranks):
                            ax.plot(x_values, individual_predictions[rank, i, :], 
                                   color=f'C{rank}', alpha=0.1, linewidth=0.5)
                    
                    # Plot the average curves on top
                    for rank in range(num_ranks):
                        ax.plot(x_values, averaged_predictions[rank], 
                               color=f'C{rank}', linestyle='--', linewidth=2, 
                               label=f'Average Rank (PDP)')
                
                ax.set_xlabel(feature, fontsize=12, labelpad=6)
                ax.set_ylabel("Prediction", fontsize=12, labelpad=6)
                ax.set_title(f"ICE Plot for {feature}", fontsize=14, pad=15)
                ax.grid()
                ax.legend()

            # Hide empty subplots
            for idx in range(num_features, num_rows * num_cols):
                row, col = divmod(idx, num_cols)
                fig.delaxes(axes[row, col])

            plt.tight_layout(pad=3.0)  # Increase padding to avoid overlap
            plt.show()
        
        print(f"Generated ICE Plots for features: {feature_subset}")
        return results 