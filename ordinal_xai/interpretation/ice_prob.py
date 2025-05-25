"""
Individual Conditional Expectation (ICE) Plot implementation for probability visualization in ordinal regression.

This module implements ICE plots specifically designed for visualizing probability distributions
in ordinal regression models. It extends the standard ICE plot to show how class probabilities
change as feature values change, using stacked area plots to represent the probability
distribution across ordinal classes.

Key Features:
- Stacked area plots to visualize probability distributions
- Support for both individual instance analysis and average behavior (PDP)
- Automatic handling of categorical and numerical features
- Detailed probability annotations at original feature values
- Color-coded visualization of ordinal class probabilities

The implementation is particularly useful for:
- Understanding how features affect the probability distribution across ordinal classes
- Analyzing the relationship between feature values and class probabilities
- Comparing individual instance behavior with average behavior
- Visualizing the uncertainty in ordinal predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from interpretation.base_interpretation import BaseInterpretation
from utils import pdp_modified
import matplotlib.cm as cm
from matplotlib.patches import Patch
import textwrap

class ICEProb(BaseInterpretation):
    """
    Individual Conditional Expectation (ICE) Plot interpretation method for probabilities.
    
    This class implements ICE plots specifically designed for visualizing probability
    distributions in ordinal regression models. It uses stacked area plots to show how
    the probability distribution across ordinal classes changes as feature values change.
    
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
        Initialize the ICE Plot interpretation method for probabilities.
        
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
        Generate Individual Conditional Expectation Plots for probabilities.
        
        This method computes and optionally visualizes how the model's probability
        distribution changes as feature values change. It uses stacked area plots
        to show the probability distribution across ordinal classes.
        
        Parameters
        ----------
        observation_idx : int, optional
            Index of specific instance to highlight in the plot. If provided,
            only this instance's probability distribution will be shown along with
            the average (PDP) distribution.
        feature_subset : list, optional
            List of feature names or indices to plot. If None, all features are used.
        plot : bool, default=False
            Whether to create visualizations of the ICE plots.
            
        Returns
        -------
        dict
            Dictionary containing ICE results for each feature:
            - 'grid_values': Feature values used for prediction
            - 'average': Average probability predictions (PDP) for each class
            - 'individual': Individual probability predictions for each instance and class
            
        Notes
        -----
        - Uses stacked area plots to visualize probability distributions
        - Shows both individual instance probabilities and average probabilities
        - Includes probability annotations at original feature values
        - Uses a viridis colormap for different ordinal classes
        - Automatically handles both categorical and numerical features
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
                response_method="predict_proba", kind="both"
            )
            results[feature] = ice_result

        # Create visualizations if requested
        if plot:
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7 * num_cols, 5 * num_rows))
            if num_features == 1:
                axes = np.array([[axes]])
            elif num_features <= num_cols:
                axes = axes.reshape(1, -1)
            
            legend_elements = None  # Initialize legend elements

            for idx, feature in enumerate(feature_subset):
                row, col = divmod(idx, num_cols)
                ax = axes[row, col]
                
                ice_result = results[feature]
                x_values = ice_result['grid_values'][0]
                averaged_predictions = ice_result['average']  # Shape: (n_classes, n_grid_points)
                individual_predictions = ice_result['individual']  # Shape: (n_classes, n_instances, n_grid_points)
                num_ranks = averaged_predictions.shape[0]
                
                # Create colormap for different ranks
                cmap = cm.get_cmap('viridis', num_ranks)
                colors = [cmap(i) for i in range(num_ranks)]
                
                # Plot curves based on whether observation_idx is specified
                if observation_idx is not None:
                    # Create a stacked area plot for the specified instance
                    instance_probs = individual_predictions[:, observation_idx, :]
                    
                    # Create legend elements if not already created
                    if legend_elements is None:
                        legend_elements = [
                            Patch(facecolor=colors[rank], alpha=0.7, label=f'R{rank}') for rank in range(num_ranks)
                        ] + [
                            Patch(facecolor=colors[rank], alpha=0.15, label=f'R{rank} avg', hatch='//') for rank in range(num_ranks)
                        ]
                    
                    # Plot stacked areas for instance
                    ax.stackplot(x_values, instance_probs, colors=colors, alpha=0.7, zorder=2)
                    
                    # Plot stacked areas for average probabilities
                    baseline = np.zeros(len(x_values), dtype=np.float64)
                    
                    for rank in range(num_ranks):
                        # Create stacked area for this rank
                        ax.fill_between(x_values, baseline, baseline + averaged_predictions[rank], 
                                       color=colors[rank], alpha=0.15, zorder=1)
                        
                        # Add dashed line at the top edge of this rank's area
                        ax.plot(x_values, baseline + averaged_predictions[rank], 
                               color=colors[rank], linestyle='--', linewidth=1.5, zorder=3)
                        
                        # Update baseline for next rank
                        baseline = baseline + averaged_predictions[rank]
                    
                    # Add original value marker and vertical line
                    original_value = self.X.iloc[observation_idx][feature]
                    
                    # Add vertical line at original feature value
                    ymin, ymax = 0, 1  # Probability bounds
                    ax.vlines(x=original_value, ymin=ymin, ymax=ymax, 
                            colors='black', linestyles='dashed', linewidth=1.5, zorder=5)
                    
                    # Find probabilities at original value
                    if isinstance(original_value, (int, float)):
                        closest_idx = np.argmin(np.abs(x_values - original_value))
                    else:
                        closest_idx = np.where(x_values == original_value)[0][0]
                    
                    # Get and display probabilities at original value
                    probs_at_orig = instance_probs[:, closest_idx]
                    prob_str = ", ".join([f"R{i}: {p:.2f}" for i, p in enumerate(probs_at_orig)])
                    
                    # Add probability annotation
                    caption = f"Probabilities at {feature}={original_value}: {prob_str}"
                    wrapped_caption = "\n".join(textwrap.wrap(caption, width=60))
                    ax.text(
                        0.5, 1.02, wrapped_caption,
                        ha='center', va='bottom',
                        transform=ax.transAxes,
                        fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
                    )
                    
                    # Set y-axis limits
                    ax.set_ylim(0, 1)
                    
                else:
                    # Plot all instances and average (standard line plot)
                    for i in range(len(self.X)):
                        for rank in range(num_ranks):
                            ax.plot(x_values, individual_predictions[rank, i, :], 
                                   color=colors[rank], alpha=0.1, linewidth=0.5)
                    
                    # Plot the average curves
                    for rank in range(num_ranks):
                        ax.plot(x_values, averaged_predictions[rank], 
                               color=colors[rank], linestyle='--', linewidth=2, 
                               label=f'R{rank}')
                
                # Create simple legend with one entry per rank color
                if legend_elements is None:
                    legend_elements = [
                        Patch(facecolor=colors[rank], label=f'Rank {rank}') for rank in range(num_ranks)
                    ]
                
                ax.set_xlabel(feature, fontsize=12, labelpad=6)
                ax.set_ylabel("Probability", fontsize=8, labelpad=4)
                ax.grid(alpha=0.3)

            # Add shared legend
            if legend_elements is not None:
                fig.legend(
                    handles=legend_elements,
                    loc='lower right',
                    fontsize=12,
                    ncol=min(num_ranks, 3),  # Adjust number of columns based on number of ranks
                    handletextpad=0.5,
                    columnspacing=0.5,
                    frameon=True,
                    borderpad=0.5,
                    labelspacing=0.5,
                    handlelength=1.5,
                    borderaxespad=0.5,
                    fancybox=True
                )

            # Hide empty subplots
            for idx in range(num_features, num_rows * num_cols):
                row, col = divmod(idx, num_cols)
                fig.delaxes(axes[row, col])

            plt.tight_layout(pad=3.0, h_pad=8.0)
            plt.subplots_adjust(bottom=0.25)
            plt.show()
        
        print(f"Generated ICE Plots for features: {feature_subset}")
        return results 