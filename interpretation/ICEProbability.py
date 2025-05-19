import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from interpretation.base_interpretation import BaseInterpretation
from utils import pdp_modified
import matplotlib.cm as cm
from matplotlib.patches import Patch
import textwrap

class ICEProbability(BaseInterpretation):
    """Individual Conditional Expectation (ICE) Plot interpretation method for probabilities."""
    
    def __init__(self, model, X, y=None):
        """
        Initialize the ICE Plot interpretation method.
        
        Parameters:
        - model: The trained ordinal regression model.
        - X: DataFrame containing the dataset used for interpretation.
        - y: (Optional) Series containing target labels.
        """
        super().__init__(model, X, y)
    
    def explain(self, observation_idx=None, feature_subset=None, plot=False):
        """
        Generate Individual Conditional Expectation Plots for probabilities.
        
        Parameters:
        - observation_idx: (Optional) Index of specific instance to highlight.
        - feature_subset: (Optional) List of feature names or indices to plot.
        - plot: (Optional) Whether to create visualizations. Default is False.
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
            
            legend_elements = None  # <-- Add this before the plotting loop

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
                    # Create a stacked area plot for the specified instance
                    instance_probs = individual_predictions[:, observation_idx, :]
                    
                    # We need to stack the probabilities from bottom to top
                    # First, create a colormap for the different ranks
                    cmap = cm.get_cmap('viridis', num_ranks)
                    colors = [cmap(i) for i in range(num_ranks)]
                    
                    # Create a custom legend
                    if legend_elements is None:
                        # Only create legend_elements once
                        legend_elements = [
                            Patch(facecolor=colors[rank], alpha=0.7, label=f'R{rank}') for rank in range(num_ranks)
                        ] + [
                            Patch(facecolor=colors[rank], alpha=0.15, label=f'R{rank} avg', hatch='//') for rank in range(num_ranks)
                        ]
                    
                    # Plot stacked areas for instance
                    ax.stackplot(x_values, instance_probs, colors=colors, alpha=0.7, zorder=2)
                    
                    # Plot the stacked area for the average probabilities with dashed edges between areas
                    # First, create an array to hold the baseline for each layer
                    baseline = np.zeros(len(x_values), dtype=np.float64)
                    
                    for rank in range(num_ranks):
                        # Create stacked area for this rank
                        ax.fill_between(x_values, baseline, baseline + averaged_predictions[rank], 
                                       color=colors[rank], alpha=0.15, zorder=1)
                        
                        # Add dashed line at the top edge of this rank's area
                        ax.plot(x_values, baseline + averaged_predictions[rank], 
                               color=colors[rank], linestyle='--', linewidth=1.5, zorder=3)
                        
                        # Update baseline for next rank - explicitly convert if needed
                        baseline = baseline + averaged_predictions[rank]
                    
                    # Plot original value marker and vertical line
                    original_value = self.X.iloc[observation_idx][feature]
                    
                    # Add vertical line at original feature value
                    ymin, ymax = 0, 1  # Probability bounds
                    ax.vlines(x=original_value, ymin=ymin, ymax=ymax, 
                            colors='black', linestyles='dashed', linewidth=1.5, zorder=5)
                    
                    # Find the probabilities at the original value
                    if isinstance(original_value, (int, float)):
                        closest_idx = np.argmin(np.abs(x_values - original_value))
                    else:
                        # For categorical features, find the exact match
                        closest_idx = np.where(x_values == original_value)[0][0]
                    
                    # Get probabilities at original value
                    probs_at_orig = instance_probs[:, closest_idx]
                    prob_str = ", ".join([f"R{i}: {p:.2f}" for i, p in enumerate(probs_at_orig)])
                    
                    # Insert line breaks in subcaption if too long
                    caption = f"Probabilities at {feature}={original_value}: {prob_str}"
                    wrapped_caption = "\n".join(textwrap.wrap(caption, width=60))
                    ax.text(
                        0.5, 1.02, wrapped_caption,  # Moved to top position
                        ha='center', va='bottom',
                        transform=ax.transAxes,
                        fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
                    )
                    
                    # Set y-axis limits for the plot
                    ax.set_ylim(0, 1)
                    
                    # Adjust layout to make room for subcaption
                    plt.subplots_adjust(bottom=0.2)
                    
                else:
                    # Plot all instances and average (standard line plot)
                    for i in range(len(self.X)):
                        for rank in range(num_ranks):
                            ax.plot(x_values, individual_predictions[rank, i, :], 
                                   color=f'C{rank}', alpha=0.1, linewidth=0.5)
                    
                    # Plot the average curves on top
                    for rank in range(num_ranks):
                        ax.plot(x_values, averaged_predictions[rank], 
                               color=f'C{rank}', linestyle='--', linewidth=2, 
                               label=f'R{rank}')
                    
                ax.set_xlabel(feature, fontsize=12, labelpad=6)
                ax.set_ylabel("Probability", fontsize=8, labelpad=4)
                ax.grid(alpha=0.3)

            # After the plotting loop, add the shared legend to the figure
            if legend_elements is not None:
                fig.legend(
                    handles=legend_elements,
                    loc='lower right',           # Move to bottom right
                    fontsize=12,                 # Increase font size
                    ncol=3,
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

            plt.tight_layout(pad=3.0, h_pad=8.0)  # Increased vertical padding with h_pad
            plt.subplots_adjust(bottom=0.25)  # Increased bottom margin for subcaptions
            plt.show()
        
        print(f"Generated ICE Plots for features: {feature_subset}")
        return results 