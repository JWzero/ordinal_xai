import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from interpretation.base_interpretation import BaseInterpretation
from utils import pdp_modified
import matplotlib.cm as cm
from matplotlib.patches import Patch

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
    
    def explain(self, observation_idx=None, feature_subset=None):
        """
        Generate Individual Conditional Expectation Plots for probabilities.
        
        Parameters:
        - observation_idx: (Optional) Index of specific instance to highlight.
        - feature_subset: (Optional) List of feature names or indices to plot.
        """
        if feature_subset is None:
            feature_subset = self.X.columns.tolist()
        else:
            feature_subset = [self.X.columns[i] if isinstance(i, int) else i for i in feature_subset]
        
        num_features = len(feature_subset)
        num_cols = min(num_features, 4)  # Max 4 plots per row
        num_rows = int(np.ceil(num_features / num_cols))  # Compute required rows

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 4 * num_rows))
        if num_features == 1:
            axes = np.array([[axes]])
        elif num_features <= num_cols:
            axes = axes.reshape(1, -1)
        
        self.model.fit(self.X, self.y)  # Ensure model is fitted

        results = {}

        for idx, feature in enumerate(feature_subset):
            row, col = divmod(idx, num_cols)
            ax = axes[row, col]
            
            feature_idx = [self.X.columns.get_loc(feature)]
            ice_result = pdp_modified.partial_dependence(
                self.model, self.X, features=feature_idx, 
                response_method="predict_proba", kind="both"
            )
            results[feature] = ice_result

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
                legend_elements = []
                
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
                    
                    # Add to legend
                    legend_elements.append(Patch(facecolor=colors[rank], alpha=0.7, 
                                               label=f'Instance - Rank {rank}'))
                    legend_elements.append(Patch(facecolor=colors[rank], alpha=0.15, 
                                               label=f'Average - Rank {rank}', hatch='//'))
                
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
                
                # Add text annotation for probabilities at the original value
                probs_at_orig = instance_probs[:, closest_idx]
                prob_str = ", ".join([f"Rank {i}: {p:.2f}" for i, p in enumerate(probs_at_orig)])
                ax.annotate(f"Probabilities at {feature}={original_value}:\n{prob_str}", 
                           xy=(original_value, 0.5), xytext=(10, 0),
                           textcoords="offset points", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                           ha='left', va='center', zorder=6)
                
                # Set y-axis limits for the plot
                ax.set_ylim(0, 1)
                
                # Add custom legend
                ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
                
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
                           label=f'Average Rank {rank}')
                
                # Add a legend
                ax.legend()
            
            ax.set_xlabel(feature, fontsize=12, labelpad=10)
            ax.set_ylabel("Probability", fontsize=12, labelpad=10)
            ax.set_title(f"ICE Probability Plot for {feature}", fontsize=14, pad=15)
            ax.grid(alpha=0.3)

        # Hide empty subplots
        for idx in range(num_features, num_rows * num_cols):
            row, col = divmod(idx, num_cols)
            fig.delaxes(axes[row, col])

        plt.tight_layout(pad=3.0)  # Increase padding to avoid overlap
        plt.show()
        
        print(f"Generated ICE Plots for features: {feature_subset}")
        return results 