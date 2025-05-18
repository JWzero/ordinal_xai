import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from interpretation.base_interpretation import BaseInterpretation
from utils import pdp_modified

class ProbabilityPDP(BaseInterpretation):
    """Partial Dependence Plot (PDP) interpretation method for ordinal regression models."""
    
    def __init__(self, model, X, y=None):
        """
        Initialize the PDP interpretation method.
        
        Parameters:
        - model: The trained ordinal regression model.
        - X: DataFrame containing the dataset used for interpretation.
        - y: (Optional) Series containing target labels.
        """
        super().__init__(model, X, y)
    

    def explain(self, observation_idx=None, feature_subset=None, plot=False):
        """
        Generate Partial Dependence Plots as stacked area plots.
        
        Parameters:
        - observation_idx: Ignored (PDP is a global method).
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
            pdp_result = pdp_modified.partial_dependence(self.model, self.X, features=feature_idx, response_method="predict_proba")
            results[feature] = pdp_result

        # Create visualizations if requested
        if plot:
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 4 * num_rows))
            axes = np.array(axes).reshape(num_rows, num_cols)  # Ensure it's a 2D array
            
            for idx, feature in enumerate(feature_subset):
                row, col = divmod(idx, num_cols)
                ax = axes[row, col]
                
                pdp_result = results[feature]
                num_ranks = pdp_result['average'].shape[0]
                x_values = pdp_result['grid_values'][0]
                probabilities = pdp_result['average']
                
                # Use stackplot for stacked probability visualization
                ax.stackplot(x_values, probabilities, labels=[f"Rank {rank}" for rank in range(num_ranks)], alpha=0.7)
                
                ax.legend(loc="upper right")
                ax.set_xlabel(feature, fontsize=12, labelpad=10)
                ax.set_ylabel("Partial Dependence", fontsize=12, labelpad=10)
                ax.set_title(f"PDP for {feature}", fontsize=14, pad=15)
                ax.grid()

            # Hide empty subplots
            for idx in range(num_features, num_rows * num_cols):
                row, col = divmod(idx, num_cols)
                fig.delaxes(axes[row, col])

            plt.tight_layout(pad=3.0)  # Increase padding to avoid overlap
            plt.show()
        
        print(f"Generated PDPs for features: {feature_subset}")
        return results
