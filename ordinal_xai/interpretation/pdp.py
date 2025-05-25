"""
Partial Dependence Plot (PDP) interpretation method.

This module implements Partial Dependence Plots (PDPs) for interpreting ordinal regression models. It is based on the sklearn.inspection.partial_dependence function.
PDPs show how a feature affects predictions while accounting for the average effect of other features.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .base_interpretation import BaseInterpretation
from ..utils import pdp_modified
from typing import Optional, List, Union, Dict, Any

class PDP(BaseInterpretation):
    """
    Partial Dependence Plot (PDP) interpretation method.
    
    This class implements Partial Dependence Plots for interpreting ordinal regression models.
    PDPs show how a feature affects predictions while accounting for the average effect of other
    features. This helps understand the relationship between features and predictions in a
    model-agnostic way.
    
    Parameters
    ----------
    model : BaseOrdinalModel
        The trained ordinal regression model to interpret
    X : pd.DataFrame
        Dataset used for interpretation, shape (n_samples, n_features)
    y : Optional[pd.Series], default=None
        Target labels, shape (n_samples,)
        
    Attributes
    ----------
    model : BaseOrdinalModel
        The trained model to interpret
    X : pd.DataFrame
        Dataset used for interpretation
    y : Optional[pd.Series]
        Target labels
    feature_names_ : List[str]
        Names of features in the dataset
        
    Notes
    -----
    - PDPs are global interpretation methods that show the average effect of a feature
    - The method works by varying one feature while keeping others at their observed values
    - For each feature value, predictions are made and averaged across all samples
    - This helps understand the marginal effect of features on predictions
    """
    
    def __init__(self, model: Any, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Initialize the PDP interpretation method.
        
        Parameters
        ----------
        model : BaseOrdinalModel
            The trained ordinal regression model to interpret
        X : pd.DataFrame
            Dataset used for interpretation, shape (n_samples, n_features)
        y : Optional[pd.Series], default=None
            Target labels, shape (n_samples,)
            
        Raises
        ------
        ValueError
            If X is not a DataFrame or y is not a Series
        """
        super().__init__(model, X, y)
    
    def explain(
        self,
        observation_idx: Optional[int] = None,
        feature_subset: Optional[Union[List[str], List[int]]] = None,
        plot: bool = False,
        max_features_per_figure: int = 12
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate Partial Dependence Plots for the selected features.
        
        This method computes and optionally visualizes the partial dependence of the model's
        predictions on the selected features. For each feature, it shows how the average
        prediction changes as the feature value changes, while accounting for the effect of
        other features.
        
        Parameters
        ----------
        observation_idx : Optional[int], default=None
            Ignored (PDP is a global method)
        feature_subset : Optional[Union[List[str], List[int]]], default=None
            List of feature names or indices to plot. If None, all features are used
        plot : bool, default=False
            Whether to create visualizations
        max_features_per_figure : int, default=12
            Maximum number of features to display per figure (for large datasets)
            
        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            Dictionary containing PDP results for each feature:
            - 'grid_values': Feature values used for plotting
            - 'average': Average predictions at each feature value
            
        Raises
        ------
        ValueError
            If feature_subset contains invalid feature names or indices
        RuntimeError
            If model is not fitted and cannot be fitted automatically
        """
        # Handle feature subset
        if feature_subset is None:
            feature_subset = self.X.columns.tolist()
        else:
            feature_subset = [
                self.X.columns[i] if isinstance(i, int) else i 
                for i in feature_subset
            ]
        
        num_features = len(feature_subset)
        results = {}

        # Ensure model is fitted
        if not self.model.is_fitted_:
            if self.y is None:
                raise RuntimeError(
                    "Model is not fitted and target values (y) are not provided "
                    "for automatic fitting"
                )
            self.model.fit(self.X, self.y)

        # Compute PDPs for each feature
        for feature in feature_subset:
            feature_idx = [self.X.columns.get_loc(feature)]
            pdp_result = pdp_modified.partial_dependence(
                self.model,
                self.X,
                features=feature_idx
            )
            results[feature] = pdp_result

        # Create visualizations if requested
        if plot:
            self._plot_pdps(results, feature_subset, max_features_per_figure)
        
        print(f"Generated PDPs for features: {feature_subset}")
        return results
    
    def _plot_pdps(
        self,
        results: Dict[str, Dict[str, np.ndarray]],
        feature_subset: List[str],
        max_features_per_figure: int = 12
    ) -> None:
        """
        Create visualization of Partial Dependence Plots with pagination for large feature sets.
        
        Parameters
        ----------
        results : Dict[str, Dict[str, np.ndarray]]
            PDP results for each feature
        feature_subset : List[str]
            List of features to plot
        max_features_per_figure : int, default=12
            Maximum number of features to display per figure
        """
        num_features = len(feature_subset)
        pages = int(np.ceil(num_features / max_features_per_figure))
        for page in range(pages):
            start = page * max_features_per_figure
            end = min((page + 1) * max_features_per_figure, num_features)
            features_page = feature_subset[start:end]
            n = len(features_page)
            num_cols = min(n, 4)
            num_rows = int(np.ceil(n / num_cols))
            fig, axes = plt.subplots(
                nrows=num_rows,
                ncols=num_cols,
                figsize=(5 * num_cols, 4 * num_rows)
            )
            axes = np.array(axes).reshape(num_rows, num_cols)
            for idx, feature in enumerate(features_page):
                row, col = divmod(idx, num_cols)
                ax = axes[row, col]
                pdp_result = results[feature]
                ax.plot(
                    pdp_result['grid_values'][0],
                    pdp_result['average'][0],
                    marker='o'
                )
                ax.set_xlabel(feature, fontsize=10, labelpad=6)
                ax.set_ylabel("Partial Dependence", fontsize=10, labelpad=6)
                ax.set_title(f"PDP for {feature}", fontsize=11, pad=10)
                ax.grid(True)
                # Rotate x-tick labels for readability
                for label in ax.get_xticklabels():
                    label.set_rotation(30)
                    label.set_fontsize(8)
                for label in ax.get_yticklabels():
                    label.set_fontsize(8)
            # Hide empty subplots
            for idx in range(n, num_rows * num_cols):
                row, col = divmod(idx, num_cols)
                fig.delaxes(axes[row, col])
            plt.tight_layout(pad=4)
            plt.show()