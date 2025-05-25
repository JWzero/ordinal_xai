import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .base_interpretation import BaseInterpretation
from ..utils import pdp_modified
from typing import Optional, List, Union, Dict, Any

class PDPProb(BaseInterpretation):
    """
    Partial Dependence Plot (PDP) interpretation method for ordinal regression models.
    This version visualizes class probabilities as stacked area plots.
    """
    
    def __init__(self, model: Any, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Initialize the PDPProb interpretation method.
        
        Parameters
        ----------
        model : BaseOrdinalModel
            The trained ordinal regression model.
        X : pd.DataFrame
            DataFrame containing the dataset used for interpretation.
        y : Optional[pd.Series], default=None
            Series containing target labels.
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
        Generate Partial Dependence Plots as stacked area plots for class probabilities.
        
        Parameters
        ----------
        observation_idx : Optional[int], default=None
            Ignored (PDP is a global method).
        feature_subset : Optional[Union[List[str], List[int]]], default=None
            List of feature names or indices to plot. If None, all features are used.
        plot : bool, default=False
            Whether to create visualizations.
        max_features_per_figure : int, default=12
            Maximum number of features to display per figure (for large datasets).
        
        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            Dictionary containing PDP results for each feature:
            - 'grid_values': Feature values used for plotting
            - 'average': Average predicted probabilities at each feature value
        """
        if feature_subset is None:
            feature_subset = self.X.columns.tolist()
        else:
            feature_subset = [self.X.columns[i] if isinstance(i, int) else i for i in feature_subset]
        
        num_features = len(feature_subset)
        results = {}

        if not self.model.is_fitted_:
            self.model.fit(self.X, self.y) # Ensure model is fitted

        for feature in feature_subset:
            feature_idx = [self.X.columns.get_loc(feature)]
            pdp_result = pdp_modified.partial_dependence(
                self.model, self.X, features=feature_idx, response_method="predict_proba"
            )
            results[feature] = pdp_result

        if plot:
            self._plot_pdps_prob(results, feature_subset, max_features_per_figure)
        
        print(f"Generated PDPs for features: {feature_subset}")
        return results

    def _plot_pdps_prob(
        self,
        results: Dict[str, Dict[str, np.ndarray]],
        feature_subset: List[str],
        max_features_per_figure: int = 12
    ) -> None:
        """
        Create visualization of PDP probability plots with pagination for large feature sets.
        
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
            legend_handles = None
            for idx, feature in enumerate(features_page):
                row, col = divmod(idx, num_cols)
                ax = axes[row, col]
                pdp_result = results[feature]
                num_ranks = pdp_result['average'].shape[0]
                x_values = pdp_result['grid_values'][0]
                probabilities = pdp_result['average']
                stack = ax.stackplot(
                    x_values, probabilities,
                    labels=[f"Rank {rank}" for rank in range(num_ranks)], alpha=0.7
                )
                if legend_handles is None:
                    legend_handles = stack
                ax.set_xlabel(feature, fontsize=10, labelpad=6)
                ax.set_ylabel("Partial Dependence Probabilities", fontsize=10, labelpad=6)
                ax.set_title(f"PDP for {feature}", fontsize=11, pad=10)
                ax.grid(True)
                for label in ax.get_xticklabels():
                    label.set_rotation(30)
                    label.set_fontsize(8)
                for label in ax.get_yticklabels():
                    label.set_fontsize(8)
            # Shared legend for the figure
            if legend_handles is not None:
                fig.legend(
                    handles=legend_handles,
                    loc='lower right',
                    fontsize=10,
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
            for idx in range(n, num_rows * num_cols):
                row, col = divmod(idx, num_cols)
                fig.delaxes(axes[row, col])
            plt.tight_layout(pad=4)
            plt.show()
