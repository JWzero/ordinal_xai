"""
Local Interpretable Model-agnostic Explanations (LIME) for ordinal regression models.

This module implements LIME for ordinal regression models, providing local explanations
by fitting interpretable surrogate models to explain individual predictions. The implementation
extends standard LIME to handle ordinal data by comparing predictions with adjacent or
following classes.

Key Features:
- Local explanations for individual predictions
- Support for both logistic regression and decision tree surrogate models
- Multiple sampling strategies (grid, uniform, permutation)
- Customizable kernel functions for sample weighting
- Visualization of feature importance through coefficients or decision trees
- Support for both numerical and categorical features

The implementation is particularly useful for:
- Understanding individual predictions in ordinal regression models
- Identifying key features that influence specific predictions
- Comparing feature importance across different classes
- Providing interpretable explanations for model decisions
"""

from typing import Optional, List, Dict, Union, Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from .base_interpretation import BaseInterpretation
import matplotlib.pyplot as plt
import gower
from sklearn.model_selection import ParameterGrid, train_test_split
import logging
from sklearn.metrics import accuracy_score, log_loss
import re
logger = logging.getLogger(__name__)

class LIME(BaseInterpretation):
    """
    Local Interpretable Model-agnostic Explanations for ordinal regression models.
    
    This class implements LIME for ordinal regression models, providing local explanations
    by fitting interpretable surrogate models to explain individual predictions. The implementation
    extends standard LIME to handle ordinal data by comparing predictions with adjacent or
    following classes.
    
    Parameters
    ----------
    model : object
        The trained ordinal regression model. Must implement predict and transform methods.
    X : pd.DataFrame
        Dataset used for interpretation. Should contain the same features used
        during model training.
    y : np.ndarray, optional
        Target labels. Not required for interpretation but useful for reference.
    comparison_method : str, default='one_vs_following'
        Method for comparing classes:
        - 'one_vs_next': Compare with adjacent classes only
        - 'one_vs_following': Compare with all higher/lower classes
    model_type : str, default='logistic'
        Type of surrogate model to use:
        - 'logistic': Logistic regression model
        - 'decision_tree': Decision tree model
    kernel_width : float, default=0.75
        Width of the exponential kernel for sample weighting. Controls how quickly
        the weight of samples decreases with distance.
    custom_kernel : callable, optional
        Custom kernel function for sample weighting. Should take distances as input
        and return weights.
    sampling : str, default='permute'
        Sampling strategy for generating perturbed samples:
        - 'grid': Generate samples on a grid (for small feature spaces)
        - 'uniform': Sample uniformly from feature ranges
        - 'permute': Permute feature values from the dataset
    max_samples : int, default=10000
        Maximum number of samples to generate for surrogate model training.
        
    Attributes
    ----------
    model : object
        The trained ordinal regression model
    X : pd.DataFrame
        Training data
    y : np.ndarray
        Target labels
    comparison_method : str
        Method for comparing classes
    model_type : str
        Type of surrogate model
    kernel_width : float
        Width of the exponential kernel
    custom_kernel : callable
        Custom kernel function
    sampling : str
        Sampling strategy
    max_samples : int
        Maximum number of samples
        
    Raises
    ------
    ValueError
        If comparison_method is invalid, kernel_width is non-positive,
        or sampling strategy is invalid
    """
    
    def __init__(self, 
                 model, 
                 X: pd.DataFrame, 
                 y: Optional[np.ndarray] = None, 
                 comparison_method: str = 'one_vs_following',
                 model_type: str = "logistic",
                 kernel_width: float = 0.75,
                 custom_kernel: Optional[Callable] = None,
                 sampling: str = "permute",
                 max_samples: int = 10000) -> None:
        """
        Initialize LIME interpretation.
        
        Parameters
        ----------
        model : object
            The trained ordinal regression model
        X : pd.DataFrame
            Training data
        y : np.ndarray, optional
            Target labels
        comparison_method : "one_vs_following" | "one_vs_next", default='one_vs_following'
            Method for comparing classes
        model_type : "logistic" | "decision_tree", default='logistic'
            Type of surrogate model to use
        kernel_width : float, default=0.75
            Width of the exponential kernel
        custom_kernel : callable, optional
            Custom kernel function
        sampling : "grid" | "uniform" | "permute", default='permute'
            Sampling strategy
        max_samples : int, default=10000
            Maximum number of samples
            
        Raises
        ------
        ValueError
            If comparison_method is invalid, kernel_width is non-positive,
            or sampling strategy is invalid
        """
        super().__init__(model, X, y)
        
        if comparison_method not in ["one_vs_next", "one_vs_following"]:
            raise ValueError("comparison_method must be either 'one_vs_next' or 'one_vs_following'")
        if kernel_width <= 0:
            raise ValueError("kernel_width must be positive")
        if sampling not in ["grid", "uniform", "permute"]:
            raise ValueError("sampling must be one of: 'grid', 'uniform', 'permute'")
            
        self.model = model
        self.comparison_method = comparison_method
        self.model_type = model_type
        self.kernel_width = kernel_width
        self.custom_kernel = custom_kernel
        self.sampling = sampling
        self.max_samples = max_samples
        
        logger.info(f"Initialized LIME with {comparison_method} comparison method and {sampling} sampling")
    
    def _compute_weights(self, 
                        samples: pd.DataFrame, 
                        observation: pd.Series, 
                        kernel: Optional[Callable] = None) -> np.ndarray:
        """
        Compute sample weights using exponential kernel with Gower's distance.
        
        This method calculates weights for perturbed samples based on their distance
        from the observation being explained. It uses Gower's distance to handle
        both numerical and categorical features.
        
        Parameters
        ----------
        samples : pd.DataFrame
            DataFrame containing perturbed samples
        observation : pd.Series
            Series containing the observation to explain
        kernel : callable, optional
            Custom kernel function for weight calculation
            
        Returns
        -------
        np.ndarray
            Array of weights for each sample
            
        Notes
        -----
        - Uses Gower's distance to handle mixed data types
        - Applies exponential kernel by default
        - Weights decrease exponentially with distance
        """
        # Convert observation to DataFrame
        observation_df = pd.DataFrame([observation])
        
        # Calculate Gower's distances from observation to all samples
        distances = gower.gower_matrix(samples, observation_df).flatten()
        
        if kernel is None:
            # Apply exponential kernel
            weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        else:
            weights = kernel(distances)
        return weights
    
    def _get_comparison_labels(self, 
                             pred_class: int, 
                             samples_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get binary labels for comparison based on method.
        
        This method creates binary labels for the surrogate model based on the
        comparison method chosen. It identifies which samples are in higher or
        lower classes than the prediction.
        
        Parameters
        ----------
        pred_class : int
            Predicted class of the observation
        samples_preds : np.ndarray
            Array of predictions for all samples
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (higher_mask, lower_mask) arrays indicating which samples
            are in higher/lower classes than the prediction
            
        Notes
        -----
        - For 'one_vs_next', only compares with adjacent classes
        - For 'one_vs_following', compares with all higher/lower classes
        """
        if self.comparison_method == 'one_vs_next':
            # Compare with next class
            higher_mask = samples_preds == pred_class + 1
            lower_mask = samples_preds == pred_class - 1
        else:  # one_vs_following
            # Compare with all higher/lower classes
            higher_mask = samples_preds > pred_class
            lower_mask = samples_preds < pred_class
            
        return higher_mask, lower_mask
    
    def _plot_coefficients(self, 
                          higher_coef: Optional[np.ndarray], 
                          lower_coef: Optional[np.ndarray], 
                          feature_names: List[str], 
                          observation_idx: int, 
                          observation: pd.Series, 
                          pred_class: int,
                          fidelity_in: Optional[float] = None,
                          fidelity_out: Optional[float] = None,
                          is_effect: bool = False) -> None:
        """
        Plot horizontal bars for either surrogate coefficients or predictor
        effects (coefficient × value) for higher / lower class comparisons.
        
        This method creates horizontal bar plots showing the coefficients of the
        logistic regression surrogate model for both higher and lower class
        comparisons.
        
        Parameters
        ----------
        higher_coef : np.ndarray, optional
            Coefficients for higher class comparison
        lower_coef : np.ndarray, optional
            Coefficients for lower class comparison
        feature_names : List[str]
            List of feature names
        observation_idx : int
            Index of the observation being explained
        observation : pd.Series
            The observation being explained
        pred_class : int
            Predicted class of the observation
            
        Notes
        -----
        - Creates separate plots for higher and lower class comparisons
        - Shows feature values in the title
        - Uses blue for higher class and red for lower class
        - Includes observation details in the title
        """
        # Determine availability of coefficient sets
        show_higher = higher_coef is not None
        show_lower = lower_coef is not None
        if not (show_higher or show_lower):
            logger.warning("No coefficients to plot.")
            return

        # Compose observation info and fidelity lines
        obs_header = f"Observation {observation_idx}  |  Predicted class: {pred_class}"
        obs_values = ",  ".join([f"{name}: {value}" for name, value in observation.items()])
        fidelity_line = ""
        if fidelity_in is not None or fidelity_out is not None:
            parts = []
            if fidelity_in is not None:
                parts.append(f"Fidelity(in-sample): {fidelity_in:.3f}")
            if fidelity_out is not None:
                parts.append(f"Fidelity(out-of-sample): {fidelity_out:.3f}")
            fidelity_line = " | ".join(parts)
        obs_info = f"{obs_header}  |  {obs_values}"
        if fidelity_line:
            obs_info = f"{obs_info}\n{fidelity_line}"

        # CASE 1 – both coefficient vectors are present → create one stacked bar chart
        if show_higher and show_lower:
            # Interleave bars per feature: higher first, then lower, maintaining horizontal orientation
            n_features = len(feature_names)
            y_high = np.arange(0, n_features * 2, 2)
            y_low = y_high + 1

            fig_height = max(6, n_features)  # scale figure height with number of rows
            fig, ax = plt.subplots(figsize=(10, fig_height))
            fig.suptitle(obs_info, fontsize=8)

            # Plot bars
            ax.barh(y_high, higher_coef, color="#4682b4", label="Higher rank")
            ax.barh(y_low, lower_coef, color="#b44646", label="Lower rank")
            # X-axis label
            xlabel = "Predictor effect" if is_effect else "Coefficient"
            ax.set_xlabel(xlabel)

            # Y-tick labels duplicated for high/low rows
            yticks = np.concatenate([y_high, y_low])
            ylabels = []
            for name in feature_names:
                ylabels.extend([f"{name} ↑"])
            for name in feature_names:
                ylabels.extend([f"{name} ↓"])
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=9)


            ax.set_xlabel("Predictor Effect Value" if is_effect else "Coefficient Value")
            if self.comparison_method == "one_vs_next":
                title = (
                    f"Surrogate Model {'Predictor Effects' if is_effect else 'Coefficients'} – Class {pred_class - 1} & {pred_class + 1}"
                )
            else:
                title = f"Surrogate Model {'Predictor Effects' if is_effect else 'Coefficients'} – Classes < {pred_class} & > {pred_class}"
            ax.set_title(title, fontsize=13, pad=10)
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.legend()

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()
            return

        # CASE 2 – only one of the coefficient/effect sets is present → single-bar chart
        coef = higher_coef if show_higher else lower_coef
        color = "#4682b4" if show_higher else "#b44646"
        label = "Higher rank" if show_higher else "Lower rank"

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(obs_info, fontsize=8)

        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, coef, color=color, label=label)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=10)
        title = (
            f"Surrogate Model Coefficients for Classes > {pred_class}"
            if show_higher
            else f"Surrogate Model Coefficients for Classes < {pred_class}"
        )
        if self.comparison_method == "one_vs_next":
            if show_higher:
                title = f"Surrogate Model Coefficients for Class {pred_class + 1}"
            else:
                title = f"Surrogate Model Coefficients for Class {pred_class - 1}"
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("Coefficient Value")
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    def _generate_grid_samples(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate samples using grid sampling strategy.
        
        This method creates a grid of samples by:
        1. Identifying categorical and numerical columns
        2. Using all unique values for categorical features
        3. Creating evenly spaced points for numerical features
        4. Combining all possible combinations
        
        Parameters
        ----------
        X : pd.DataFrame
            Original dataset
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing grid samples
            
        Raises
        ------
        ValueError
            If grid size would exceed max_samples
            
        Notes
        -----
        - For numerical features, uses linspace between min and max values
        - For categorical features, uses all unique values
        - Caps number of points per numerical feature at 100
        - Samples randomly if grid size exceeds max_samples
        """
        # Identify categorical and numerical columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        n_features = len(X.columns)
        
        # Throw error if 2^n_features > max_samples (for binary features, grid explosion)
        if 2 ** n_features > self.max_samples:
            raise ValueError(
                f"Grid would have {2 ** n_features} rows, which exceeds max_samples={self.max_samples}. "
                "Reduce the number of features or increase max_samples."
            )
            
        # For each categorical, use all unique values
        cat_values = [X[col].unique() for col in cat_cols]
        
        # For each numerical, use linspace across min/max
        num_ranges = [(X[col].min(), X[col].max()) for col in num_cols]
        n_cats = [len(vals) for vals in cat_values]
        n_num = len(num_cols)
        
        grid_dict = {}
        if n_num > 0 and not n_cats:  # Only numerical features
            n_per_num = max(10, int(np.round(self.max_samples ** (1 / n_num))))
            for col, (vmin, vmax) in zip(num_cols, num_ranges):
                grid_dict[col] = np.linspace(vmin, vmax, n_per_num)
        else:
            n_grid_num = max(10, int(np.floor(self.max_samples / np.prod(n_cats)**(1/n_num))) if n_cats else self.max_samples)
            num_grids = [min(n_grid_num, 100) for _ in num_cols]  # cap at 100 per feature for safety
            for col, (vmin, vmax), n in zip(num_cols, num_ranges, num_grids):
                grid_dict[col] = np.linspace(vmin, vmax, n)
                
        for col, vals in zip(cat_cols, cat_values):
            grid_dict[col] = vals
            
        # Use sklearn's ParameterGrid for grid creation
        grid = list(ParameterGrid(grid_dict))
        X_grid = pd.DataFrame(grid, columns=X.columns)
        
        # If grid is too large, sample max_samples rows
        if len(X_grid) > self.max_samples:
            X_grid = X_grid.sample(n=self.max_samples, random_state=42).reset_index(drop=True)
            
        return X_grid

    def _generate_uniform_samples(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate samples using uniform sampling strategy.
        
        This method creates samples by drawing uniformly from the range of each
        feature. For numerical features, samples from the min-max range. For
        categorical features, samples from unique values.
        
        Parameters
        ----------
        X : pd.DataFrame
            Original dataset
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing uniformly sampled points
            
        Notes
        -----
        - Numerical features: uniform sampling between min and max
        - Categorical features: random choice from unique values
        - Number of samples controlled by max_samples parameter
        """
        n_samples = self.max_samples
        samples = pd.DataFrame(index=range(n_samples), columns=X.columns)
        
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                samples[col] = np.random.uniform(X[col].min(), X[col].max(), n_samples)
            else:
                samples[col] = np.random.choice(X[col].unique(), n_samples)
                
        return samples

    def _generate_permute_samples(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate samples using permutation sampling strategy.
        
        This method creates samples by randomly permuting the values of each
        feature from the original dataset. This preserves the marginal
        distribution of each feature.
        
        Parameters
        ----------
        X : pd.DataFrame
            Original dataset
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing permuted samples
            
        Notes
        -----
        - Preserves marginal distributions of features
        - Breaks correlations between features
        - Number of samples controlled by max_samples parameter
        """
        n_samples = self.max_samples
        samples = pd.DataFrame(index=range(n_samples), columns=X.columns)
        
        for col in X.columns:
            samples[col] = np.random.choice(X[col], n_samples, replace=True)
            
        return samples

    def _plot_decision_tree(self,
                           higher_model: Optional[DecisionTreeClassifier],
                           lower_model: Optional[DecisionTreeClassifier],
                           feature_names: List[str],
                           observation_idx: int,
                           observation: pd.Series,
                           pred_class: int,
                           fidelity_in: Optional[float] = None,
                           fidelity_out: Optional[float] = None,
                           is_effect: bool = False) -> None:
        """
        Plot decision tree surrogate models.
        
        This method creates visualizations of the decision tree surrogate models
        for both higher and lower class comparisons. The trees show the decision
        rules learned by the surrogate model.
        
        Parameters
        ----------
        higher_model : DecisionTreeClassifier, optional
            Fitted decision tree model for higher class comparison
        lower_model : DecisionTreeClassifier, optional
            Fitted decision tree model for lower class comparison
        feature_names : List[str]
            List of feature names
        observation_idx : int
            Index of the observation being explained
        observation : pd.Series
            The observation being explained
        pred_class : int
            Predicted class of the observation
        fidelity_in : float, optional
            Fidelity of the model on the training data
        fidelity_out : float, optional
            Fidelity of the model on the test data
            
        Notes
        -----
        - Creates separate plots for higher and lower class comparisons
        - Shows feature values in the title
        - Uses simplified node text for better readability
        - Includes observation details in the title
        """
        # Determine which plots to show
        show_higher = higher_model is not None
        show_lower = lower_model is not None
        n_plots = show_higher + show_lower
        if n_plots == 0:
            logger.warning("No trees to plot.")
            return
            
        fig, axes = plt.subplots(1, n_plots, figsize=(20, 10 * n_plots))
        if n_plots == 1:
            axes = [axes]
            
        # Compose observation info (multi-line, compact)
        obs_header = f"Observation {observation_idx}  |  Predicted class: {pred_class}"
        obs_values = ",  ".join([f"{name}: {value}" for name, value in observation.items()])
        fidelity_line = ""
        if fidelity_in is not None or fidelity_out is not None:
            parts = []
            if fidelity_in is not None:
                parts.append(f"Fidelity(in-sample): {fidelity_in:.3f}")
            if fidelity_out is not None:
                parts.append(f"Fidelity(out-of-sample): {fidelity_out:.3f}")
            fidelity_line = " | ".join(parts)
        suptitle = f"{obs_header}  |  {obs_values}"
        if fidelity_line:
            suptitle = f"{suptitle}\n{fidelity_line}"
        fig.suptitle(suptitle, fontsize=8)
        
        plot_idx = 0
        if show_higher:
            ax = axes[plot_idx]
            tree= plot_tree(higher_model, 
                     feature_names=feature_names,
                     class_names=["Same", "Higher"],
                     filled=True,
                     rounded=True,
                     impurity=False,
                     proportion=False,
                     label="all",
                     fontsize=10,
                     ax=ax)
            #modify node texts
            for i, t in enumerate(ax.texts):
                text = t.get_text()
                if not "True" in text and not "False" in text:
                    text = text.split('\n')[:-3]+[text.split('\n')[-1]]
                    t.set_text('\n'.join(text))
            if self.comparison_method == 'one_vs_next':
                title = f'Decision Tree for Class {pred_class + 1}'
            else:
                title = f'Decision Tree for Classes > {pred_class}'
            ax.set_title(title, fontsize=13, pad=10)
            plot_idx += 1
            
        if show_lower:
            ax = axes[plot_idx]
            tree = plot_tree(lower_model, 
                     feature_names=feature_names,
                     class_names=["Same", "Lower"],
                     filled=True,
                     rounded=True,
                     impurity=False,
                     proportion=False,
                     fontsize=10,
                     label="all",
                     ax=ax)
            #modify node texts
            for i, t in enumerate(ax.texts):
                text = t.get_text()
                if not "True" in text and not "False" in text:
                    text = text.split('\n')[:-3]+[text.split('\n')[-1]]
                    t.set_text('\n'.join(text))
            if self.comparison_method == 'one_vs_next':
                title = f'Decision Tree for Class {pred_class - 1}'
            else:
                title = f'Decision Tree for Classes < {pred_class}'
            ax.set_title(title, fontsize=13, pad=10)
            
        plt.tight_layout(h_pad=4)
        plt.subplots_adjust(top=0.92)
        plt.show()

    def explain(self, 
                observation_idx: Optional[int] = None, 
                feature_subset: Optional[List[Union[int, str]]] = None, 
                plot: bool = False, 
                show_coefficients: bool = False,
                **kwargs) -> Dict[str, Union[List[str], np.ndarray, DecisionTreeClassifier]]:
        """
        Generate LIME explanations for a specific observation.

        Depending on the `show_coefficients` flag the horizontal bar plot will
        display either (a) raw surrogate coefficients or (b) *predictor effects*
        defined as `coefficient × local feature value` for numerical features.
        Categorical one-hot features always use the raw coefficient value.
        
        This method creates local explanations by:
        1. Generating perturbed samples around the observation
        2. Computing sample weights based on distance
        3. Fitting surrogate models to explain the prediction
        4. Visualizing the results if requested
        
        Parameters
        ----------
        observation_idx : int, optional
            Index of the observation to explain
        feature_subset : List[Union[int, str]], optional
            List of feature indices or names to include
        plot : bool, default=False
            If True, visualise the explanation.
        show_coefficients : bool, default=False
            If False (default) plot predictor effects; if True plot raw
            coefficients.
            Whether to create visualizations
        **kwargs : dict
            Additional keyword arguments
            
        Returns
        -------
        Dict[str, Union[List[str], np.ndarray, DecisionTreeClassifier]]
            Dictionary containing:
            - features: List of feature names
            - higher_model: Decision tree model for higher class comparison (if model_type="decision_tree")
            - lower_model: Decision tree model for lower class comparison (if model_type="decision_tree")
            - higher_coef: Coefficients for higher class comparison (if model_type="logistic")
            - lower_coef: Coefficients for lower class comparison (if model_type="logistic")
            
        Raises
        ------
        ValueError
            If observation_idx is not specified or model_type is invalid
            
        Notes
        -----
        - Requires observation_idx to be specified
        - Supports both logistic regression and decision tree surrogate models
        - Can focus on specific features using feature_subset
        - Provides visualizations of coefficients or decision trees
        """
        if observation_idx is None:
            raise ValueError("observation_idx must be specified for LIME")
            
        if self.model_type not in ["logistic", "decision_tree"]:
            
            raise ValueError("model_type must be either 'logistic' or 'decision_tree'")
            
        # Get the observation
        observation = self.X.iloc[observation_idx]
        if feature_subset is not None:
            observation = observation[feature_subset]

        # Choose samples for surrogate model fitting
        if self.sampling == "grid":
            samples = self._generate_grid_samples(self.X)
        elif self.sampling == "uniform":
            samples = self._generate_uniform_samples(self.X)
        elif self.sampling == "permute":
            samples = self._generate_permute_samples(self.X)
        else:
            samples = self.X
        
        # Get predictions for all data points
        samples_preds = self.model.predict(samples)
        
        # Get original prediction
        pred_class = self.model.predict(self.X)[observation_idx]
        logger.info(f"Original prediction class: {pred_class}")
        n_classes = len(np.unique(samples_preds))
        
        # Compute sample weights
        weights = self._compute_weights(samples, observation, self.custom_kernel)
        
        # Get comparison labels
        higher_mask, lower_mask = self._get_comparison_labels(pred_class, samples_preds)

        # Transform features
        X_transformed = self.model.transform(samples, fit=False, no_scaling=True)
        # Transformed observation for effect calculation (ensure correct dtypes)
        obs_transformed = self.model.transform(self.X.iloc[[observation_idx]], fit=False, no_scaling=True).iloc[0].values
        feature_names = X_transformed.columns.tolist()
        
        if feature_subset is not None:
            if all(isinstance(f, int) for f in feature_subset):
                idxs = feature_subset
            else:
                idxs = [feature_names.index(f) for f in feature_subset]
            feature_names = [feature_names[i] for i in idxs]
            X_transformed = X_transformed.iloc[:, idxs]

        result = {'features': feature_names}
        
        if self.model_type == "logistic":
            higher_coef = None
            lower_coef = None
            
            if pred_class < n_classes - 1:
                # Check if we have any positive samples for higher class
                if np.any(higher_mask):
                    try:
                        # Train/test split for fidelity estimation
                        idx_train, idx_test = train_test_split(np.arange(len(higher_mask)), test_size=0.2, random_state=42, stratify=higher_mask)
                        X_train, X_test = X_transformed.iloc[idx_train], X_transformed.iloc[idx_test]
                        y_train, y_test = higher_mask[idx_train], higher_mask[idx_test]
                        w_train, w_test = weights[idx_train], weights[idx_test]

                        higher_model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=10000)
                        higher_model.fit(X_train, y_train, sample_weight=w_train)
                        higher_coef = higher_model.coef_[0]
                        if not show_coefficients:
                            num_mask = ~np.isin(obs_transformed, [0, 1])
                            higher_effect = higher_coef * (obs_transformed * num_mask + (~num_mask) * 1)  # multiply only numeric
                            result['higher_effect'] = higher_effect
                        result['higher_coef'] = higher_coef

                        # Fidelity (binary cross-entropy) in and out of sample
                        prob_train = higher_model.predict_proba(X_train)[:, 1]
                        prob_test = higher_model.predict_proba(X_test)[:, 1]
                        in_loss = log_loss(y_train, prob_train, sample_weight=w_train, labels=[0, 1])
                        out_loss = log_loss(y_test, prob_test, sample_weight=w_test, labels=[0, 1])
                        result['higher_fidelity_in'] = in_loss
                        result['higher_fidelity_out'] = out_loss
                    except Exception as e:
                        logger.warning(f"Failed to fit higher class model: {str(e)}")
                        result['higher_coef'] = np.zeros(X_transformed.shape[1])
                else:
                    logger.warning("No positive samples for higher class comparison")
                    result['higher_coef'] = np.zeros(X_transformed.shape[1])
                
            if pred_class > 0:
                # Check if we have any positive samples for lower class
                if np.any(lower_mask):
                    try:
                        idx_train_l, idx_test_l = train_test_split(np.arange(len(lower_mask)), test_size=0.2, random_state=42, stratify=lower_mask)
                        X_train_l, X_test_l = X_transformed.iloc[idx_train_l], X_transformed.iloc[idx_test_l]
                        y_train_l, y_test_l = lower_mask[idx_train_l], lower_mask[idx_test_l]
                        w_train_l, w_test_l = weights[idx_train_l], weights[idx_test_l]

                        lower_model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=10000)
                        lower_model.fit(X_train_l, y_train_l, sample_weight=w_train_l)
                        lower_coef = lower_model.coef_[0]
                        if not show_coefficients:
                            num_mask = ~np.isin(obs_transformed, [0, 1])
                            lower_effect = lower_coef * (obs_transformed * num_mask + (~num_mask) * 1)
                            result['lower_effect'] = lower_effect
                        result['lower_coef'] = lower_coef
                        prob_train_l = lower_model.predict_proba(X_train_l)[:, 1]
                        prob_test_l = lower_model.predict_proba(X_test_l)[:, 1]
                        in_loss_l = log_loss(y_train_l, prob_train_l, sample_weight=w_train_l, labels=[0, 1])
                        out_loss_l = log_loss(y_test_l, prob_test_l, sample_weight=w_test_l, labels=[0, 1])
                        result['lower_fidelity_in'] = in_loss_l
                        result['lower_fidelity_out'] = out_loss_l
                    except Exception as e:
                        logger.warning(f"Failed to fit lower class model: {str(e)}")
                        result['lower_coef'] = np.zeros(X_transformed.shape[1])
                else:
                    logger.warning("No positive samples for lower class comparison")
                    result['lower_coef'] = np.zeros(X_transformed.shape[1])
                
            # Aggregate in- and out-of-sample BCE losses
            losses_in = []
            losses_out = []
            if 'higher_fidelity_in' in result:
                losses_in.append(result['higher_fidelity_in'])
            if 'lower_fidelity_in' in result:
                losses_in.append(result['lower_fidelity_in'])
            if 'higher_fidelity_out' in result:
                losses_out.append(result['higher_fidelity_out'])
            if 'lower_fidelity_out' in result:
                losses_out.append(result['lower_fidelity_out'])
            if losses_in:
                result['fidelity_in'] = float(np.mean(losses_in))
            if losses_out:
                result['fidelity_out'] = float(np.mean(losses_out))

            if plot:
                # Determine data to plot: coefficients or effects
                plot_high = higher_coef if show_coefficients else result.get('higher_effect', higher_coef)
                plot_low = lower_coef if show_coefficients else result.get('lower_effect', lower_coef)

                self._plot_coefficients(plot_high, plot_low, feature_names,
                                      observation_idx, observation, pred_class,
                                      fidelity_in=result.get('fidelity_in'),
                                      fidelity_out=result.get('fidelity_out'),
                                      is_effect=not show_coefficients)
                                     
        elif self.model_type == "decision_tree":
            higher_model = None
            lower_model = None
            
            if pred_class < n_classes - 1:
                # Check if we have any positive samples for higher class
                if np.any(higher_mask):
                    try:
                        idx_train_h, idx_test_h = train_test_split(np.arange(len(higher_mask)), test_size=0.2, random_state=42, stratify=higher_mask)
                        X_train_h, X_test_h = X_transformed.iloc[idx_train_h], X_transformed.iloc[idx_test_h]
                        y_train_h, y_test_h = higher_mask[idx_train_h], higher_mask[idx_test_h]
                        w_train_h, w_test_h = weights[idx_train_h], weights[idx_test_h]

                        higher_model = DecisionTreeClassifier(random_state=42, max_depth=3)
                        higher_model.fit(X_train_h, y_train_h, sample_weight=w_train_h)
                        result['higher_model'] = higher_model
                        pred_train_h = higher_model.predict(X_train_h)
                        pred_test_h = higher_model.predict(X_test_h)
                        loss_in_h = 1.0 - accuracy_score(y_train_h, pred_train_h, sample_weight=w_train_h)
                        loss_out_h = 1.0 - accuracy_score(y_test_h, pred_test_h, sample_weight=w_test_h)
                        result['higher_fidelity_in'] = loss_in_h
                        result['higher_fidelity_out'] = loss_out_h
                    except Exception as e:
                        logger.warning(f"Failed to fit higher class model: {str(e)}")
                else:
                    logger.warning("No positive samples for higher class comparison")
                
            if pred_class > 0:
                # Check if we have any positive samples for lower class
                if np.any(lower_mask):
                    try:
                        idx_train_l, idx_test_l = train_test_split(np.arange(len(lower_mask)), test_size=0.2, random_state=42, stratify=lower_mask)
                        X_train_l, X_test_l = X_transformed.iloc[idx_train_l], X_transformed.iloc[idx_test_l]
                        y_train_l, y_test_l = lower_mask[idx_train_l], lower_mask[idx_test_l]
                        w_train_l, w_test_l = weights[idx_train_l], weights[idx_test_l]

                        lower_model = DecisionTreeClassifier(random_state=42, max_depth=3)
                        lower_model.fit(X_train_l, y_train_l, sample_weight=w_train_l)
                        result['lower_model'] = lower_model
                        pred_train_l = lower_model.predict(X_train_l)
                        pred_test_l = lower_model.predict(X_test_l)
                        loss_in_l = 1.0 - accuracy_score(y_train_l, pred_train_l, sample_weight=w_train_l)
                        loss_out_l = 1.0 - accuracy_score(y_test_l, pred_test_l, sample_weight=w_test_l)
                        result['lower_fidelity_in'] = loss_in_l
                        result['lower_fidelity_out'] = loss_out_l
                    except Exception as e:
                        logger.warning(f"Failed to fit lower class model: {str(e)}")
                else:
                    logger.warning("No positive samples for lower class comparison")
                
            # Aggregate losses
            losses_in = []
            losses_out = []
            if 'higher_fidelity_in' in result:
                losses_in.append(result['higher_fidelity_in'])
            if 'lower_fidelity_in' in result:
                losses_in.append(result['lower_fidelity_in'])
            if 'higher_fidelity_out' in result:
                losses_out.append(result['higher_fidelity_out'])
            if 'lower_fidelity_out' in result:
                losses_out.append(result['lower_fidelity_out'])
            if losses_in:
                result['fidelity_in'] = float(np.mean(losses_in))
            if losses_out:
                result['fidelity_out'] = float(np.mean(losses_out))

            if plot:
                # pass fidelity for trees if needed (uses 'fidelity')
                self._plot_decision_tree(higher_model, lower_model, feature_names,
                                      observation_idx, observation, pred_class,
                                      fidelity_in=result.get('fidelity_in'),
                                      fidelity_out=result.get('fidelity_out'),
                                      is_effect=not show_coefficients)
                    
        return result