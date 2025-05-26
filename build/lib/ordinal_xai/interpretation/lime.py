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
from sklearn.model_selection import ParameterGrid
import logging
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
        comparison_method : str, default='one_vs_following'
            Method for comparing classes
        model_type : str, default='logistic'
            Type of surrogate model to use
        kernel_width : float, default=0.75
            Width of the exponential kernel
        custom_kernel : callable, optional
            Custom kernel function
        sampling : str, default='permute'
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
        
        if comparison_method not in ['one_vs_next', 'one_vs_following']:
            raise ValueError("comparison_method must be either 'one_vs_next' or 'one_vs_following'")
        if kernel_width <= 0:
            raise ValueError("kernel_width must be positive")
        if sampling not in ['grid', 'uniform', 'permute']:
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
                          pred_class: int) -> None:
        """
        Plot feature coefficients for higher and lower class predictions.
        
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
        # Determine which plots to show
        show_higher = higher_coef is not None
        show_lower = lower_coef is not None
        n_plots = show_higher + show_lower
        if n_plots == 0:
            logger.warning("No coefficients to plot.")
            return
            
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 6 * n_plots))
        if n_plots == 1:
            axes = [axes]
            
        # Compose observation info (multi-line, compact)
        obs_header = f"Observation {observation_idx}  |  Predicted class: {pred_class}"
        obs_values = ",  ".join([f"{name}: {value}" for name, value in observation.items()])
        obs_info = f"{obs_header}  |  {obs_values}"
        fig.suptitle(obs_info, fontsize=8)
        
        plot_idx = 0
        if show_higher:
            ax = axes[plot_idx]
            ax.barh(range(len(higher_coef)), higher_coef, color='#4682b4')
            ax.set_yticks(range(len(higher_coef)))
            ax.set_yticklabels(feature_names, fontsize=10)
            if self.comparison_method == 'one_vs_next':
                title = f'Surrogate Model Coefficients for Class {pred_class + 1}'
            else:
                title = f'Surrogate Model Coefficients for Classes > {pred_class}'
            ax.set_title(title, fontsize=13, pad=10)
            ax.set_xlabel('Coefficient Value')
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            plot_idx += 1
            
        if show_lower:
            ax = axes[plot_idx]
            ax.barh(range(len(lower_coef)), lower_coef, color='#b44646')
            ax.set_yticks(range(len(lower_coef)))
            ax.set_yticklabels(feature_names, fontsize=10)
            if self.comparison_method == 'one_vs_next':
                title = f'Surrogate Model Coefficients for Class {pred_class - 1}'
            else:
                title = f'Surrogate Model Coefficients for Classes < {pred_class}'
            ax.set_title(title, fontsize=13, pad=10)
            ax.set_xlabel('Coefficient Value')
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            
        plt.tight_layout(h_pad=4)
        plt.subplots_adjust(top=0.92)
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
                          pred_class: int) -> None:
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
        obs_info = f"{obs_header}  |  {obs_values}"
        fig.suptitle(obs_info, fontsize=8)
        
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
                **kwargs) -> Dict[str, Union[List[str], np.ndarray, DecisionTreeClassifier]]:
        """
        Generate LIME explanations for a specific observation.
        
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
                        higher_model = LogisticRegression(random_state=42, class_weight="balanced")
                        higher_model.fit(X_transformed, higher_mask, sample_weight=weights)
                        higher_coef = higher_model.coef_[0]
                        result['higher_coef'] = higher_coef
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
                        lower_model = LogisticRegression(random_state=42, class_weight="balanced")
                        lower_model.fit(X_transformed, lower_mask, sample_weight=weights)
                        lower_coef = lower_model.coef_[0]
                        result['lower_coef'] = lower_coef
                    except Exception as e:
                        logger.warning(f"Failed to fit lower class model: {str(e)}")
                        result['lower_coef'] = np.zeros(X_transformed.shape[1])
                else:
                    logger.warning("No positive samples for lower class comparison")
                    result['lower_coef'] = np.zeros(X_transformed.shape[1])
                
            if plot:
                self._plot_coefficients(higher_coef, lower_coef, feature_names, 
                                     observation_idx, observation, pred_class)
                                     
        elif self.model_type == "decision_tree":
            higher_model = None
            lower_model = None
            
            if pred_class < n_classes - 1:
                # Check if we have any positive samples for higher class
                if np.any(higher_mask):
                    try:
                        higher_model = DecisionTreeClassifier(random_state=42, max_depth=3)
                        higher_model.fit(X_transformed, higher_mask, sample_weight=weights)
                        result['higher_model'] = higher_model
                    except Exception as e:
                        logger.warning(f"Failed to fit higher class model: {str(e)}")
                else:
                    logger.warning("No positive samples for higher class comparison")
                
            if pred_class > 0:
                # Check if we have any positive samples for lower class
                if np.any(lower_mask):
                    try:
                        lower_model = DecisionTreeClassifier(random_state=42, max_depth=3)
                        lower_model.fit(X_transformed, lower_mask, sample_weight=weights)
                        result['lower_model'] = lower_model
                    except Exception as e:
                        logger.warning(f"Failed to fit lower class model: {str(e)}")
                else:
                    logger.warning("No positive samples for lower class comparison")
                
            if plot:
                self._plot_decision_tree(higher_model, lower_model, feature_names,
                                      observation_idx, observation, pred_class)
                    
        return result