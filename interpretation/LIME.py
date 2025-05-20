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
    """Local Interpretable Model-agnostic Explanations for ordinal regression models.
    
    This class implements LIME (Local Interpretable Model-agnostic Explanations) for
    ordinal regression models. It provides local explanations by fitting interpretable
    models to explain individual predictions.
    
    Attributes:
        model: The trained ordinal regression model
        X: Training data
        y: Target labels
        comparison_method: Method for comparing classes ('one_vs_next' or 'one_vs_following')
        model_type: Type of surrogate model to use
        kernel_width: Width of the exponential kernel for sample weighting
        custom_kernel: Custom kernel function for sample weighting
        sampling: Sampling strategy for generating perturbed samples
        max_samples: Maximum number of samples to generate
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
        """Initialize LIME interpretation.
        
        Args:
            model: The trained ordinal regression model
            X: Training data
            y: Target labels
            comparison_method: Either 'one_vs_next' or 'one_vs_following'
            model_type: Type of surrogate model to use
            kernel_width: Width of the exponential kernel for sample weighting
            custom_kernel: Custom kernel function for sample weighting
            sampling: Sampling strategy ('grid', 'uniform', or 'permute')
            max_samples: Maximum number of samples to generate
            
        Raises:
            ValueError: If comparison_method is invalid or kernel_width is non-positive
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
        """Compute sample weights using exponential kernel with Gower's distance.
        
        Args:
            samples: DataFrame containing perturbed samples
            observation: Series containing the observation to explain
            kernel: Optional custom kernel function
            
        Returns:
            Array of weights for each sample
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
        """Get binary labels for comparison based on method.
        
        Args:
            pred_class: Predicted class of the observation
            samples_preds: Array of predictions for all samples
            
        Returns:
            Tuple of (higher_mask, lower_mask) arrays indicating which samples
            are in higher/lower classes than the prediction
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
        """Plot feature coefficients for higher and lower class predictions.
        
        Args:
            higher_coef: Coefficients for higher class comparison
            lower_coef: Coefficients for lower class comparison
            feature_names: List of feature names
            observation_idx: Index of the observation being explained
            observation: The observation being explained
            pred_class: Predicted class of the observation
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
        """Generate samples using grid sampling strategy.
        
        Args:
            X: Original dataset
            
        Returns:
            DataFrame containing grid samples
            
        Raises:
            ValueError: If grid size would exceed max_samples
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
        """Generate samples using uniform sampling strategy.
        
        Args:
            X: Original dataset
            
        Returns:
            DataFrame containing uniformly sampled points
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
        """Generate samples using permutation sampling strategy.
        
        Args:
            X: Original dataset
            
        Returns:
            DataFrame containing permuted samples
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
        """Plot decision tree surrogate models.
        
        Args:
            higher_model: Fitted decision tree model for higher class comparison
            lower_model: Fitted decision tree model for lower class comparison
            feature_names: List of feature names
            observation_idx: Index of the observation being explained
            observation: The observation being explained
            pred_class: Predicted class of the observation
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
        """Generate LIME explanations for a specific observation.
        
        Args:
            observation_idx: Index of the observation to explain
            feature_subset: Optional list of feature indices or names to include
            plot: Whether to create visualizations of the coefficients or trees
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing:
            - features: List of feature names
            - higher_model: Decision tree model for higher class comparison (if model_type="decision_tree")
            - lower_model: Decision tree model for lower class comparison (if model_type="decision_tree")
            - higher_coef: Coefficients for higher class comparison (if model_type="logistic")
            - lower_coef: Coefficients for lower class comparison (if model_type="logistic")
            
        Raises:
            ValueError: If observation_idx is not specified or model_type is invalid
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
                higher_model = LogisticRegression(random_state=42, class_weight="balanced")
                higher_model.fit(X_transformed, higher_mask, sample_weight=weights)
                higher_coef = higher_model.coef_[0]
                result['higher_coef'] = higher_coef
                
            if pred_class > 0:
                lower_model = LogisticRegression(random_state=42, class_weight="balanced")
                lower_model.fit(X_transformed, lower_mask, sample_weight=weights)
                lower_coef = lower_model.coef_[0]
                result['lower_coef'] = lower_coef
                
            if plot:
                self._plot_coefficients(higher_coef, lower_coef, feature_names, 
                                     observation_idx, observation, pred_class)
                                     
        elif self.model_type == "decision_tree":
            higher_model = None
            lower_model = None
            
            if pred_class < n_classes - 1:
                higher_model = DecisionTreeClassifier(random_state=42, max_depth=3)
                higher_model.fit(X_transformed, higher_mask, sample_weight=weights)
                result['higher_model'] = higher_model
                
            if pred_class > 0:
                lower_model = DecisionTreeClassifier(random_state=42, max_depth=3)
                lower_model.fit(X_transformed, lower_mask, sample_weight=weights)
                result['lower_model'] = lower_model
                
            if plot:
                self._plot_decision_tree(higher_model, lower_model, feature_names,
                                      observation_idx, observation, pred_class)
                    
        return result