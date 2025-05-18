import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from .base_interpretation import BaseInterpretation
import pandas as pd

class LIME(BaseInterpretation):
    """Local Interpretable Model-agnostic Explanations for ordinal regression models."""
    
    def __init__(self, model, X: pd.DataFrame, y: np.ndarray = None, 
                 comparison_method: str = 'one_vs_next',
                 kernel_width: float = 0.75,
                 n_samples: int = 1000,
                 sampling: str = None):
        """
        Initialize LIME interpretation.
        
        Parameters:
        - model: The trained ordinal regression model
        - X: Training data
        - y: Target labels
        - comparison_method: Either 'one_vs_next' or 'one_vs_following'
        - kernel_width: Width of the exponential kernel for sample weighting
        - n_samples: Number of samples to generate around the observation
        - sampling: Sampling method for generating samples:
            - None: Use training set X
            - 'uniform': Sample uniformly from feature ranges
            - 'grid': Create equidistant grid over feature ranges
        """
        print("Initializing LIME...")
        super().__init__(model, X, y)
        self.comparison_method = comparison_method
        self.kernel_width = kernel_width
        self.n_samples = n_samples
        self.sampling = sampling
        
        # Identify categorical and numerical columns
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        print(f"Categorical columns: {self.categorical_cols}")
        print(f"Numerical columns: {self.numerical_cols}")
        
        # Store feature ranges for sampling
        self.feature_ranges = {}
        for col in self.numerical_cols:
            self.feature_ranges[col] = (X[col].min(), X[col].max())
        for col in self.categorical_cols:
            self.feature_ranges[col] = X[col].unique()
        
        # Store feature names for plotting
        self.feature_names = list(self.numerical_cols) + list(self.categorical_cols)
        print(f"Feature names: {self.feature_names}")
        
    def _generate_samples(self, observation: pd.Series) -> pd.DataFrame:
        """Generate samples around the observation."""
        print("Generating samples...")
        if self.sampling is None:
            # Use training set samples
            return self.X
        
        # Create empty DataFrame with same columns as observation
        samples = pd.DataFrame(columns=observation.index)
        
        if self.sampling == 'uniform':
            # Sample uniformly from feature ranges
            for col in observation.index:
                if col in self.numerical_cols:
                    samples[col] = np.random.uniform(
                        self.feature_ranges[col][0],
                        self.feature_ranges[col][1],
                        self.n_samples
                    )
                else:  # categorical
                    samples[col] = np.random.choice(
                        self.feature_ranges[col],
                        self.n_samples
                    )
        
        elif self.sampling == 'grid':
            # Create equidistant grid over feature ranges
            for col in observation.index:
                if col in self.numerical_cols:
                    samples[col] = np.linspace(
                        self.feature_ranges[col][0],
                        self.feature_ranges[col][1],
                        self.n_samples
                    )
                else:  # categorical
                    samples[col] = np.random.choice(
                        self.feature_ranges[col],
                        self.n_samples
                    )
        
        return samples
    
    def _compute_weights(self, samples: pd.DataFrame, observation: pd.Series) -> np.ndarray:
        """Compute sample weights using exponential kernel."""
        print("Computing weights...")
        # Use model's transform for distance calculation
        samples_encoded = self.model.transform(samples, fit=False)
        observation_encoded = self.model.transform(observation.to_frame().T, fit=False)
        
        # Calculate distances in the encoded space
        distances = np.linalg.norm(samples_encoded - observation_encoded, axis=1)
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        return weights
    
    def _get_comparison_labels(self, pred_class: int, samples_preds: np.ndarray) -> tuple:
        """Get binary labels for comparison based on method."""
        print("Getting comparison labels...")
        if self.comparison_method == 'one_vs_next':
            # Compare with next class
            higher_mask = samples_preds > pred_class
            lower_mask = samples_preds < pred_class
        else:  # one_vs_following
            # Compare with all higher/lower classes
            higher_mask = samples_preds > pred_class
            lower_mask = samples_preds < pred_class
            
        return higher_mask, lower_mask
    
    def _plot_coefficients(self, higher_coef: np.ndarray, lower_coef: np.ndarray):
        """Plot feature coefficients for higher and lower class predictions."""
        plt.figure(figsize=(10, 6))
        
        # Get feature names
        feature_names = self.X.columns
        
        # Create bar positions
        x = np.arange(len(feature_names))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, higher_coef, width, label='Higher Class')
        plt.bar(x + width/2, lower_coef, width, label='Lower Class')
        
        # Customize plot
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.title('LIME Feature Importance')
        plt.xticks(x, feature_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def explain(self, observation_idx: int, feature_subset: list = None, plot: bool = False) -> dict:
        """
        Generate local explanations for a specific observation.
        
        Parameters:
        - observation_idx: Index of the observation to explain
        - feature_subset: (Optional) List of feature names to consider
        - plot: (Optional) Whether to create visualizations
        
        Returns:
        - Dictionary containing feature importance coefficients
        """
        if not self.model.is_fitted_:
            self.model.fit(self.X, self.y) # Ensure model is fitted
            
        print(f"Starting explanation for observation {observation_idx}...")
        if observation_idx is None:
            raise ValueError("observation_idx must be specified for LIME")
            
        # Get the observation
        observation = self.X.iloc[observation_idx]
        if feature_subset is not None:
            observation = observation[feature_subset]
        print(f"Observation shape: {observation.shape}")
            
        # Generate samples around the observation
        samples = self._generate_samples(observation)
        print(f"Samples shape: {samples.shape}")
        
        # Use model's transform for prediction
        samples_transformed_for_model = self.model.transform(samples, fit=False)
        observation_transformed_for_model = self.model.transform(observation.to_frame().T, fit=False)
        samples_preds = self.model.predict(samples_transformed_for_model)
        print(f"Sample predictions shape: {samples_preds.shape}")
        
        # Get original prediction
        pred_class = self.model.predict(observation_transformed_for_model)[0]
        print(f"Original prediction class: {pred_class}")
        
        # Compute sample weights
        weights = self._compute_weights(samples, observation)
        print(f"Weights shape: {weights.shape}")
        
        # Get comparison labels
        higher_mask, lower_mask = self._get_comparison_labels(pred_class, samples_preds)
        print(f"Higher mask sum: {higher_mask.sum()}, Lower mask sum: {lower_mask.sum()}")
        
        # Transform samples for training (use model.transform)
        samples_transformed = self.model.transform(samples, fit=False)
        print(f"Transformed samples shape: {samples_transformed.shape}")
        
        # Train surrogate models
        higher_model = LogisticRegression(random_state=42)
        lower_model = LogisticRegression(random_state=42)
        
        # Fit models with sample weights
        higher_model.fit(samples_transformed, higher_mask, sample_weight=weights)
        lower_model.fit(samples_transformed, lower_mask, sample_weight=weights)
        print("Models fitted successfully")
        
        # Get coefficients
        higher_coef = higher_model.coef_[0]
        lower_coef = lower_model.coef_[0]
        print(f"Coefficients shape: {higher_coef.shape}")
        
        # Plot if requested
        if plot:
            self._plot_coefficients(higher_coef, lower_coef)
        
        return {
            'higher_coef': higher_coef,
            'lower_coef': lower_coef
        } 