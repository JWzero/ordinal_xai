"""
Evaluation metrics for ordinal regression and classification.

This module provides a comprehensive set of metrics specifically designed for evaluating
ordinal regression and classification models. It includes both hard-label metrics (based on
predicted class labels) and probability-based metrics (based on predicted probabilities).

The metrics are designed to account for the ordinal nature of the data, where classes
have a natural ordering and misclassification costs increase with the distance between
predicted and true classes.

Available Metrics:
------------------
Hard Label Metrics:
- accuracy: Standard classification accuracy
- adjacent_accuracy: Proportion of predictions within one class of true label
- mze: Mean Zero-One Error (1 - accuracy)
- mae: Mean Absolute Error
- mse: Mean Squared Error
- weighted_kappa: Cohen's Kappa with linear or quadratic weights
- cem: Closeness Evaluation Measure
- spearman_correlation: Spearman's rank correlation
- kendall_tau: Kendall's Tau correlation

Probability-Based Metrics:
- ranked_probability_score: RPS for probabilistic predictions
- ordinal_weighted_ce: Ordinal weighted cross-entropy loss (Ordinal Log Loss)
"""

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from scipy.stats import kendalltau

def accuracy(y_true, y_pred):
    """
    Calculate accuracy for ordinal regression.
    
    This is the standard classification accuracy, measuring the proportion of
    correct predictions. While simple, it doesn't account for the ordinal nature
    of the data.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    y_pred : array-like of shape (n_samples,)
        Predicted ordinal labels
        
    Returns
    -------
    float
        Accuracy score between 0 and 1, where 1 indicates perfect predictions
    """
    return accuracy_score(y_true, y_pred)

def mze(y_true, y_pred):
    """
    Calculate Mean Zero-One Error (MZE) for ordinal regression.
    
    MZE is the complement of accuracy (1 - accuracy). It measures the proportion
    of incorrect predictions, treating all misclassifications equally regardless
    of their distance from the true class.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    y_pred : array-like of shape (n_samples,)
        Predicted ordinal labels
        
    Returns
    -------
    float
        Mean Zero-One Error between 0 and 1, where 0 indicates perfect predictions
    """
    return 1 - accuracy_score(y_true, y_pred)

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) for ordinal regression.
    
    MAE measures the average absolute difference between predicted and true labels.
    Unlike accuracy, it accounts for the ordinal nature of the data by penalizing
    predictions based on their distance from the true class.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    y_pred : array-like of shape (n_samples,)
        Predicted ordinal labels
        
    Returns
    -------
    float
        Mean Absolute Error, where 0 indicates perfect predictions
    """
    return mean_absolute_error(y_true, y_pred)

def mse(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) for ordinal regression.
    
    MSE measures the average squared difference between predicted and true labels.
    It penalizes larger errors more heavily than MAE due to the squaring operation.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    y_pred : array-like of shape (n_samples,)
        Predicted ordinal labels
        
    Returns
    -------
    float
        Mean Squared Error, where 0 indicates perfect predictions
    """
    return mean_squared_error(y_true, y_pred)

def weighted_kappa(y_true, y_pred, weights='quadratic'):
    """
    Calculate weighted kappa for ordinal regression.
    
    Weighted kappa extends Cohen's kappa to account for the ordinal nature of the data
    by applying weights to the confusion matrix (Cohen (1968)). The weights can be linear or quadratic,
    with quadratic weights penalizing larger misclassifications more heavily.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    y_pred : array-like of shape (n_samples,)
        Predicted ordinal labels
    weights : {'linear', 'quadratic', 'none'}, default='quadratic'
        Weighting scheme for the confusion matrix:
        - 'linear': Linear weights based on distance
        - 'quadratic': Quadratic weights (squared distance)
        - 'none': No weights (standard kappa)
        
    Returns
    -------
    float
        Weighted kappa score between -1 and 1, where:
        - 1 indicates perfect agreement
        - 0 indicates agreement equivalent to chance
        - -1 indicates perfect disagreement
    """
    return cohen_kappa_score(y_true, y_pred, weights=weights)

def _get_class_counts(y):
    """
    Calculate the count of items per class.
    
    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Array of class labels
        
    Returns
    -------
    dict
        Dictionary mapping class labels to their counts
    """
    unique_labels, counts = np.unique(y, return_counts=True)
    return dict(zip(unique_labels, counts))

def _calculate_proximity(c1, c2, class_counts, total_items):
    """
    Calculate proximity between two classes.
    
    This is a helper function for the CEM metric that calculates the proximity
    between two classes based on their positions and the distribution of classes
    in the dataset.
    
    Parameters
    ----------
    c1 : int
        First class label
    c2 : int
        Second class label
    class_counts : dict
        Dictionary mapping class labels to their counts
    total_items : int
        Total number of items in the dataset
        
    Returns
    -------
    float
        Proximity value between the two classes
    """
    if c1 == c2:
        return -np.log(class_counts[c1] / (2 * total_items))
    
    if c1 > c2:
        c1, c2 = c2, c1
    
    sum_counts = 0
    for k in range(c1 + 1, c2 + 1):
        sum_counts += class_counts.get(k, 0)
    
    return -np.log((class_counts[c1] / 2 + sum_counts) / total_items)

def cem(y_true, y_pred, class_counts=None):
    """
    Calculate Closeness Evaluation Measure (CEM) for ordinal classification.
    
    CEM is a metric proposed by Amigo et al. (2020) that evaluates the performance of ordinal classifiers based on measure and information theory. It uses a proximity-based
    approach that penalizes misclassifications based on their distance from the true class
    and the distribution of classes in the dataset.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    y_pred : array-like of shape (n_samples,)
        Predicted ordinal labels
    class_counts : dict, optional
        Dictionary mapping class labels to their counts. If None, calculated from y_true.
        Useful for local explanations where class distribution might differ from training.
        
    Returns
    -------
    float
        CEM score between 0 and 1, where:
        - 1 indicates perfect predictions
        - 0 indicates worst possible predictions
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if class_counts is None:
        class_counts = _get_class_counts(y_true)
        total_items = len(y_true)
    else:
        total_items = sum(class_counts.values())
    
    sum_pred_proximities = 0
    sum_true_proximities = 0
    
    for true_label, pred_label in zip(y_true, y_pred):
        pred_prox = _calculate_proximity(pred_label, true_label, class_counts, total_items)
        sum_pred_proximities += pred_prox
        
        true_prox = _calculate_proximity(true_label, true_label, class_counts, total_items)
        sum_true_proximities += true_prox
    
    if sum_true_proximities == 0:
        return 0.0
    
    return sum_pred_proximities / sum_true_proximities

def spearman_correlation(y_true, y_pred):
    """
    Calculate Spearman rank correlation for ordinal regression.
    
    Spearman (1904)'s rank correlation measures the monotonic relationship between predicted and
    true labels. It's particularly useful for ordinal data as it only considers
    the ranking of values, not their absolute differences.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    y_pred : array-like of shape (n_samples,)
        Predicted ordinal labels
        
    Returns
    -------
    float
        Spearman rank correlation coefficient between -1 and 1, where:
        - 1 indicates perfect positive correlation
        - 0 indicates no correlation
        - -1 indicates perfect negative correlation
    """
    correlation, _ = spearmanr(y_true, y_pred)
    return correlation

def kendall_tau(y_true, y_pred):
    """
    Calculate Kendall's Tau correlation coefficient for ordinal data.
    
    Kendall(1945)'s Tau-b measures the ordinal association between two rankings. It's
    particularly suitable for ordinal data as it considers the concordance of
    pairs of observations and the number of tied ranks.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    y_pred : array-like of shape (n_samples,)
        Predicted ordinal labels
        
    Returns
    -------
    float
        Kendall's Tau correlation coefficient between -1 and 1, where:
        - 1 indicates perfect agreement in rankings
        - 0 indicates no association between rankings
        - -1 indicates perfect disagreement in rankings
    """
    correlation, _ = kendalltau(y_true, y_pred)
    return correlation

def _create_one_hot_encoding(y_true, n_classes=None, zero_indexed = False):
    """
    Create one-hot encoding for ordinal labels.
    
    This helper function converts ordinal labels to one-hot encoded format,
    handling arbitrary label ranges by shifting to 0-based indexing.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    n_classes : int, optional
        Number of classes. If None, inferred from unique labels.
        
    Returns
    -------
    tuple
        (one_hot_matrix, min_label, n_classes)
        - one_hot_matrix: 2D array of shape (n_samples, n_classes)
        - min_label: Minimum label value in the original data
        - n_classes: Number of unique classes
    """
    y_true = np.asarray(y_true)
    unique_labels = np.unique(y_true)
    if zero_indexed:
        min_label = 0
    else:
        min_label = np.min(unique_labels)
    
    if n_classes is None:
        n_classes = len(unique_labels)
    
    n_samples = len(y_true)
    y_true_one_hot = np.zeros((n_samples, n_classes))
    
    shifted_labels = y_true - min_label
    for i, label in enumerate(shifted_labels):
        y_true_one_hot[i, int(label)] = 1
    
    return y_true_one_hot, min_label, n_classes

def ranked_probability_score(y_true, y_pred_proba, zero_indexed = False):
    """
    Calculate Ranked Probability Score (RPS) for ordinal regression.
    
    Epstein (1969)'s Ranked Probability Score (RPS) evaluates probabilistic predictions for ordinal data by comparing the
    cumulative predicted probabilities with the cumulative observed probabilities.
    It penalizes predictions that deviate from the true class distribution.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    y_pred_proba : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class
        
    Returns
    -------
    float
        Ranked Probability Score, where:
        - 0 indicates perfect predictions
        - Higher values indicate worse predictions
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    y_true_one_hot, _, _ = _create_one_hot_encoding(y_true, n_classes=y_pred_proba.shape[1], zero_indexed = zero_indexed)
    
    y_pred_cumsum = np.cumsum(y_pred_proba, axis=1)
    y_true_cumsum = np.cumsum(y_true_one_hot, axis=1)
    
    rps = np.mean(np.sum((y_pred_cumsum - y_true_cumsum) ** 2, axis=1))
    
    return rps

def ordinal_weighted_ce(y_true, y_pred_proba, alpha=1, zero_indexed = False):
    """
    Calculate ordinal weighted cross-entropy loss.
    
    This loss function extends standard cross-entropy to account for the ordinal
    nature of the data by weighting the loss based on the distance between
    predicted and true classes, see Polat et al. (2025). Also known as ordinal log loss (Castagnos et al. (2022)).
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    y_pred_proba : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class
    alpha : float, default=1
        Exponent for the absolute difference. Higher values increase the penalty
        for predictions far from the true class.
        
    Returns
    -------
    float
        Loss value, where:
        - Lower values indicate better predictions
        - The loss is always non-negative
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    n_samples, n_classes = y_pred_proba.shape
    eps = 1e-15  # To avoid log(0)

    if zero_indexed:
        min_label = 0
    else:
        min_label = np.min(y_true)
    y_true_shifted = y_true - min_label

    loss = 0.0
    for i in range(n_samples):
        for k in range(n_classes):
            pi_k = np.clip(y_pred_proba[i, k], eps, 1 - eps)
            loss += (np.log(1 - pi_k) * np.power(abs(k - y_true_shifted[i]),alpha))
    loss = -loss / n_samples
    return loss

def adjacent_accuracy(y_true, y_pred):
    """
    Calculate Adjacent Accuracy for ordinal regression.
    
    Adjacent accuracy measures the proportion of predictions that are either
    correct or off by one class. This is particularly useful for ordinal data
    where predictions close to the true class are more acceptable than those
    far away.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    y_pred : array-like of shape (n_samples,)
        Predicted ordinal labels
        
    Returns
    -------
    float
        Adjacent accuracy score between 0 and 1, where:
        - 1 indicates all predictions are either correct or off by one class
        - 0 indicates all predictions are off by more than one class
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    correct_or_adjacent = np.sum(np.abs(y_true - y_pred) <= 1)
    
    return correct_or_adjacent / len(y_true)

def evaluate_ordinal_model(y_true, y_pred, y_pred_proba=None, metrics=None, class_counts=None, zero_indexed = False):
    """
    Evaluate an ordinal regression model using multiple metrics.
    
    This function computes a comprehensive set of evaluation metrics for ordinal
    regression models, including both hard-label metrics and probability-based metrics
    if probability predictions are available.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True ordinal labels
    y_pred : array-like of shape (n_samples,)
        Predicted ordinal labels
    y_pred_proba : array-like of shape (n_samples, n_classes), optional
        Predicted class probabilities
    metrics : list of str, optional
        List of metric names to compute. If None, all available metrics are used.
    class_counts : dict, optional
        Dictionary mapping class labels to their counts. If None, calculated from y_true.
        Useful for local explanations where class distribution might differ from training.
    zero_indexed : bool, optional
        Whether the labels are zero-indexed. If False, the labels are shifted to zero-indexed.
    Returns
    -------
    dict
        Dictionary containing evaluation results for each metric
        
    Notes
    -----
    The function automatically selects appropriate metrics based on the available
    predictions. Probability-based metrics are only computed if y_pred_proba is provided.
    """
    available_hard_metrics = {
        'accuracy': accuracy,
        'adjacent_accuracy': adjacent_accuracy,
        'mze': mze,
        'mae': mae,
        'mse': mse,
        'weighted_kappa_quadratic': lambda yt, yp: weighted_kappa(yt, yp, weights='quadratic'),
        'weighted_kappa_linear': lambda yt, yp: weighted_kappa(yt, yp, weights='linear'),
        'cem': lambda yt, yp: cem(yt, yp, class_counts=class_counts),
        'spearman_correlation': spearman_correlation,
        'kendall_tau': kendall_tau,
    }
    available_proba_metrics = {
        'ranked_probability_score':     lambda yt, yp: ranked_probability_score(yt, yp, zero_indexed=zero_indexed),
        'ordinal_weighted_ce_linear': lambda yt, yp: ordinal_weighted_ce(yt, yp, alpha=1, zero_indexed=zero_indexed),
        'ordinal_weighted_ce_quadratic': lambda yt, yp: ordinal_weighted_ce(yt, yp, alpha=2, zero_indexed=zero_indexed),
    }
    
    if metrics is None:
        metrics = list(available_hard_metrics.keys()) + list(available_proba_metrics.keys())
    
    results = {}
    
    for metric, func in available_hard_metrics.items():
        if metric in metrics:
            try:
                results[metric] = func(y_true, y_pred)
            except Exception as e:
                print(f"Warning: Could not calculate {metric}: {e}")
    
    if y_pred_proba is not None:
        try:
            y_pred_proba = np.asarray(y_pred_proba)
            if len(y_pred_proba.shape) == 1:
                y_true_one_hot, min_label, n_classes = _create_one_hot_encoding(y_true)
                one_hot = np.zeros((len(y_pred), n_classes))
                for i, pred in enumerate(y_pred):
                    one_hot[i, int(pred - min_label)] = y_pred_proba[i]
                y_pred_proba = one_hot
            row_sums = y_pred_proba.sum(axis=1)
            if not np.allclose(row_sums, 1.0):
                y_pred_proba = y_pred_proba / row_sums[:, np.newaxis]
        except Exception as e:
            print(f"Warning: Could not preprocess y_pred_proba: {e}")
            y_pred_proba = None
        
        if y_pred_proba is not None:
            for metric, func in available_proba_metrics.items():
                if metric in metrics:
                    try:
                        results[metric] = func(y_true, y_pred_proba)
                    except Exception as e:
                        print(f"Warning: Could not calculate {metric}: {e}")
    
    return results

def print_evaluation_results(results):
    """
    Print evaluation results in a formatted way.
    
    This function provides a clear, formatted output of the evaluation metrics,
    grouping them into hard label metrics and probability-based metrics.
    
    Parameters
    ----------
    results : dict
        Dictionary containing evaluation metrics as returned by evaluate_ordinal_model
        
    Notes
    -----
    - Metrics are printed with 4 decimal places
    - Hard label metrics are printed first, followed by probability-based metrics
    - Metric names are formatted for better readability
    """
    print("\nOrdinal Regression Evaluation Results:")
    print("-" * 50)
    
    print("Hard Label Metrics:")
    for metric in ['accuracy', 'adjacent_accuracy', 'mze', 'mae', 'mse', 'weighted_kappa_quadratic', 'weighted_kappa_linear', 'cem', 
                  'spearman_correlation', 'kendall_tau']:
        if metric in results:
            print(f"  {metric.replace('_', ' ').title()}: {results[metric]:.4f}")
    
    if 'ranked_probability_score' in results:
        print("\nProbability-Based Metrics:")
        for metric in ['ranked_probability_score', 'ordinal_weighted_ce_linear', 'ordinal_weighted_ce_quadratic']:
            if metric in results:
                print(f"  {metric.replace('_', ' ').title()}: {results[metric]:.4f}")
    
    print("-" * 50) 