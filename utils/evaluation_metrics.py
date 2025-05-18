import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from scipy.stats import kendalltau

def accuracy(y_true, y_pred):
    """
    Calculate accuracy for ordinal regression.
    
    Parameters:
    -----------
    y_true : array-like
        True ordinal labels
    y_pred : array-like
        Predicted ordinal labels
        
    Returns:
    --------
    float
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)

def mze(y_true, y_pred):
    """
    Calculate Mean Zero-One Error (MZE) for ordinal regression.
    MZE = 1 - accuracy
    
    Parameters:
    -----------
    y_true : array-like
        True ordinal labels
    y_pred : array-like
        Predicted ordinal labels
        
    Returns:
    --------
    float
        Mean Zero-One Error
    """
    return 1 - accuracy_score(y_true, y_pred)

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) for ordinal regression.
    
    Parameters:
    -----------
    y_true : array-like
        True ordinal labels
    y_pred : array-like
        Predicted ordinal labels
        
    Returns:
    --------
    float
        Mean Absolute Error
    """
    return mean_absolute_error(y_true, y_pred)

def mse(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) for ordinal regression.
    
    Parameters:
    -----------
    y_true : array-like
        True ordinal labels
    y_pred : array-like
        Predicted ordinal labels
        
    Returns:
    --------
    float
        Mean Squared Error
    """
    return mean_squared_error(y_true, y_pred)

def weighted_kappa(y_true, y_pred, weights='quadratic'):
    """
    Calculate weighted kappa for ordinal regression.
    
    Parameters:
    -----------
    y_true : array-like
        True ordinal labels
    y_pred : array-like
        Predicted ordinal labels
    weights : str, optional
        Weighting scheme for the confusion matrix. Options: 'linear', 'quadratic', 'none'
        
    Returns:
    --------
    float
        Weighted kappa score
    """
    return cohen_kappa_score(y_true, y_pred, weights=weights)

def _get_class_counts(y):
    """
    Calculate the count of items per class.
    
    Parameters:
    -----------
    y : array-like
        Array of class labels
        
    Returns:
    --------
    dict
        Dictionary mapping class labels to their counts
    """
    unique_labels, counts = np.unique(y, return_counts=True)
    return dict(zip(unique_labels, counts))

def _calculate_proximity(c1, c2, class_counts, total_items):
    """
    Calculate proximity between two classes.
    
    Parameters:
    -----------
    c1 : int
        First class label
    c2 : int
        Second class label
    class_counts : dict
        Dictionary mapping class labels to their counts
    total_items : int
        Total number of items
        
    Returns:
    --------
    float
        Proximity value between the two classes
    """
    if c1 == c2:
        # For correct predictions, use the same formula but with c1=c2
        return -np.log(class_counts[c1] / (2 * total_items))
    
    # Ensure c1 < c2
    if c1 > c2:
        c1, c2 = c2, c1
    
    # Calculate sum of counts between c1 and c2 (excluding c1)
    sum_counts = 0
    for k in range(c1 + 1, c2 + 1):
        sum_counts += class_counts.get(k, 0)
    
    # Calculate proximity
    return -np.log((class_counts[c1] / 2 + sum_counts) / total_items)

def cem(y_true, y_pred, class_counts=None):
    """
    Calculate Closeness Evaluation Measure (CEM) for ordinal classification.
    Optionally use class_counts from a reference dataset (for local explanations).
    
    Parameters:
    -----------
    y_true : array-like
        True ordinal labels (gold standard)
    y_pred : array-like
        Predicted ordinal labels (system predictions)
    class_counts : dict, optional
        Dictionary mapping class labels to their counts. If None, calculated from y_true.
        
    Returns:
    --------
    float
        CEM score between 0 and 1, where higher values indicate better performance
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    
    # Get class counts and total number of items
    if class_counts is None:
        class_counts = _get_class_counts(y_true)
        total_items = len(y_true)
    else:
        total_items = sum(class_counts.values())
    
    # Calculate sum of proximities for predictions
    sum_pred_proximities = 0
    sum_true_proximities = 0
    
    for true_label, pred_label in zip(y_true, y_pred):
        # Calculate proximity for prediction
        pred_prox = _calculate_proximity(pred_label, true_label, class_counts, total_items)
        sum_pred_proximities += pred_prox
        
        # Calculate proximity for correct prediction (for normalization)
        true_prox = _calculate_proximity(true_label, true_label, class_counts, total_items)
        sum_true_proximities += true_prox
    
    # Calculate final CEM score
    if sum_true_proximities == 0:
        return 0.0
    
    return sum_pred_proximities / sum_true_proximities

def spearman_correlation(y_true, y_pred):
    """
    Calculate Spearman rank correlation for ordinal regression.
    
    Parameters:
    -----------
    y_true : array-like
        True ordinal labels
    y_pred : array-like
        Predicted ordinal labels
        
    Returns:
    --------
    float
        Spearman rank correlation coefficient
    """
    correlation, _ = spearmanr(y_true, y_pred)
    return correlation

def kendall_tau(y_true, y_pred):
    """
    Calculate Kendall's Tau correlation coefficient for ordinal data.
    This measures the ordinal association between two rankings.
    
    Parameters:
    -----------
    y_true : array-like
        True ordinal labels
    y_pred : array-like
        Predicted ordinal labels
        
    Returns:
    --------
    float
        Kendall's Tau correlation coefficient between -1 and 1
        where 1 indicates perfect agreement and -1 indicates perfect disagreement
    """
    correlation, _ = kendalltau(y_true, y_pred)
    return correlation

def _create_one_hot_encoding(y_true, n_classes=None):
    """
    Create one-hot encoding for ordinal labels with arbitrary label range.
    
    Parameters:
    -----------
    y_true : array-like
        True ordinal labels
    n_classes : int, optional
        Number of classes. If None, inferred from unique labels.
        
    Returns:
    --------
    tuple
        (one_hot_matrix, min_label, n_classes)
    """
    y_true = np.asarray(y_true)
    unique_labels = np.unique(y_true)
    min_label = np.min(unique_labels)
    
    if n_classes is None:
        n_classes = len(unique_labels)
    
    n_samples = len(y_true)
    y_true_one_hot = np.zeros((n_samples, n_classes))
    
    # Shift labels to 0-based indexing for one-hot encoding
    shifted_labels = y_true - min_label
    for i, label in enumerate(shifted_labels):
        y_true_one_hot[i, int(label)] = 1
    
    return y_true_one_hot, min_label, n_classes

def ranked_probability_score(y_true, y_pred_proba):
    """
    Calculate Ranked Probability Score (RPS) for ordinal regression.
    RPS = sum((F_i - O_i)^2) where F_i is the cumulative predicted probability
    and O_i is the cumulative observed probability.
    
    Parameters:
    -----------
    y_true : array-like
        True ordinal labels
    y_pred_proba : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class
        
    Returns:
    --------
    float
        Ranked Probability Score
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Create one-hot encoding with proper label range
    y_true_one_hot, _, _ = _create_one_hot_encoding(y_true, n_classes=y_pred_proba.shape[1])
    
    # Calculate cumulative probabilities
    y_pred_cumsum = np.cumsum(y_pred_proba, axis=1)
    y_true_cumsum = np.cumsum(y_true_one_hot, axis=1)
    
    # Calculate RPS
    rps = np.mean(np.sum((y_pred_cumsum - y_true_cumsum) ** 2, axis=1))
    
    return rps

def ordinal_weighted_ce(y_true, y_pred_proba, alpha=1):
    """
    Calculate the custom ordinal weighted cross-entropy loss as per the provided formula:
    L(y, pi_hat) := -1/n * sum_i sum_k log(1 - pi_k^(i)) * |k - y^(i)|^alpha

    Parameters:
    -----------
    y_true : array-like
        True ordinal labels
    y_pred_proba : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class
    alpha : float, optional
        Exponent for the absolute difference (default=1)
    Returns:
    --------
    float
        Loss value
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    n_samples, n_classes = y_pred_proba.shape
    eps = 1e-15  # To avoid log(0)

    # Shift y_true to 0-based indexing if needed
    min_label = np.min(y_true)
    y_true_shifted = y_true - min_label

    loss = 0.0
    for i in range(n_samples):
        for k in range(n_classes):
            pi_k = np.clip(y_pred_proba[i, k], eps, 1 - eps)
            loss += np.log(1 - pi_k) * (abs(k - y_true_shifted[i]) ** alpha)
    loss = -loss / n_samples
    return loss

def adjacent_accuracy(y_true, y_pred):
    """
    Calculate Adjacent Accuracy for ordinal regression.
    This measures the proportion of predictions that are either correct or off by one class.
    
    Parameters:
    -----------
    y_true : array-like
        True ordinal labels
    y_pred : array-like
        Predicted ordinal labels
        
    Returns:
    --------
    float
        Adjacent Accuracy score between 0 and 1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Count predictions that are either correct or off by one class
    correct_or_adjacent = np.sum(np.abs(y_true - y_pred) <= 1)
    
    return correct_or_adjacent / len(y_true)

def evaluate_ordinal_model(y_true, y_pred, y_pred_proba=None, metrics=None):
    """
    Evaluate an ordinal regression model using multiple metrics.
    Parameters:
    -----------
    y_true : array-like
        True ordinal labels
    y_pred : array-like
        Predicted ordinal labels
    y_pred_proba : array-like of shape (n_samples, n_classes), optional
        Predicted probabilities for each class
    metrics : list of str, optional
        List of metrics to compute. If None, use all available metrics.
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    # Define all available metrics
    available_hard_metrics = {
        'accuracy': accuracy,
        'adjacent_accuracy': adjacent_accuracy,
        'mze': mze,
        'mae': mae,
        'mse': mse,
        'weighted_kappa_quadratic': lambda yt, yp: weighted_kappa(yt, yp, weights='quadratic'),
        'weighted_kappa_linear': lambda yt, yp: weighted_kappa(yt, yp, weights='linear'),
        'cem': cem,
        'spearman_correlation': spearman_correlation,
        'kendall_tau': kendall_tau,
    }
    available_proba_metrics = {
        'ranked_probability_score': ranked_probability_score,
        'ordinal_weighted_ce_linear': lambda yt, yp: ordinal_weighted_ce(yt, yp, alpha=1),
        'ordinal_weighted_ce_quadratic': lambda yt, yp: ordinal_weighted_ce(yt, yp, alpha=2),
    }
    # Default metrics if not specified
    if metrics is None:
        metrics = list(available_hard_metrics.keys()) + list(available_proba_metrics.keys())
    results = {}
    # Compute hard label metrics
    for metric, func in available_hard_metrics.items():
        if metric in metrics:
            try:
                results[metric] = func(y_true, y_pred)
            except Exception as e:
                print(f"Warning: Could not calculate {metric}: {e}")
    # Compute probability-based metrics
    if y_pred_proba is not None:
        # Ensure y_pred_proba is in the right shape
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
    
    Parameters:
    -----------
    results : dict
        Dictionary containing evaluation metrics
    """
    print("\nOrdinal Regression Evaluation Results:")
    print("-" * 50)
    
    # Print hard label metrics
    print("Hard Label Metrics:")
    for metric in ['accuracy', 'adjacent_accuracy', 'mze', 'mae', 'mse', 'weighted_kappa_quadratic', 'weighted_kappa_linear', 'cem', 
                  'spearman_correlation', 'kendall_tau']:
        if metric in results:
            print(f"  {metric.replace('_', ' ').title()}: {results[metric]:.4f}")
    
    # Print probability-based metrics if available
    if 'ranked_probability_score' in results:
        print("\nProbability-Based Metrics:")
        for metric in ['ranked_probability_score', 'ordinal_weighted_ce_linear', 'ordinal_weighted_ce_quadratic']:
            if metric in results:
                print(f"  {metric.replace('_', ' ').title()}: {results[metric]:.4f}")
    
    print("-" * 50) 