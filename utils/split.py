import numpy as np
from numba import njit
from sklearn.feature_selection import mutual_info_classif
from math import floor

@njit(fastmath=True)
def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    entropy = 0.0
    for prob in probabilities:
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy

@njit(fastmath=True)
def calculate_gini_impurity(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities**2)

def split_node(X, y, feature_idx, threshold):
    left_mask = X[:, feature_idx] <= threshold
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[~left_mask], y[~left_mask]
    return X_left, y_left, X_right, y_right

def calculate_information_gain(X, y):
    return mutual_info_classif(X, y, discrete_features=False, random_state=42)

def find_best_split(X, y, feature_indices, criterion='gini', min_samples_leaf=1, max_unique_for_exact=10):
    n_samples, _ = X.shape
    if n_samples <= 1:
        return None, None, 0.0, None, None

    # Select impurity function based on criterion
    if criterion == 'entropy':
        parent_impurity = calculate_entropy(y)
        calculate_impurity = calculate_entropy
    else:  # default to gini
        parent_impurity = calculate_gini_impurity(y)
        calculate_impurity = calculate_gini_impurity

    # Initialize best values
    best_score = 0.0
    best_feature_idx = None
    best_threshold = None
    best_left_indices = None
    best_right_indices = None

    # Evaluate each feature
    for feature_idx in feature_indices:
        feature_values = X[:, feature_idx]

        # Skip feature if all values are identical
        if len(np.unique(feature_values)) <= 1:
            continue

        # For categorical/few unique values vs. continuous features
        unique_values = np.unique(feature_values)
        if len(unique_values) < max_unique_for_exact:
            # Use midpoints between sorted unique values
            thresholds = unique_values[:-1] + np.diff(unique_values)/2
        else:
            step_size = max(1, floor(100 * 100 / max_unique_for_exact) / 100)

            # Generate percentiles with the calculated step size
            percentiles = np.percentile(feature_values, np.arange(floor(step_size/2), 100, step_size))
            thresholds = np.unique(percentiles)

        # Evaluate each threshold
        for threshold in thresholds:
            left_indices = feature_values <= threshold
            right_indices = ~left_indices

            # Skip if split doesn't meet min_samples_leaf
            if np.sum(left_indices) < min_samples_leaf or np.sum(right_indices) < min_samples_leaf:
                continue

            # Calculate impurity for children
            left_impurity = calculate_impurity(y[left_indices])
            right_impurity = calculate_impurity(y[right_indices])

            # Calculate information gain
            n_left, n_right = np.sum(left_indices), np.sum(right_indices)
            weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
            information_gain = parent_impurity - weighted_impurity

            # Update best values if this split is better
            if information_gain > best_score:
                best_score = information_gain
                best_feature_idx = feature_idx
                best_threshold = threshold
                best_left_indices = left_indices
                best_right_indices = right_indices

    return best_feature_idx, best_threshold, best_score, best_left_indices, best_right_indices