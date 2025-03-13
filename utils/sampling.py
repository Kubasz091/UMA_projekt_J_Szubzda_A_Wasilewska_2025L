import numpy as np
from numba import njit

@njit(fastmath=True)
def uniform_distribution(n):
    return np.ones(n) / n

@njit(fastmath=True)
def normalize_weights(weights, min_weight=None):
    sum_weights = np.sum(weights)

    if sum_weights <= 0:
        return np.ones(len(weights)) / len(weights)

    normalized = weights / sum_weights

    if min_weight is not None and min_weight > 0:
        # Quick check if min_weight is too high
        if min_weight * len(weights) >= 1.0:
            return np.ones(len(weights)) / len(weights)

        # Iterative adjustment to ensure all weights >= min_weight
        max_iterations = 10  # Safety to prevent infinite loops and give acceptable precision
        iteration = 0

        while True:
            below_min = normalized < min_weight
            count_below = np.sum(below_min)

            if count_below == 0 or iteration >= max_iterations:
                break

            # Calculate total weight to be allocated to below-min elements
            total_min_weight = count_below * min_weight

            # If total min weight would exceed 1.0, use uniform distribution
            if total_min_weight >= 1.0:
                return np.ones(len(weights)) / len(weights)

            # Set all below-min values to min_weight
            normalized[below_min] = min_weight

            # Scale down above-min values proportionally
            above_min = ~below_min
            sum_above = np.sum(normalized[above_min])

            if sum_above > 0:
                scale_factor = (1.0 - total_min_weight) / sum_above
                normalized[above_min] *= scale_factor

            iteration += 1

        # Final normalization to ensure sum is exactly 1.0
        normalized = normalized / np.sum(normalized)

    return normalized

def random_selection(n_features, max_features):
    max_features = min(max_features, n_features)
    return np.random.choice(n_features, size=max_features, replace=False)

def weighted_random_selection(max_features, weights):
    n_features = len(weights)
    max_features = min(max_features, n_features)

    if np.sum(weights) == 0:
        return random_selection(n_features, max_features)

    probs = weights / np.sum(weights)
    return np.random.choice(n_features, size=max_features, replace=False, p=probs)

def sample_with_replacement(indices, size, p=None):
    if p is None:
        return np.random.choice(indices, size=size, replace=True)

    p_normalized = p / np.sum(p)
    return np.random.choice(indices, size=size, replace=True, p=p_normalized)