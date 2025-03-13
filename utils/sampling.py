import numpy as np
from numba import njit

@njit(fastmath=True)
def uniform_distribution(n):
    return np.ones(n) / n

@njit(fastmath=True)
def normalize_weights(weights, min_weight=None):
    sum_w = np.sum(weights)

    if sum_w <= 0:
        return np.ones(len(weights)) / len(weights)
    norm = weights / sum_w

    if min_weight is not None and min_weight > 0:
        if min_weight * len(weights) >= 1.0:
            return np.ones(len(weights)) / len(weights)

        max_iter = 10
        iter = 0

        while True:
            below = norm < min_weight
            count = np.sum(below)
            if count == 0 or iter >= max_iter:
                break

            total_min = count * min_weight
            if total_min >= 1.0:
                return np.ones(len(weights)) / len(weights)

            norm[below] = min_weight
            above = ~below
            sum_above = np.sum(norm[above])

            if sum_above > 0:
                scale = (1.0 - total_min) / sum_above
                norm[above] *= scale
            iter += 1
        norm = norm / np.sum(norm)
    return norm

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

    p_norm = p / np.sum(p)
    return np.random.choice(indices, size=size, replace=True, p=p_norm)