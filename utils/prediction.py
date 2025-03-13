import numpy as np
from numba import njit

@njit(fastmath=True)
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def majority_vote(predictions):
    if len(predictions.shape) == 1:
        return predictions
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

def weighted_majority_vote(predictions, weights, n_classes=None):
    if len(predictions) == 0:
        return np.array([])

    if len(predictions) == 1:
        return predictions[0]

    if weights is None or len(weights) != len(predictions):
        return majority_vote(predictions)

    predictions = np.asarray(predictions)

    n_trees, n_samples = predictions.shape

    if n_classes is None:
        n_classes = np.max(predictions) + 1

    weighted_votes = np.zeros((n_samples, n_classes))

    for tree_idx in range(n_trees):
        for sample_idx in range(n_samples):
            class_idx = predictions[tree_idx, sample_idx]
            weighted_votes[sample_idx, class_idx] += weights[tree_idx]

    return np.argmax(weighted_votes, axis=1)