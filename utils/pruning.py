import numpy as np
from utils.prediction import accuracy

def calculate_error(tree, X, y):
    if len(y) == 0:
        return 0.0
    predictions = tree.predict(X)
    return np.sum(predictions != y) / len(y)

def evaluate_prune_candidate(tree, node, X, y):
    if node.value is not None:
        return False, 0.0, 0.0

    err_before = calculate_error(tree, X, y)

    orig_left = node.left
    orig_right = node.right
    orig_feature = node.feature_idx
    orig_threshold = node.threshold

    if len(y) > 0:
        unique_vals, counts = np.unique(y, return_counts=True)
        maj_class = unique_vals[np.argmax(counts)]
    else:
        maj_class = 0  # default

    node.left = None
    node.right = None
    node.feature_idx = None
    node.threshold = None
    node.value = maj_class

    err_after = calculate_error(tree, X, y)

    should_prune = err_after <= err_before

    if not should_prune:
        node.left = orig_left
        node.right = orig_right
        node.feature_idx = orig_feature
        node.threshold = orig_threshold
        node.value = None

    return should_prune, err_before, err_after

def prune_tree(tree, X_val, y_val):
    def _prune_node(node):
        if node.value is not None:
            return node

        if node.left is not None:
            node.left = _prune_node(node.left)

        if node.right is not None:
            node.right = _prune_node(node.right)

        should_prune, _, _ = evaluate_prune_candidate(tree, node, X_val, y_val)

        return node

    if tree.root is not None:
        tree.root = _prune_node(tree.root)

    return tree