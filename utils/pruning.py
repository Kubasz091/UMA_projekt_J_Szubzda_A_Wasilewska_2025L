import numpy as np
from utils.prediction import accuracy

def calculate_error(tree, X, y):
    if len(y) == 0:
        return 0.0
    predictions = tree.predict(X)
    return np.sum(predictions != y) / len(y)

def evaluate_prune_candidate(tree, node, X, y):
    """
    Check if pruning a specific node improves accuracy.

    Args:
        tree: DecisionTree containing the node
        node: Node to consider pruning
        X: Feature matrix
        y: Target labels

    Returns:
        Tuple of (should_prune, error_before, error_after)
    """
    # Skip if node is already a leaf
    if node.value is not None:
        return False, 0.0, 0.0

    # Calculate error with current subtree
    error_before = calculate_error(tree, X, y)

    # Store original node attributes
    original_left = node.left
    original_right = node.right
    original_feature = node.feature_idx
    original_threshold = node.threshold

    # Determine majority class
    if len(y) > 0:
        unique_vals, counts = np.unique(y, return_counts=True)
        majority_class = unique_vals[np.argmax(counts)]
    else:
        majority_class = 0  # Default if no data

    # Convert node to leaf
    node.left = None
    node.right = None
    node.feature_idx = None
    node.threshold = None
    node.value = majority_class

    # Calculate error after pruning
    error_after = calculate_error(tree, X, y)

    # Check if pruning improves or maintains accuracy
    should_prune = error_after <= error_before

    # Restore node if pruning doesn't help
    if not should_prune:
        node.left = original_left
        node.right = original_right
        node.feature_idx = original_feature
        node.threshold = original_threshold
        node.value = None

    return should_prune, error_before, error_after

def prune_tree(tree, X_val, y_val):
    """
    Prune a decision tree to prevent overfitting.

    Args:
        tree: DecisionTree to prune
        X_val: Validation feature matrix
        y_val: Validation target labels

    Returns:
        Pruned DecisionTree
    """
    def _prune_node(node):
        # Base case - leaf node
        if node.value is not None:
            return node

        # Recursively prune children first (bottom-up)
        if node.left is not None:
            node.left = _prune_node(node.left)

        if node.right is not None:
            node.right = _prune_node(node.right)

        # Evaluate pruning this node
        should_prune, _, _ = evaluate_prune_candidate(tree, node, X_val, y_val)

        # Node is already modified if should_prune is True

        return node

    # Start pruning from the root
    if tree.root is not None:
        tree.root = _prune_node(tree.root)

    return tree