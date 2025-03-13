import numpy as np
from utils.split import find_best_split
from numba import njit

@njit(fastmath=True)
def is_pure(y):
    return len(np.unique(y)) == 1

@njit(fastmath=True)
def compute_leaf_value(y):
    if len(y) == 0:
        return 0
    unique_vals, counts = np.unique(y, return_counts=True)
    return unique_vals[np.argmax(counts)]

@njit(fastmath=True)
def check_stopping_criteria(depth, max_depth, n_samples, min_samples_split, n_classes):
    return (max_depth is not None and depth >= max_depth) or \
           (n_samples < min_samples_split) or \
           (n_classes <= 1)

class Node:
    """
    Tree node class for decision tree.
    Using __slots__ for memory optimization.
    """
    __slots__ = ('feature_idx', 'threshold', 'left', 'right', 'value')

    def __init__(self, feature_idx=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.value = value              # Value for leaf nodes (classification)

class DecisionTree:
    def __init__(self, max_depth=None, criterion='gini', min_samples_split=2,
                 min_samples_leaf=1, random_state=None):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.root = None
        self.feature_importances_ = None
        self.n_classes_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y, feature_indices=None):
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_classes_ = len(np.unique(y))
        _, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features)

        if feature_indices is None:
            feature_indices = np.arange(n_features)

        self.root = self._build_tree(X, y, feature_indices)

        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

        return self

    def _build_tree(self, X, y, feature_indices, depth=0):
        n_samples, _ = X.shape
        n_classes = len(np.unique(y))

        # Check stopping criteria
        if check_stopping_criteria(depth, self.max_depth, n_samples,
                                  self.min_samples_split, n_classes) or is_pure(y):
            return self._create_leaf(y)

        # Find best split
        feature_idx, threshold, info_gain, left_indices, right_indices = find_best_split(
            X, y, feature_indices, self.criterion, self.min_samples_leaf)

        # If no valid split found, create leaf
        if feature_idx is None or info_gain <= 0.0:
            return self._create_leaf(y)

        # Update feature importance
        if info_gain > 0:
            self.feature_importances_[feature_idx] += info_gain * n_samples

        # Recursively build left and right subtrees
        left_tree = self._build_tree(
            X[left_indices], y[left_indices], feature_indices, depth + 1
        )

        right_tree = self._build_tree(
            X[right_indices], y[right_indices], feature_indices, depth + 1
        )

        # Create and return internal node
        return self._create_node(feature_idx, threshold, left_tree, right_tree)

    def _create_leaf(self, y):
        value = compute_leaf_value(y)
        return Node(value=value)

    def _create_node(self, feature_idx, threshold, left, right):
        return Node(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left,
            right=right
        )

    def predict(self, X):
        X = np.asarray(X)

        if len(X.shape) == 1:
            return self._predict_sample(self.root, X)

        return np.array([self._predict_sample(self.root, sample) for sample in X])

    def _predict_sample(self, node, x):
        if node.value is not None:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(node.left, x)
        else:
            return self._predict_sample(node.right, x)

    def to_dict(self):
        def node_to_dict(node):
            if node is None:
                return None

            if node.value is not None:
                return {'value': int(node.value)}

            return {
                'feature_idx': int(node.feature_idx),
                'threshold': float(node.threshold),
                'left': node_to_dict(node.left),
                'right': node_to_dict(node.right)
            }

        return {
            'params': {
                'max_depth': self.max_depth,
                'criterion': self.criterion,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf
            },
            'tree': node_to_dict(self.root),
            'n_classes_': self.n_classes_,
            'feature_importances_': self.feature_importances_.tolist() if self.feature_importances_ is not None else None
        }

    @classmethod
    def from_dict(cls, data):
        def dict_to_node(node_dict):
            if node_dict is None:
                return None

            if 'value' in node_dict:
                return Node(value=node_dict['value'])

            return Node(
                feature_idx=node_dict['feature_idx'],
                threshold=node_dict['threshold'],
                left=dict_to_node(node_dict['left']),
                right=dict_to_node(node_dict['right'])
            )

        tree = cls(**data['params'])
        tree.root = dict_to_node(data['tree'])
        tree.n_classes_ = data['n_classes_']
        tree.feature_importances_ = np.array(data['feature_importances_']) if data['feature_importances_'] else None

        return tree