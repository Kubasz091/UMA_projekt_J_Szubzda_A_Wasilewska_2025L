import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tree import Node, DecisionTree
from utils.pruning import calculate_error, evaluate_prune_candidate, prune_tree

# ------------------- Fixtures -------------------

@pytest.fixture
def simple_tree():
    """
    Creates a simple tree with the structure:
              Root (feature 0, threshold 5)
             /                          \
    Left (feature 1, threshold 2)     Right Leaf (value=1)
    /                   \
    LL (value=0)        LR (value=1)
    """
    # Create the tree structure
    ll_node = Node(value=0)
    lr_node = Node(value=1)
    left_node = Node(feature_idx=1, threshold=2, left=ll_node, right=lr_node)
    right_node = Node(value=1)

    root = Node(feature_idx=0, threshold=5, left=left_node, right=right_node)

    tree = DecisionTree()
    tree.root = root

    return tree

@pytest.fixture
def perfect_tree():
    """Creates a tree that perfectly classifies the test data."""
    # Create a simple decision tree
    left_node = Node(value=0)
    right_node = Node(value=1)
    root = Node(feature_idx=0, threshold=5, left=left_node, right=right_node)

    tree = DecisionTree()
    tree.root = root

    return tree

@pytest.fixture
def simple_data():
    """Simple classification data."""
    X = np.array([
        [2, 1],  # classified as 0 (left subtree, left leaf)
        [2, 3],  # classified as 1 (left subtree, right leaf)
        [8, 1]   # classified as 1 (right subtree)
    ])
    y = np.array([0, 1, 1])
    return X, y

@pytest.fixture
def perfect_data():
    """Data that is perfectly classified by the perfect_tree."""
    X = np.array([
        [2, 1],
        [3, 2],
        [4, 3],
        [6, 1],
        [7, 2],
        [8, 3]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y

@pytest.fixture
def validation_data():
    """Validation data for pruning evaluation."""
    X_val = np.array([
        [2, 1],
        [2, 3],
        [8, 1],
        [3, 1],
        [3, 3],
        [9, 1],
        [2, 3]
    ])
    y_val = np.array([0, 1, 1, 0, 0, 1, 1])
    return X_val, y_val

@pytest.fixture
def overfitted_tree():
    """
    Creates a tree that is overfitted and should be pruned.
    This tree has unnecessary splits that hurt generalization.
    """
    # Create a more complex tree structure that overfits
    lll_node = Node(value=0)
    llr_node = Node(value=0)
    # Change feature_idx=2 to feature_idx=0 to stay within bounds
    ll_node = Node(feature_idx=0, threshold=0.5, left=lll_node, right=llr_node)

    lr_node = Node(value=1)
    left_node = Node(feature_idx=1, threshold=2, left=ll_node, right=lr_node)

    rl_node = Node(value=1)
    rr_node = Node(value=1)
    right_node = Node(feature_idx=1, threshold=3, left=rl_node, right=rr_node)

    root = Node(feature_idx=0, threshold=5, left=left_node, right=right_node)

    tree = DecisionTree()
    tree.root = root

    return tree

# ------------------- Tests for calculate_error -------------------

def test_calculate_error_perfect(perfect_tree, perfect_data):
    """Test error calculation with perfect predictions."""
    X, y = perfect_data
    error = calculate_error(perfect_tree, X, y)
    assert error == 0.0

def test_calculate_error_mixed(simple_tree, simple_data):
    """Test error calculation with mixed predictions."""
    X, y = simple_data

    # Create modified labels to introduce errors
    modified_y = np.array([1, 0, 0])  # All predictions will be wrong

    error = calculate_error(simple_tree, X, modified_y)
    assert error == 1.0  # All predictions are wrong

    # Test with a mix of correct and incorrect predictions
    mixed_y = np.array([0, 0, 1])  # Only the middle prediction is wrong
    error = calculate_error(simple_tree, X, mixed_y)
    assert error == 1/3  # 1 out of 3 predictions is wrong

def test_calculate_error_empty():
    """Test error calculation with empty data."""
    tree = DecisionTree()
    tree.root = Node(value=0)  # Default prediction is 0

    X = np.array([]).reshape(0, 2)
    y = np.array([])

    error = calculate_error(tree, X, y)
    assert error == 0.0  # No data means no errors

# ------------------- Tests for evaluate_prune_candidate -------------------

def test_evaluate_prune_candidate_already_leaf():
    """Test evaluating pruning for a node that's already a leaf."""
    # Create a tree with a single leaf node
    tree = DecisionTree()
    tree.root = Node(value=1)

    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 1])

    # Attempt to evaluate pruning the leaf node
    should_prune, error_before, error_after = evaluate_prune_candidate(tree, tree.root, X, y)

    assert should_prune == False
    assert error_before == 0.0
    assert error_after == 0.0

def test_evaluate_prune_candidate_should_prune(simple_tree, validation_data):
    """Test evaluating a node that should be pruned."""
    X_val, y_val = validation_data

    # Modify the validation data so that pruning the left node improves accuracy
    modified_y_val = np.array([0, 0, 1, 0, 0, 1, 0])  # All left subtree samples should be class 0

    # Get the left node (which has split but could be pruned to a leaf)
    left_node = simple_tree.root.left

    # Evaluate pruning
    should_prune, error_before, error_after = evaluate_prune_candidate(
        simple_tree, left_node, X_val, modified_y_val
    )

    assert should_prune == True
    assert error_after <= error_before

def test_evaluate_prune_candidate_should_not_prune(simple_tree, validation_data):
    """Test evaluating a node that should not be pruned."""
    X_val, y_val = validation_data

    # Use the original validation data where the split is beneficial

    # Get the left node (which has split that is useful)
    left_node = simple_tree.root.left

    # Evaluate pruning
    should_prune, error_before, error_after = evaluate_prune_candidate(
        simple_tree, left_node, X_val, y_val
    )

    print(error_before, "   ", error_after)

    assert should_prune == False
    assert error_before < error_after

def test_evaluate_prune_candidate_empty_data():
    """Test pruning evaluation with empty data."""
    tree = DecisionTree()
    tree.root = Node(feature_idx=0, threshold=5,
                    left=Node(value=0),
                    right=Node(value=1))

    X = np.array([]).reshape(0, 2)
    y = np.array([])

    should_prune, error_before, error_after = evaluate_prune_candidate(
        tree, tree.root, X, y
    )

    # With no data, pruning should either maintain or improve error (both 0)
    assert error_before == 0.0
    assert error_after == 0.0

# ------------------- Tests for prune_tree -------------------

def test_prune_tree_should_prune(overfitted_tree, validation_data):
    """Test pruning a tree that should be pruned."""
    X_val, y_val = validation_data

    # Save the structure before pruning
    has_subtrees_before = (
        overfitted_tree.root.left.left.left is not None or
        overfitted_tree.root.right.left is not None
    )

    # Count nodes before pruning
    def count_nodes(node):
        if node is None:
            return 0
        return 1 + count_nodes(node.left) + count_nodes(node.right)

    nodes_before = count_nodes(overfitted_tree.root)

    # Run pruning
    pruned_tree = prune_tree(overfitted_tree, X_val, y_val)

    # Check that pruning happened
    nodes_after = count_nodes(pruned_tree.root)

    assert nodes_after < nodes_before

    # Verify that accuracy is maintained or improved
    error_before = calculate_error(overfitted_tree, X_val, y_val)
    error_after = calculate_error(pruned_tree, X_val, y_val)
    assert error_after <= error_before

def test_prune_tree_efficient_structure():
    """Test that pruning produces an efficient tree structure."""
    # Create a tree with redundant splits
    tree = DecisionTree()

    # Create a tree where right subtree has identical leaves and should be pruned
    root = Node(feature_idx=0, threshold=5,
               left=Node(value=0),
               right=Node(feature_idx=1, threshold=2,
                          left=Node(value=1),
                          right=Node(value=1)))

    tree.root = root

    # Simple validation data
    X_val = np.array([[2, 1], [8, 1], [8, 3]])
    y_val = np.array([0, 1, 1])

    # Run pruning
    pruned_tree = prune_tree(tree, X_val, y_val)

    # Right node should be pruned to a leaf since both children have same value
    assert pruned_tree.root.right.value == 1
    assert pruned_tree.root.right.left is None and pruned_tree.root.right.right is None

def test_prune_empty_tree():
    """Test pruning an empty tree."""
    tree = DecisionTree()
    tree.root = None

    X_val = np.array([[1, 2], [3, 4]])
    y_val = np.array([0, 1])

    pruned_tree = prune_tree(tree, X_val, y_val)

    # Should handle gracefully and return still-empty tree
    assert pruned_tree.root is None

def test_prune_tree_just_root():
    """Test pruning a tree with just a root node."""
    tree = DecisionTree()
    tree.root = Node(value=1)  # Root is already a leaf

    X_val = np.array([[1, 2], [3, 4]])
    y_val = np.array([1, 1])

    pruned_tree = prune_tree(tree, X_val, y_val)

    # Should remain unchanged
    assert pruned_tree.root.value == 1
    assert pruned_tree.root.left is None and pruned_tree.root.right is None

def test_prune_tree_accuracy_improvement():
    """Test that pruning can improve accuracy on validation data."""
    # Create a tree that overfits training data
    tree = DecisionTree()

    # This tree memorizes individual points rather than generalizing
    root = Node(feature_idx=0, threshold=3,
                left=Node(feature_idx=1, threshold=2,
                         left=Node(value=0),
                         right=Node(value=1)),
                right=Node(feature_idx=1, threshold=5,
                          left=Node(value=1),
                          right=Node(value=0)))

    tree.root = root

    # Validation data with a simpler pattern: x[0] < 5 -> class 0, else class 1
    X_val = np.array([
        [2, 3], [3, 7], [4, 1],  # Should be class 0
        [6, 2], [7, 8], [8, 4]   # Should be class 1
    ])
    y_val = np.array([0, 0, 0, 1, 1, 1])

    # Calculate initial error
    initial_error = calculate_error(tree, X_val, y_val)

    # Prune the tree
    pruned_tree = prune_tree(tree, X_val, y_val)

    # Calculate error after pruning
    final_error = calculate_error(pruned_tree, X_val, y_val)

    # Check that error decreased or stayed the same
    assert final_error <= initial_error

# Add more specific test cases as needed

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])