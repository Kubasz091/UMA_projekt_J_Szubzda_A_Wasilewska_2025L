import pytest
import numpy as np
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tree import is_pure, compute_leaf_value, check_stopping_criteria, Node, DecisionTree
from utils.split import find_best_split

# ------------------- Fixtures -------------------

@pytest.fixture
def binary_data():
    """Simple binary classification data."""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23, 24]
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y

@pytest.fixture
def multiclass_data():
    """Simple multi-class classification data."""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23, 24],
        [25, 26, 27]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return X, y

@pytest.fixture
def pure_data():
    """Data with all samples in the same class."""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    y = np.array([1, 1, 1])
    return X, y

@pytest.fixture
def single_sample_data():
    """Dataset with a single sample."""
    X = np.array([[1, 2, 3]])
    y = np.array([0])
    return X, y

@pytest.fixture
def empty_data():
    """Empty dataset."""
    X = np.array([]).reshape(0, 3)
    y = np.array([])
    return X, y

@pytest.fixture
def linearly_separable_data():
    """Simple linearly separable data for easy testing."""
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [10, 10],
        [11, 11],
        [12, 12],
        [13, 13],
        [14, 14]
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return X, y

@pytest.fixture
def sample_tree():
    """Creates a simple decision tree without using the DecisionTree class."""
    # Create a simple tree structure:
    #       Root (feature 0, threshold 7.5)
    #      /                       \
    # Left Leaf (value=0)        Right Leaf (value=1)

    left_node = Node(value=0)
    right_node = Node(value=1)
    root = Node(feature_idx=0, threshold=7.5, left=left_node, right=right_node)

    return root

# ------------------- Tests for helper functions -------------------

def test_is_pure_homogeneous():
    """Test is_pure with homogeneous class labels."""
    y = np.array([1, 1, 1, 1])
    assert is_pure(y) == True

def test_is_pure_heterogeneous():
    """Test is_pure with heterogeneous class labels."""
    y = np.array([0, 1, 0, 1])
    assert is_pure(y) == False

def test_is_pure_single_value():
    """Test is_pure with a single value."""
    y = np.array([5])
    assert is_pure(y) == True

def test_is_pure_empty():
    """Test is_pure with an empty array."""
    y = np.array([])
    assert is_pure(y) == True  # An empty array is technically pure

def test_compute_leaf_value_majority():
    """Test compute_leaf_value with majority class."""
    y = np.array([0, 1, 0, 1, 0])
    assert compute_leaf_value(y) == 0

def test_compute_leaf_value_tie():
    """Test compute_leaf_value with a tie (should choose first class)."""
    y = np.array([0, 1, 2, 0, 1, 2])
    assert compute_leaf_value(y) in [0, 1, 2]  # Implementation-dependent

def test_compute_leaf_value_single_value():
    """Test compute_leaf_value with single value."""
    y = np.array([5])
    assert compute_leaf_value(y) == 5

def test_compute_leaf_value_empty():
    """Test compute_leaf_value with empty array."""
    y = np.array([])
    assert compute_leaf_value(y) == 0  # Default for empty is 0

def test_check_stopping_criteria_max_depth():
    """Test stopping criteria based on max_depth."""
    assert check_stopping_criteria(depth=10, max_depth=10, n_samples=100, min_samples_split=2, n_classes=3) == True
    assert check_stopping_criteria(depth=9, max_depth=10, n_samples=100, min_samples_split=2, n_classes=3) == False

def test_check_stopping_criteria_min_samples():
    """Test stopping criteria based on min_samples_split."""
    assert check_stopping_criteria(depth=5, max_depth=10, n_samples=1, min_samples_split=2, n_classes=3) == True
    assert check_stopping_criteria(depth=5, max_depth=10, n_samples=2, min_samples_split=2, n_classes=3) == False

def test_check_stopping_criteria_n_classes():
    """Test stopping criteria based on number of classes."""
    assert check_stopping_criteria(depth=5, max_depth=10, n_samples=100, min_samples_split=2, n_classes=1) == True
    assert check_stopping_criteria(depth=5, max_depth=10, n_samples=100, min_samples_split=2, n_classes=2) == False

def test_check_stopping_criteria_none_max_depth():
    """Test stopping criteria with None max_depth."""
    assert check_stopping_criteria(depth=100, max_depth=None, n_samples=100, min_samples_split=2, n_classes=3) == False

# ------------------- Tests for Node class -------------------

def test_node_initialization():
    """Test basic node initialization."""
    leaf_node = Node(value=1)
    assert leaf_node.value == 1
    assert leaf_node.feature_idx is None
    assert leaf_node.threshold is None
    assert leaf_node.left is None
    assert leaf_node.right is None

    split_node = Node(feature_idx=0, threshold=5.0, left=leaf_node, right=None)
    assert split_node.feature_idx == 0
    assert split_node.threshold == 5.0
    assert split_node.left == leaf_node
    assert split_node.right is None
    assert split_node.value is None

# ------------------- Tests for DecisionTree class -------------------

def test_decision_tree_initialization():
    """Test DecisionTree initialization with different parameters."""
    tree = DecisionTree()
    assert tree.max_depth is None
    assert tree.criterion == 'gini'
    assert tree.min_samples_split == 2
    assert tree.min_samples_leaf == 1
    assert tree.random_state is None
    assert tree.root is None
    assert tree.feature_importances_ is None
    assert tree.n_classes_ is None

    tree_with_params = DecisionTree(max_depth=5, criterion='entropy',
                                   min_samples_split=10, min_samples_leaf=5,
                                   random_state=42)
    assert tree_with_params.max_depth == 5
    assert tree_with_params.criterion == 'entropy'
    assert tree_with_params.min_samples_split == 10
    assert tree_with_params.min_samples_leaf == 5
    assert tree_with_params.random_state == 42

def test_decision_tree_fit_basic(binary_data):
    """Test basic fitting of decision tree."""
    X, y = binary_data

    tree = DecisionTree(random_state=42)
    tree.fit(X, y)

    # Check that tree has been built
    assert tree.root is not None
    assert tree.n_classes_ == 2
    assert tree.feature_importances_ is not None
    assert len(tree.feature_importances_) == X.shape[1]

    # Sum of feature importances should be 1.0 or 0.0 (if no splits were made)
    assert np.isclose(np.sum(tree.feature_importances_), 1.0) or np.isclose(np.sum(tree.feature_importances_), 0.0)

def test_decision_tree_fit_multiclass(multiclass_data):
    """Test fitting with multiclass data."""
    X, y = multiclass_data

    tree = DecisionTree(random_state=42)
    tree.fit(X, y)

    assert tree.n_classes_ == 3

    # Predict and check classes
    predictions = tree.predict(X)
    assert set(np.unique(predictions)) <= set([0, 1, 2])

def test_decision_tree_fit_pure(pure_data):
    """Test fitting with pure class data."""
    X, y = pure_data

    tree = DecisionTree()
    tree.fit(X, y)

    # With pure data, should just create a leaf node
    assert tree.root.value == 1
    assert tree.root.feature_idx is None  # It's a leaf node

    predictions = tree.predict(X)
    assert np.all(predictions == 1)

def test_decision_tree_fit_custom_feature_indices(binary_data):
    """Test fitting with custom feature indices."""
    X, y = binary_data

    # Only use first feature
    feature_indices = np.array([0])

    tree = DecisionTree(random_state=42)
    tree.fit(X, y, feature_indices)

    # If any splits were made, they should be on feature 0
    if tree.root.feature_idx is not None:
        assert tree.root.feature_idx == 0

    # Feature importance should be non-zero only for feature 0
    for i in range(len(tree.feature_importances_)):
        if i != 0:
            assert tree.feature_importances_[i] == 0

def test_decision_tree_predict(linearly_separable_data):
    """Test prediction functionality."""
    X, y = linearly_separable_data

    tree = DecisionTree(random_state=42)
    tree.fit(X, y)

    # Predict training data
    train_predictions = tree.predict(X)
    train_accuracy = np.mean(train_predictions == y)

    # This dataset should be easily separable, so accuracy should be high
    assert train_accuracy > 0.9

    # Test single sample prediction
    single_pred = tree.predict(X[0])
    assert single_pred == y[0]

    # Test multiple sample prediction
    multi_pred = tree.predict(X[:3])
    np.testing.assert_array_equal(multi_pred, y[:3])

def test_decision_tree_to_dict(binary_data):
    """Test converting tree to dictionary."""
    X, y = binary_data

    tree = DecisionTree(max_depth=2, random_state=42)
    tree.fit(X, y)

    tree_dict = tree.to_dict()

    # Check dictionary structure
    assert "params" in tree_dict
    assert "tree" in tree_dict
    assert "n_classes_" in tree_dict
    assert "feature_importances_" in tree_dict

    # Check parameter values
    assert tree_dict["params"]["max_depth"] == 2
    assert tree_dict["params"]["criterion"] == "gini"
    assert tree_dict["n_classes_"] == 2
    assert len(tree_dict["feature_importances_"]) == X.shape[1]

def test_decision_tree_from_dict(binary_data):
    """Test reconstructing tree from dictionary."""
    X, y = binary_data

    # Create a tree and convert to dict
    original_tree = DecisionTree(max_depth=2, random_state=42)
    original_tree.fit(X, y)
    tree_dict = original_tree.to_dict()

    # Reconstruct from dict
    reconstructed_tree = DecisionTree.from_dict(tree_dict)

    # Compare parameters
    assert reconstructed_tree.max_depth == original_tree.max_depth
    assert reconstructed_tree.criterion == original_tree.criterion
    assert reconstructed_tree.min_samples_split == original_tree.min_samples_split
    assert reconstructed_tree.min_samples_leaf == original_tree.min_samples_leaf
    assert reconstructed_tree.n_classes_ == original_tree.n_classes_

    # Compare predictions (functional equivalence)
    original_preds = original_tree.predict(X)
    reconstructed_preds = reconstructed_tree.predict(X)
    np.testing.assert_array_equal(original_preds, reconstructed_preds)

def test_decision_tree_max_depth_limit(binary_data):
    """Test max_depth parameter effectively limits tree depth."""
    X, y = binary_data

    # Tree with small max_depth
    shallow_tree = DecisionTree(max_depth=1, random_state=42)
    shallow_tree.fit(X, y)

    # Tree with large max_depth
    deep_tree = DecisionTree(max_depth=10, random_state=42)
    deep_tree.fit(X, y)

    # Count depth of trees
    def count_depth(node):
        if node is None or (node.left is None and node.right is None):
            return 0
        return 1 + max(count_depth(node.left), count_depth(node.right))

    shallow_depth = count_depth(shallow_tree.root)
    deep_depth = count_depth(deep_tree.root)

    # Shallow tree should have depth at most max_depth
    assert shallow_depth <= 1

    # Deep tree might be deeper (or could be the same if data doesn't need deeper splits)
    assert deep_depth >= shallow_depth

def test_decision_tree_min_samples_split(binary_data):
    """Test min_samples_split parameter."""
    X, y = binary_data

    # Setting min_samples_split high should prevent splits with few samples
    tree = DecisionTree(min_samples_split=X.shape[0], random_state=42)
    tree.fit(X, y)
    print(tree.root.left.left is None)


    assert tree.root is not None
    assert tree.root.feature_idx is not None
    assert tree.root.value is None

    assert tree.root.left.feature_idx is None
    assert tree.root.left.value is not None
    assert tree.root.left.value == 0
    assert tree.root.left.feature_idx is None

    assert tree.root.right.feature_idx is None
    assert tree.root.right.value is not None
    assert tree.root.right.value == 1
    assert tree.root.right.feature_idx is None

def test_decision_tree_criterion(binary_data):
    """Test different criterion parameters."""
    X, y = binary_data

    # Test with gini criterion
    gini_tree = DecisionTree(criterion='gini', random_state=42)
    gini_tree.fit(X, y)

    # Test with entropy criterion
    entropy_tree = DecisionTree(criterion='entropy', random_state=42)
    entropy_tree.fit(X, y)

    # Both should produce a valid tree
    assert gini_tree.root is not None
    assert entropy_tree.root is not None

    # The trees might be different due to different splitting criteria
    # but we can't test that directly without controlling the data perfectly

def test_decision_tree_predict_sample_private_method(sample_tree):
    """Test the _predict_sample private method."""
    tree = DecisionTree()
    tree.root = sample_tree

    # Sample below threshold should go to left node
    sample_below = np.array([5, 10, 15])
    prediction_below = tree._predict_sample(tree.root, sample_below)
    assert prediction_below == 0

    # Sample above threshold should go to right node
    sample_above = np.array([10, 20, 30])
    prediction_above = tree._predict_sample(tree.root, sample_above)
    assert prediction_above == 1

def test_decision_tree_create_leaf_private_method():
    """Test the _create_leaf private method."""
    tree = DecisionTree()
    y = np.array([1, 1, 2, 2, 2])

    leaf = tree._create_leaf(y)

    assert leaf.value == 2  # Majority class
    assert leaf.feature_idx is None
    assert leaf.threshold is None
    assert leaf.left is None
    assert leaf.right is None

def test_decision_tree_create_node_private_method():
    """Test the _create_node private method."""
    tree = DecisionTree()
    left_node = Node(value=0)
    right_node = Node(value=1)

    node = tree._create_node(feature_idx=5, threshold=10.0, left=left_node, right=right_node)

    assert node.feature_idx == 5
    assert node.threshold == 10.0
    assert node.left == left_node
    assert node.right == right_node
    assert node.value is None

def test_decision_tree_with_single_feature():
    """Test decision tree with a single feature."""
    X = np.array([[1], [2], [3], [6], [7], [8]])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTree(random_state=42)
    tree.fit(X, y)

    # Tree should successfully split on the only feature
    assert tree.root.feature_idx == 0

    # Predictions should match the pattern in y
    predictions = tree.predict(X)
    np.testing.assert_array_equal(predictions, y)

    # Test point in the middle should go to one side or the other
    middle_point = np.array([[4.5]])
    pred = tree.predict(middle_point)
    assert pred[0] in [0, 1]

def test_decision_tree_with_constant_feature():
    """Test decision tree with a constant feature."""
    # All samples have same value for feature 1
    X = np.array([
        [1, 5, 3],
        [2, 5, 6],
        [3, 5, 9],
        [7, 5, 12],
        [8, 5, 15],
        [9, 5, 18]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTree(random_state=42)
    tree.fit(X, y)

    # Tree should not split on the constant feature (feature_idx should never be 1)
    def check_no_split_on_feature(node, feature_idx):
        if node is None or node.feature_idx is None:
            return True
        if node.feature_idx == feature_idx:
            return False
        return (check_no_split_on_feature(node.left, feature_idx) and
                check_no_split_on_feature(node.right, feature_idx))

    assert check_no_split_on_feature(tree.root, 1)

def test_decision_tree_with_random_state():
    """Test that setting random_state produces deterministic results."""
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)

    # Two trees with same random state
    tree1 = DecisionTree(random_state=42)
    tree1.fit(X, y)

    tree2 = DecisionTree(random_state=42)
    tree2.fit(X, y)

    # Predictions should be identical
    predictions1 = tree1.predict(X)
    predictions2 = tree2.predict(X)
    np.testing.assert_array_equal(predictions1, predictions2)

    # Tree with different random state might give different results
    # Note: This is not guaranteed as the randomness might not affect the result
    tree3 = DecisionTree(random_state=999)
    tree3.fit(X, y)
    predictions3 = tree3.predict(X)

    # We can't assert inequality because the trees might be identical by chance

# ------------------- Edge Cases and Corner Cases -------------------

def test_decision_tree_with_empty_data(empty_data):
    """Test behavior with empty data."""
    X, y = empty_data

    tree = DecisionTree()

    try:
        tree.fit(X, y)
        # If it doesn't error, it should handle empty data gracefully
        assert tree.root is not None
    except Exception as e:
        # If it errors, it should be with a clear message
        assert "empty" in str(e).lower() or "zero" in str(e).lower()

def test_decision_tree_with_single_sample(single_sample_data):
    """Test with a single sample."""
    X, y = single_sample_data

    tree = DecisionTree()
    tree.fit(X, y)

    # Should create a single leaf node with the class
    assert tree.root.value == y[0]
    assert tree.root.feature_idx is None

    # Predict should return the same class
    pred = tree.predict(X)
    assert pred[0] == y[0]

def test_decision_tree_serialization_roundtrip():
    """Test full serialization roundtrip with to_dict and from_dict."""
    X = np.random.rand(30, 3)
    y = np.random.randint(0, 3, 30)

    # Create and fit original tree
    original_tree = DecisionTree(max_depth=3, random_state=42)
    original_tree.fit(X, y)

    # Convert to dict and then to JSON and back
    tree_dict = original_tree.to_dict()
    tree_json = json.dumps(tree_dict)
    reconstructed_dict = json.loads(tree_json)
    reconstructed_tree = DecisionTree.from_dict(reconstructed_dict)

    # Compare predictions
    original_preds = original_tree.predict(X)
    reconstructed_preds = reconstructed_tree.predict(X)
    np.testing.assert_array_equal(original_preds, reconstructed_preds)

def test_decision_tree_with_list_inputs():
    """Test tree with list inputs instead of numpy arrays."""
    X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [0, 0, 1, 1]

    tree = DecisionTree()
    tree.fit(X, y)

    # Should work same as numpy arrays
    assert tree.n_classes_ == 2

    # Predict should also work with lists
    pred = tree.predict([[1, 2]])
    assert pred[0] in [0, 1]

def test_decision_tree_feature_importances():
    """Test feature importance calculation."""
    # Create dataset where feature 0 is perfectly predictive
    X = np.random.rand(20, 3)
    y = (X[:, 0] > 0.5).astype(int)

    tree = DecisionTree(random_state=42)
    tree.fit(X, y)

    # Feature 0 should have the highest importance
    assert np.argmax(tree.feature_importances_) == 0
    assert tree.feature_importances_[0] > 0.5  # Should be dominant

def test_multiple_predictions_same_result(linearly_separable_data):
    """Test that multiple predictions on the same data give consistent results."""
    X, y = linearly_separable_data

    tree = DecisionTree(random_state=42)
    tree.fit(X, y)

    # Make multiple predictions
    pred1 = tree.predict(X)
    pred2 = tree.predict(X)
    pred3 = tree.predict(X)

    # All predictions should be identical
    np.testing.assert_array_equal(pred1, pred2)
    np.testing.assert_array_equal(pred2, pred3)

def test_correct_predictions():
    """Test that predictions are correct and of the expected type."""
    # Create a simple dataset with a clear pattern:
    # If first feature > 5, then class 1, else class 0
    X = np.array([
        [2, 10, 20],
        [3, 15, 25],
        [4, 20, 30],
        [6, 5, 15],
        [7, 10, 20],
        [8, 15, 25]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    # Create and fit a simple decision tree
    tree = DecisionTree(random_state=42)
    tree.fit(X, y)

    # Check predictions on training data
    predictions = tree.predict(X)

    # Check type and shape
    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array"
    assert predictions.shape == (6,), "Predictions should have shape (n_samples,)"
    assert predictions.dtype == np.int64 or predictions.dtype == np.int32 or predictions.dtype == np.float64, \
        f"Predictions should be integer-like, got {predictions.dtype}"

    # Check prediction values match expected
    np.testing.assert_array_equal(predictions, y, "Predictions should match expected values")

    # Test single sample prediction
    single_sample = np.array([3, 25, 30])  # Should be class 0 (first feature < 5)
    single_pred = tree.predict(single_sample)
    assert isinstance(single_pred, (np.int64, np.int32, int, np.float64)), \
        f"Single prediction should be a number, got {type(single_pred)}"
    assert single_pred == 0, "Should predict class 0 for feature value < 5"

    # Test another single sample
    single_sample2 = np.array([7, 5, 10])  # Should be class 1 (first feature > 5)
    single_pred2 = tree.predict(single_sample2)
    assert single_pred2 == 1, "Should predict class 1 for feature value > 5"

    # Test list input
    list_input = [[2, 10, 20], [7, 10, 20]]
    list_predictions = tree.predict(list_input)
    assert isinstance(list_predictions, np.ndarray), "List input should still give NumPy array output"
    np.testing.assert_array_equal(list_predictions, [0, 1])

    # Test predictions with new data
    new_X = np.array([
        [1, 1, 1],   # Should be class 0
        [9, 9, 9]    # Should be class 1
    ])
    new_predictions = tree.predict(new_X)
    np.testing.assert_array_equal(new_predictions, [0, 1])

def test_invalid_criterion():
    """Test behavior with invalid criterion."""
    tree = DecisionTree(criterion='invalid')
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    try:
        tree.fit(X, y)
        # If it doesn't error, it should fall back to a valid criterion
        assert tree.root is not None
    except ValueError:
        # If it errors, it should explain the invalid criterion
        pass

def test_build_tree_with_mock(binary_data):
    """Test _build_tree method with mocked dependencies."""
    X, y = binary_data
    feature_indices = np.arange(X.shape[1])

    tree = DecisionTree(max_depth=1)  # Limit depth to stop recursion early

    # Initialize feature_importances_ to avoid NoneType error
    tree.feature_importances_ = np.zeros(X.shape[1])

    # Define a side effect function that creates correctly sized boolean masks
    def mock_find_best_split_func(X, y, feature_indices, criterion, min_samples_leaf):
        # Return appropriate sized masks based on current data size
        n_samples = X.shape[0]
        left_mask = np.zeros(n_samples, dtype=bool)
        left_mask[:n_samples//2] = True  # First half goes left
        right_mask = ~left_mask  # Second half goes right

        return 0, 5.0, 0.5, left_mask, right_mask

    # Use side_effect to dynamically generate return values
    with patch('utils.tree.find_best_split', side_effect=mock_find_best_split_func):
        # Call _build_tree directly
        root = tree._build_tree(X, y, feature_indices)

        # Check that the root node has the expected values
        assert root.feature_idx == 0
        assert root.threshold == 5.0

        # Check that find_best_split was called
        # Since max_depth=1, the children should be leaf nodes
        assert root.left is not None and root.left.feature_idx is None
        assert root.right is not None and root.right.feature_idx is None

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])