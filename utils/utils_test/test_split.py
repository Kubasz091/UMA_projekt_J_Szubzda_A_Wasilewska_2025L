import pytest
import numpy as np
import sys
import os
from unittest.mock import patch

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.split import (
    calculate_entropy,
    calculate_gini_impurity,
    split_node,
    calculate_information_gain,
    find_best_split
)

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
def empty_data():
    """Empty dataset."""
    X = np.array([]).reshape(0, 3)
    y = np.array([])
    return X, y

# ------------------- Tests for calculate_entropy -------------------

def test_calculate_entropy_binary_balanced():
    """Test entropy calculation for balanced binary classes."""
    y = np.array([0, 0, 1, 1])
    entropy = calculate_entropy(y)
    # Entropy for balanced binary distribution should be 1.0
    assert np.isclose(entropy, 1.0, atol=1e-10)

def test_calculate_entropy_binary_unbalanced():
    """Test entropy calculation for unbalanced binary classes."""
    y = np.array([0, 0, 0, 1])
    entropy = calculate_entropy(y)
    # Entropy for this distribution: -0.75*log2(0.75) - 0.25*log2(0.25) ≈ 0.811
    assert np.isclose(entropy, 0.811, atol=1e-3)

def test_calculate_entropy_multiclass_balanced():
    """Test entropy calculation for balanced multi-class."""
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    entropy = calculate_entropy(y)
    # Entropy for balanced 3-class: -3*(1/3*log2(1/3)) = log2(3) ≈ 1.585
    assert np.isclose(entropy, 1.585, atol=1e-3)

def test_calculate_entropy_pure():
    """Test entropy calculation for pure distribution (all same class)."""
    y = np.array([1, 1, 1, 1])
    entropy = calculate_entropy(y)
    # Entropy for pure distribution should be 0.0
    assert np.isclose(entropy, 0.0, atol=1e-10)

def test_calculate_entropy_empty():
    """Test entropy calculation on empty array (should handle gracefully)."""
    y = np.array([])
    entropy = calculate_entropy(y)
    # Expected behavior: entropy of empty set is 0
    assert np.isclose(entropy, 0.0, atol=1e-10)

# ------------------- Tests for calculate_gini_impurity -------------------

def test_calculate_gini_impurity_binary_balanced():
    """Test Gini impurity calculation for balanced binary classes."""
    y = np.array([0, 0, 1, 1])
    gini = calculate_gini_impurity(y)
    # Gini for balanced binary is 1 - ((1/2)² + (1/2)²) = 0.5
    assert np.isclose(gini, 0.5, atol=1e-10)

def test_calculate_gini_impurity_binary_unbalanced():
    """Test Gini impurity calculation for unbalanced binary classes."""
    y = np.array([0, 0, 0, 1])
    gini = calculate_gini_impurity(y)
    # Gini for this distribution: 1 - ((3/4)² + (1/4)²) = 0.375
    assert np.isclose(gini, 0.375, atol=1e-10)

def test_calculate_gini_impurity_multiclass_balanced():
    """Test Gini impurity calculation for balanced multi-class."""
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    gini = calculate_gini_impurity(y)
    # Gini for balanced 3-class: 1 - 3*(1/3)² = 1 - 1/3 = 2/3 ≈ 0.667
    assert np.isclose(gini, 0.667, atol=1e-3)

def test_calculate_gini_impurity_pure():
    """Test Gini impurity calculation for pure distribution."""
    y = np.array([1, 1, 1, 1])
    gini = calculate_gini_impurity(y)
    # Gini for pure distribution should be 0.0
    assert np.isclose(gini, 0.0, atol=1e-10)

def test_calculate_gini_impurity_empty():
    """Test Gini calculation on empty array."""
    y = np.array([])
    gini = calculate_gini_impurity(y)
    # Expected behavior for empty set
    assert np.isclose(gini, 0.0, atol=1e-10)

# ------------------- Tests for split_node -------------------

def test_split_node_basic(binary_data):
    """Test basic node splitting functionality."""
    X, y = binary_data
    feature_idx = 0
    threshold = 10  # This should split the data exactly in half

    X_left, y_left, X_right, y_right = split_node(X, y, feature_idx, threshold)

    # Check shapes
    assert X_left.shape[0] == 4
    assert y_left.shape[0] == 4
    assert X_right.shape[0] == 4
    assert y_right.shape[0] == 4

    # Check values
    assert np.all(X_left[:, feature_idx] <= threshold)
    assert np.all(X_right[:, feature_idx] > threshold)
    assert np.array_equal(y_left, np.array([0, 0, 0, 0]))  # All class 0
    assert np.array_equal(y_right, np.array([1, 1, 1, 1]))  # All class 1

def test_split_node_unbalanced(binary_data):
    """Test splitting with unbalanced partition."""
    X, y = binary_data
    feature_idx = 0
    threshold = 3  # This should put just one sample on the left

    X_left, y_left, X_right, y_right = split_node(X, y, feature_idx, threshold)

    # Check shapes
    assert X_left.shape[0] == 1
    assert y_left.shape[0] == 1
    assert X_right.shape[0] == 7
    assert y_right.shape[0] == 7

    # Check values
    assert np.all(X_left[:, feature_idx] <= threshold)
    assert np.all(X_right[:, feature_idx] > threshold)

def test_split_node_empty_result():
    """Test splitting when one side gets no samples."""
    X = np.array([[1], [2], [3]])
    y = np.array([0, 0, 0])
    feature_idx = 0
    threshold = 0  # This should put all samples on the right

    X_left, y_left, X_right, y_right = split_node(X, y, feature_idx, threshold)

    # Check shapes
    assert X_left.shape[0] == 0
    assert y_left.shape[0] == 0
    assert X_right.shape[0] == 3
    assert y_right.shape[0] == 3

def test_split_node_multiclass(multiclass_data):
    """Test splitting with multi-class data."""
    X, y = multiclass_data
    feature_idx = 0
    threshold = 15  # This should separate classes reasonably well

    X_left, y_left, X_right, y_right = split_node(X, y, feature_idx, threshold)

    # Check distribution of classes
    assert set(np.unique(y_left)) != set(np.unique(y))  # Not all classes on the left
    assert len(y_left) + len(y_right) == len(y)  # All samples accounted for

# ------------------- Tests for calculate_information_gain -------------------

def test_calculate_information_gain_binary(binary_data):
    """Test information gain calculation with binary data."""
    X, y = binary_data
    info_gain = calculate_information_gain(X, y)

    # Check shape of output
    assert info_gain.shape[0] == X.shape[1]

    # Values should be non-negative
    assert np.all(info_gain >= 0)

def test_calculate_information_gain_multiclass(multiclass_data):
    """Test information gain with multi-class data."""
    X, y = multiclass_data
    info_gain = calculate_information_gain(X, y)

    # Check shape of output
    assert info_gain.shape[0] == X.shape[1]

    # Values should be non-negative
    assert np.all(info_gain >= 0)

def test_calculate_information_gain_pure(pure_data):
    """Test information gain with pure class data."""
    X, y = pure_data
    info_gain = calculate_information_gain(X, y)

    # With pure data, info gain should be zero or very close to zero
    assert np.allclose(info_gain, 0, atol=1e-10)

# ------------------- Tests for find_best_split -------------------

def test_find_best_split_binary(binary_data):
    """Test finding the best split with binary data."""
    X, y = binary_data
    feature_indices = np.arange(X.shape[1])

    # Test with gini criterion (default)
    feat_idx, threshold, score, left_idx, right_idx = find_best_split(X, y, feature_indices)

    # Should find a split
    assert feat_idx is not None
    assert threshold is not None
    assert score > 0

    # Check indices
    assert np.sum(left_idx) + np.sum(right_idx) == X.shape[0]

    # Test with entropy criterion
    feat_idx, threshold, score, left_idx, right_idx = find_best_split(X, y, feature_indices, criterion='entropy')

    # Should find a split
    assert feat_idx is not None
    assert threshold is not None
    assert score > 0

def test_find_best_split_multiclass(multiclass_data):
    """Test finding best split with multi-class data."""
    X, y = multiclass_data
    feature_indices = np.arange(X.shape[1])

    feat_idx, threshold, score, left_idx, right_idx = find_best_split(X, y, feature_indices)

    # Should find a split
    assert feat_idx is not None
    assert threshold is not None
    assert score > 0

def test_find_best_split_pure(pure_data):
    """Test finding best split with pure class data."""
    X, y = pure_data
    feature_indices = np.arange(X.shape[1])

    feat_idx, threshold, score, left_idx, right_idx = find_best_split(X, y, feature_indices)

    # With pure data, should not find a meaningful split (score should be 0)
    assert score == 0.0

def test_find_best_split_min_samples_leaf(binary_data):
    """Test min_samples_leaf constraint in best split."""
    X, y = binary_data
    feature_indices = np.arange(X.shape[1])

    # Set min_samples_leaf to a high value
    feat_idx, threshold, score, left_idx, right_idx = find_best_split(X, y, feature_indices, min_samples_leaf=4)

    # Should still find a split
    assert feat_idx is not None

    # Check that split respects min_samples_leaf
    if left_idx is not None:
        assert np.sum(left_idx) >= 4
        assert np.sum(right_idx) >= 4

def test_find_best_split_max_unique_for_exact(binary_data):
    """Test max_unique_for_exact parameter."""
    X, y = binary_data
    feature_indices = np.arange(X.shape[1])

    # Try different values of max_unique_for_exact
    feat_idx1, _, _, _, _ = find_best_split(X, y, feature_indices, max_unique_for_exact=2)
    feat_idx2, _, _, _, _ = find_best_split(X, y, feature_indices, max_unique_for_exact=100)

    # Both should find splits, but they might be different due to different threshold selection strategies
    assert feat_idx1 is not None
    assert feat_idx2 is not None

def test_find_best_split_empty():
    """Test finding best split with empty data."""
    X = np.array([]).reshape(0, 3)
    y = np.array([])
    feature_indices = np.array([0, 1, 2])

    result = find_best_split(X, y, feature_indices)

    # Should return None values for empty data
    assert result[0] is None
    assert result[1] is None
    assert result[2] == 0.0
    assert result[3] is None
    assert result[4] is None

def test_find_best_split_single_sample():
    """Test finding best split with a single sample."""
    X = np.array([[1, 2, 3]])
    y = np.array([0])
    feature_indices = np.array([0, 1, 2])

    result = find_best_split(X, y, feature_indices)

    # Cannot split a single sample meaningfully
    assert result[0] is None
    assert result[1] is None
    assert result[2] == 0.0

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])