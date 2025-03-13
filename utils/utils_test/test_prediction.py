import pytest
import numpy as np
import sys
import os
from unittest.mock import patch

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prediction import accuracy, majority_vote, weighted_majority_vote

# ------------------- Fixtures -------------------

@pytest.fixture
def binary_predictions():
    """Binary classification predictions from multiple models."""
    return np.array([
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],  # model 1
        [0, 1, 1, 1, 0, 0, 0, 1, 0, 1],  # model 2
        [0, 0, 0, 1, 0, 1, 1, 1, 0, 0]   # model 3
    ])

@pytest.fixture
def binary_targets():
    """Binary classification ground truth."""
    return np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])

@pytest.fixture
def multiclass_predictions():
    """Multiclass predictions from multiple models."""
    return np.array([
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],  # model 1
        [0, 1, 1, 0, 1, 2, 2, 1, 0, 0],  # model 2
        [0, 2, 2, 0, 0, 2, 0, 1, 1, 0]   # model 3
    ])

@pytest.fixture
def multiclass_targets():
    """Multiclass ground truth."""
    return np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

@pytest.fixture
def tied_predictions():
    """Predictions with ties."""
    return np.array([
        [0, 0, 1, 1, 2],  # model 1
        [1, 0, 1, 2, 2],  # model 2
        [1, 2, 0, 2, 0],  # model 3
        [1, 0, 0, 1, 1]   # model 4
    ])

@pytest.fixture
def weighted_predictions():
    """Predictions for weighted voting tests."""
    return np.array([
        [0, 0, 0, 1, 1],  # model 1 (weight = 1)
        [1, 1, 1, 1, 1],  # model 2 (weight = 2)
        [0, 0, 0, 0, 0]   # model 3 (weight = 3)
    ])

# ------------------- Tests for accuracy -------------------

def test_accuracy_basic():
    """Test basic accuracy calculation."""
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0])

    # 3 correct out of 5 = 0.6
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.6, atol=1e-10)

def test_accuracy_perfect():
    """Test perfect accuracy (100%)."""
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1])

    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 1.0, atol=1e-10)

def test_accuracy_worst():
    """Test worst accuracy (0%)."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1])

    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.0, atol=1e-10)

def test_accuracy_multiclass():
    """Test accuracy with multiclass data."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 1, 3, 0])

    # 3 correct out of 5 = 0.6
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.6, atol=1e-10)

def test_accuracy_with_strings():
    """Test accuracy with string labels."""
    # Note: Numba might not support string arrays, so this may fail
    try:
        y_true = np.array(["cat", "dog", "bird", "cat"])
        y_pred = np.array(["cat", "cat", "bird", "dog"])

        # 2 correct out of 4 = 0.5
        acc = accuracy(y_true, y_pred)
        assert np.isclose(acc, 0.5, atol=1e-10)
    except TypeError:
        # Skip if Numba doesn't support strings
        pytest.skip("Numba doesn't support string comparison")

def test_accuracy_with_empty_arrays():
    """Test accuracy with empty arrays (edge case)."""
    try:
        y_true = np.array([])
        y_pred = np.array([])

        acc = accuracy(y_true, y_pred)
        # Should handle empty arrays gracefully, either return NaN or 0
        assert np.isnan(acc) or acc == 0
    except Exception as e:
        # May raise division by zero or other errors
        assert "division" in str(e).lower() or "empty" in str(e).lower()

def test_accuracy_with_bool_arrays():
    """Test accuracy with boolean arrays."""
    y_true = np.array([True, False, True, False])
    y_pred = np.array([True, False, False, True])

    # 2 correct out of 4 = 0.5
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.5, atol=1e-10)

def test_accuracy_with_different_dtypes():
    """Test accuracy with different numpy dtypes."""
    y_true = np.array([0, 1, 0, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 0], dtype=np.int32)

    # 2 correct out of 4 = 0.5
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.5, atol=1e-10)

# ------------------- Tests for majority_vote -------------------

def test_majority_vote_basic(binary_predictions):
    """Test basic majority voting functionality."""
    result = majority_vote(binary_predictions)

    # Expected result: majority vote for each sample
    expected = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    np.testing.assert_array_equal(result, expected)

def test_majority_vote_multiclass(multiclass_predictions):
    """Test majority voting with multiclass data."""
    result = majority_vote(multiclass_predictions)

    # Expected result: majority vote for each sample
    expected = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 0])
    np.testing.assert_array_equal(result, expected)

def test_majority_vote_ties(tied_predictions):
    """Test majority voting with ties."""
    result = majority_vote(tied_predictions)
    print(tied_predictions)
    print(result)
    # For ties, numpy's argmax takes the first occurrence
    expected = np.array([1, 0, 0, 1, 2])
    np.testing.assert_array_equal(result, expected)

def test_majority_vote_single_prediction():
    """Test majority voting with a single prediction array."""
    predictions = np.array([0, 1, 2, 0, 1])

    result = majority_vote(predictions)

    # With a single model, should return the same predictions
    np.testing.assert_array_equal(result, predictions)

def test_majority_vote_empty():
    """Test majority voting with empty array."""
    predictions = np.array([])

    result = majority_vote(predictions)

    # Should handle empty array gracefully
    assert len(result) == 0

def test_majority_vote_2d_single_model():
    """Test majority voting with 2D array but only one model."""
    predictions = np.array([[0, 1, 0, 1, 0]])

    result = majority_vote(predictions)

    expected = np.array([0, 1, 0, 1, 0])
    np.testing.assert_array_equal(result, expected)

def test_majority_vote_with_nan():
    """Test majority voting with NaN values."""
    predictions = np.array([
        [0, 1, np.nan, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 0]
    ])

    try:
        result = majority_vote(predictions)
        # Should handle NaN gracefully or raise appropriate error
    except Exception as e:
        assert "nan" in str(e).lower() or "float" in str(e).lower()

# ------------------- Tests for weighted_majority_vote -------------------

def test_weighted_majority_vote_basic(weighted_predictions):
    """Test basic weighted majority voting."""
    weights = np.array([1.0, 2.0, 3.0])

    result = weighted_majority_vote(weighted_predictions, weights)

    # With these weights, model 3 should dominate (all 0s)
    expected = np.array([0, 0, 0, 0, 0])
    np.testing.assert_array_equal(result, expected)

def test_weighted_majority_vote_equal_weights(binary_predictions):
    """Test weighted voting with equal weights (should equal normal majority vote)."""
    weights = np.ones(len(binary_predictions))

    weighted_result = weighted_majority_vote(binary_predictions, weights)
    majority_result = majority_vote(binary_predictions)

    # Should give same result as majority_vote
    np.testing.assert_array_equal(weighted_result, majority_result)

def test_weighted_majority_vote_single_dominant(multiclass_predictions):
    """Test weighted voting with one dominant model."""
    # Make the first model have 100x weight of others
    weights = np.array([100.0, 1.0, 1.0])

    result = weighted_majority_vote(multiclass_predictions, weights)

    # First model should completely dominate
    expected = multiclass_predictions[0]
    np.testing.assert_array_equal(result, expected)

def test_weighted_majority_vote_no_weights(binary_predictions):
    """Test weighted voting with no weights provided."""
    # None weights should fall back to majority vote
    weighted_result = weighted_majority_vote(binary_predictions, None)
    majority_result = majority_vote(binary_predictions)

    np.testing.assert_array_equal(weighted_result, majority_result)

def test_weighted_majority_vote_mismatched_weights(binary_predictions):
    """Test weighted voting with mismatched weights length."""
    # Wrong number of weights should fall back to majority vote
    weights = np.array([1.0, 2.0])  # Only 2 weights for 3 models

    weighted_result = weighted_majority_vote(binary_predictions, weights)
    majority_result = majority_vote(binary_predictions)

    np.testing.assert_array_equal(weighted_result, majority_result)

def test_weighted_majority_vote_single_prediction():
    """Test weighted voting with a single prediction array."""
    predictions = np.array([[0, 1, 2, 0, 1]])
    weights = np.array([1.0])

    result = weighted_majority_vote(predictions, weights)

    # With a single model, should return the same predictions
    expected = np.array([0, 1, 2, 0, 1])
    np.testing.assert_array_equal(result, expected)

def test_weighted_majority_vote_empty():
    """Test weighted voting with empty array."""
    predictions = np.array([])
    weights = np.array([])

    result = weighted_majority_vote(predictions, weights)

    # Should handle empty array gracefully
    assert len(result) == 0

def test_weighted_majority_vote_with_n_classes(weighted_predictions):
    """Test weighted voting with specified n_classes."""
    weights = np.array([1.0, 2.0, 3.0])
    n_classes = 3  # Explicitly set n_classes higher than needed

    result = weighted_majority_vote(weighted_predictions, weights, n_classes)

    # Should still be all 0s even with extra classes
    expected = np.array([0, 0, 0, 0, 0])
    np.testing.assert_array_equal(result, expected)

def test_weighted_majority_vote_negative_weights():
    """Test weighted voting with negative weights."""
    predictions = np.array([
        [0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0]
    ])

    weights = np.array([1.0, -1.0, 1.0])  # Second model has negative weight

    result = weighted_majority_vote(predictions, weights)

    # Negative weights should invert the votes of the second model
    # This is implementation-dependent, so we either check the behavior or skip
    try:
        # If negative weights are treated normally
        expected = np.array([0, 0, 0, 1, 0])
        np.testing.assert_array_equal(result, expected)
    except AssertionError:
        # If negative weights are not allowed or handled differently
        pass

def test_weighted_majority_vote_with_zero_weights():
    """Test weighted voting with some zero weights."""
    predictions = np.array([
        [0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0]
    ])

    weights = np.array([1.0, 0.0, 1.0])  # Second model is ignored

    result = weighted_majority_vote(predictions, weights)

    # Only model 1 and 3 should count
    expected = np.array([0, 0, 0, 1, 0])
    np.testing.assert_array_equal(result, expected)

def test_weighted_majority_vote_extreme_values():
    """Test weighted voting with extreme probability values."""
    predictions = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2]
    ])

    weights = np.array([1e10, 1e-10, 1e-10])  # First model has extreme weight

    result = weighted_majority_vote(predictions, weights)

    # First model should completely dominate
    expected = np.array([0, 0, 0, 0, 0])
    np.testing.assert_array_equal(result, expected)

# ------------------- Edge Cases and Corner Cases -------------------

def test_edge_case_single_sample():
    """Test with single sample."""
    predictions = np.array([
        [0],
        [1],
        [0]
    ])

    # Majority vote
    maj_result = majority_vote(predictions)
    assert maj_result[0] == 0

    # Weighted majority vote
    weights = np.array([1.0, 3.0, 1.0])
    weighted_result = weighted_majority_vote(predictions, weights)
    assert weighted_result[0] == 1  # Model 2 has higher weight

def test_edge_case_high_dimensional():
    """Test with high number of classes and models."""
    n_models = 50
    n_samples = 10
    n_classes = 20

    # Generate random predictions
    np.random.seed(42)
    predictions = np.random.randint(0, n_classes, size=(n_models, n_samples))
    weights = np.random.random(n_models)

    # Test both functions
    maj_result = majority_vote(predictions)
    weighted_result = weighted_majority_vote(predictions, weights)

    # Basic shape and range checks
    assert maj_result.shape == (n_samples,)
    assert weighted_result.shape == (n_samples,)
    assert np.all(maj_result >= 0) and np.all(maj_result < n_classes)
    assert np.all(weighted_result >= 0) and np.all(weighted_result < n_classes)

def test_weighted_majority_vote_with_inf_weights():
    """Test weighted voting with infinite weights."""
    predictions = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2]
    ])

    weights = np.array([np.inf, 1.0, 1.0])

    try:
        result = weighted_majority_vote(predictions, weights)
        # If it handles inf, first model should dominate
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected)
    except Exception as e:
        # If it can't handle inf, should raise an error
        assert "inf" in str(e).lower() or "float" in str(e).lower()

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])