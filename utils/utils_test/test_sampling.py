import pytest
import numpy as np
import sys
import os
from unittest.mock import patch

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sampling import (
    uniform_distribution,
    normalize_weights,
    random_selection,
    weighted_random_selection,
    sample_with_replacement
)

# ------------------- Fixtures -------------------

@pytest.fixture
def sample_weights():
    """Sample weights for testing normalization."""
    return np.array([0.1, 0.3, 0.6, 0.2, 0.8])

@pytest.fixture
def zero_weights():
    """Array with all zero weights."""
    return np.array([0.0, 0.0, 0.0, 0.0])

@pytest.fixture
def negative_weights():
    """Array with some negative weights."""
    return np.array([0.5, -0.2, 0.3, -0.1])

@pytest.fixture
def extreme_weights():
    """Weights with large disparity."""
    return np.array([1000.0, 0.001, 0.002, 0.005])

# ------------------- Tests for uniform_distribution -------------------

def test_uniform_distribution_basic():
    """Test basic uniform distribution creation."""
    n = 5
    dist = uniform_distribution(n)

    # Check shape and values
    assert len(dist) == n
    assert np.allclose(dist, 0.2)
    assert np.isclose(np.sum(dist), 1.0)

def test_uniform_distribution_large():
    """Test with large n values."""
    n = 1000
    dist = uniform_distribution(n)

    assert len(dist) == n
    assert np.isclose(np.sum(dist), 1.0)
    assert np.allclose(dist, 1/n)

def test_uniform_distribution_n1():
    """Test with n=1 edge case."""
    dist = uniform_distribution(1)

    assert len(dist) == 1
    assert dist[0] == 1.0

def test_uniform_distribution_varying_n():
    """Test with different n values."""
    for n in [2, 10, 50, 100]:
        dist = uniform_distribution(n)
        assert len(dist) == n
        assert np.isclose(np.sum(dist), 1.0)
        assert np.allclose(dist, 1/n)

# ------------------- Tests for normalize_weights -------------------

def test_normalize_weights_basic(sample_weights):
    """Test basic weight normalization."""
    normalized = normalize_weights(sample_weights)

    # Check sum is 1
    assert np.isclose(np.sum(normalized), 1.0)

    # Check proportions are maintained
    original_ratio = sample_weights[1] / sample_weights[0]
    normalized_ratio = normalized[1] / normalized[0]
    assert np.isclose(original_ratio, normalized_ratio)

def test_normalize_weights_zeros(zero_weights):
    """Test normalization with all zero weights."""
    normalized = normalize_weights(zero_weights)

    # Should return uniform distribution
    expected = np.ones(len(zero_weights)) / len(zero_weights)
    np.testing.assert_allclose(normalized, expected)
    assert np.isclose(np.sum(normalized), 1.0)

def test_normalize_weights_min_weight(sample_weights):
    """Test normalization with min_weight parameter."""

    min_weight = 0.15

    # Print input weights
    print(f"Original weights: {sample_weights}")

    # First normalization step
    initial_normalized = sample_weights / np.sum(sample_weights)
    print(f"Initially normalized: {initial_normalized}")

    # Apply full normalization with min_weight
    normalized = normalize_weights(sample_weights, min_weight)

    print(f"After min_weight normalization: {normalized}")
    print(f"Weights below min_weight: {normalized < min_weight}")

    # Check sum is 1
    assert np.isclose(np.sum(normalized), 1.0)

       # Check all weights are at least min_weight (accounting for floating-point precision)
    assert np.all(normalized >= (min_weight - 1e-4))

def test_normalize_weights_high_min_weight(sample_weights):
    """Test with high min_weight that forces equalization."""
    min_weight = 0.4  # This is high enough that not all values can maintain it
    normalized = normalize_weights(sample_weights, min_weight)

    # With such high min_weight, should revert to uniform
    expected = np.ones(len(sample_weights)) / len(sample_weights)
    np.testing.assert_allclose(normalized, expected)

def test_normalize_weights_negative(negative_weights):
    """Test with negative weights."""
    normalized = normalize_weights(negative_weights)

    # Should handle negative weights appropriately or revert to uniform
    assert np.isclose(np.sum(normalized), 1.0)

def test_normalize_weights_extreme(extreme_weights):
    """Test with extreme weight disparities."""
    normalized = normalize_weights(extreme_weights)

    # Check sum is 1
    assert np.isclose(np.sum(normalized), 1.0)

    # Original weights: [1000.0, 0.001, 0.002, 0.005]
    # The largest value should still dominate
    assert normalized[0] > 0.9

def test_normalize_weights_single_value():
    """Test normalization with single value."""
    single_weight = np.array([42.0])
    normalized = normalize_weights(single_weight)

    assert len(normalized) == 1
    assert normalized[0] == 1.0

# ------------------- Tests for random_selection -------------------

def test_random_selection_basic():
    """Test basic random selection behavior."""
    n_features = 100
    max_features = 10

    # Test multiple times for randomness
    for _ in range(5):
        selected = random_selection(n_features, max_features)

        # Check unique values
        assert len(selected) == max_features
        assert len(np.unique(selected)) == max_features

        # Check range
        assert np.all(selected >= 0)
        assert np.all(selected < n_features)

def test_random_selection_all_features():
    """Test when max_features equals n_features."""
    n_features = 10
    max_features = 10

    selected = random_selection(n_features, max_features)

    assert len(selected) == n_features
    assert set(selected) == set(range(n_features))

def test_random_selection_more_than_available():
    """Test when max_features is larger than n_features."""
    n_features = 5
    max_features = 10

    selected = random_selection(n_features, max_features)

    # Should limit to n_features
    assert len(selected) == n_features
    assert set(selected) == set(range(n_features))

def test_random_selection_distribution():
    """Test that the selection is roughly uniform over many trials."""
    n_features = 5
    max_features = 3
    n_trials = 1000

    counts = np.zeros(n_features)

    # Run many trials and count selections
    for _ in range(n_trials):
        selected = random_selection(n_features, max_features)
        for idx in selected:
            counts[idx] += 1

    # Check that all features have been selected at least once
    assert np.all(counts > 0)

    # Expected count per feature: (max_features / n_features) * n_trials
    expected = (max_features / n_features) * n_trials

    # Allow for statistical variation (within 20% of expected)
    assert np.all(counts > 0.8 * expected)
    assert np.all(counts < 1.2 * expected)

def test_random_selection_one_feature():
    """Test selection of just one feature."""
    n_features = 100
    max_features = 1

    selected = random_selection(n_features, max_features)

    assert len(selected) == 1
    assert 0 <= selected[0] < n_features

# ------------------- Tests for weighted_random_selection -------------------

def test_weighted_random_selection_basic(sample_weights):
    """Test basic weighted random selection."""
    max_features = 3

    selected = weighted_random_selection(max_features, sample_weights)

    # Check size and uniqueness
    assert len(selected) == max_features
    assert len(np.unique(selected)) == max_features

    # Check range
    assert np.all(selected >= 0)
    assert np.all(selected < len(sample_weights))

def test_weighted_random_selection_zero_weights(zero_weights):
    """Test weighted selection with all zero weights."""
    max_features = 2

    # Should behave like random selection
    with patch('utils.sampling.random_selection') as mock_random:
        mock_random.return_value = np.array([0, 1])
        selected = weighted_random_selection(max_features, zero_weights)

        mock_random.assert_called_once_with(len(zero_weights), max_features)
        np.testing.assert_array_equal(selected, np.array([0, 1]))

def test_weighted_random_selection_distribution(sample_weights):
    """Test that selection follows weight distribution over many trials."""
    max_features = 1
    n_trials = 10000

    # Normalized weights to get expected probabilities
    expected_probs = sample_weights / np.sum(sample_weights)
    counts = np.zeros(len(sample_weights))

    # Fix random seed for reproducibility
    np.random.seed(42)

    # Run many trials and count selections
    for _ in range(n_trials):
        selected = weighted_random_selection(max_features, sample_weights)
        counts[selected[0]] += 1

    # Convert counts to observed probabilities
    observed_probs = counts / n_trials

    # Check that relative frequencies approximate the weights
    # Allow for statistical variation (within 10% of expected)
    np.testing.assert_allclose(observed_probs, expected_probs, rtol=0.1)

def test_weighted_random_selection_all_features(sample_weights):
    """Test when max_features equals n_features."""
    max_features = len(sample_weights)

    selected = weighted_random_selection(max_features, sample_weights)

    assert len(selected) == len(sample_weights)
    assert set(selected) == set(range(len(sample_weights)))

def test_weighted_random_selection_more_than_available(sample_weights):
    """Test when max_features is larger than n_features."""
    max_features = len(sample_weights) + 5

    selected = weighted_random_selection(max_features, sample_weights)

    # Should limit to n_features
    assert len(selected) == len(sample_weights)
    assert set(selected) == set(range(len(sample_weights)))

# ------------------- Tests for sample_with_replacement -------------------

def test_sample_with_replacement_basic():
    """Test basic sampling with replacement."""
    indices = np.arange(10)
    size = 15  # More than the number of unique indices

    sampled = sample_with_replacement(indices, size)

    # Check size
    assert len(sampled) == size

    # Check values are within range
    assert np.all(np.isin(sampled, indices))

    # With replacement, expect some duplicates when size > len(indices)
    _, counts = np.unique(sampled, return_counts=True)
    assert np.any(counts > 1)

def test_sample_with_replacement_with_probs():
    """Test sampling with replacement with probabilities."""
    indices = np.array([10, 20, 30, 40, 50])
    size = 1000
    p = np.array([0.5, 0.3, 0.1, 0.05, 0.05])

    np.random.seed(42)
    sampled = sample_with_replacement(indices, size, p)

    # Check size
    assert len(sampled) == size

    # Check values are within range
    assert np.all(np.isin(sampled, indices))

    # Check distribution
    unique, counts = np.unique(sampled, return_counts=True)
    counts_dict = dict(zip(unique, counts))

    # Check the most frequent values correspond to highest probabilities
    assert counts_dict[10] > counts_dict[20] > counts_dict[30]

def test_sample_with_replacement_uniform():
    """Test sampling with uniform probabilities."""
    indices = np.array([1, 2, 3, 4, 5])
    size = 10000

    np.random.seed(42)
    sampled = sample_with_replacement(indices, size)

    # Check distribution is roughly uniform
    unique, counts = np.unique(sampled, return_counts=True)
    expected = size / len(indices)

    # Allow for statistical variation (within 10% of expected)
    assert np.all(np.abs(counts - expected) < 0.1 * expected)

def test_sample_with_replacement_extreme_probs():
    """Test with extreme probabilities."""
    indices = np.array([1, 2, 3, 4, 5])
    size = 100
    p = np.array([0.96, 0.01, 0.01, 0.01, 0.01])

    np.random.seed(42)
    sampled = sample_with_replacement(indices, size, p)

    # The first value should dominate
    unique, counts = np.unique(sampled, return_counts=True)
    counts_dict = dict(zip(unique, counts))

    assert counts_dict.get(1, 0) > 90  # Should be around 96

def test_sample_with_replacement_single_value():
    """Test with a single value."""
    indices = np.array([42])
    size = 10

    sampled = sample_with_replacement(indices, size)

    # All should be the same value
    assert len(sampled) == size
    assert np.all(sampled == 42)

def test_sample_with_replacement_with_unnormalized_probs():
    """Test with unnormalized probabilities."""
    indices = np.array([1, 2, 3])
    size = 100
    p = np.array([10, 20, 30])  # Not normalized

    sampled = sample_with_replacement(indices, size, p)

    # Function should normalize internally
    assert len(sampled) == size

    # The third value should be sampled most frequently
    unique, counts = np.unique(sampled, return_counts=True)
    counts_dict = dict(zip(unique, counts))

    if 3 in counts_dict and 2 in counts_dict:
        assert counts_dict[3] > counts_dict[2]
    if 2 in counts_dict and 1 in counts_dict:
        assert counts_dict[2] > counts_dict[1]

# ------------------- Additional Edge Cases -------------------

def test_normalize_weights_empty():
    """Test normalizing empty weights array."""
    empty_weights = np.array([])

    try:
        normalized = normalize_weights(empty_weights)
        # If it doesn't raise an error, check result is also empty
        assert len(normalized) == 0
    except Exception as e:
        # Could reasonably raise an error for empty array
        assert "empty" in str(e).lower() or "zero" in str(e).lower()

def test_weighted_random_selection_single_weight():
    """Test weighted selection with a single weight."""
    weights = np.array([5.0])
    max_features = 1

    selected = weighted_random_selection(max_features, weights)

    assert len(selected) == 1
    assert selected[0] == 0

def test_sample_with_replacement_zero_size():
    """Test sampling with size=0."""
    indices = np.array([1, 2, 3, 4, 5])
    size = 0

    sampled = sample_with_replacement(indices, size)

    assert len(sampled) == 0

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])