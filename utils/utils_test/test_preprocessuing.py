import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import impute_missing_values, delete_rows_with_missing_values, encode_categorical

# Fixtures for test data
@pytest.fixture
def X_with_nans():
    return np.array([
        [1.0, 2.0, np.nan, 4.0],
        [5.0, np.nan, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [np.nan, 14.0, 15.0, 16.0]
    ])

@pytest.fixture
def X_with_all_nans_column():
    return np.array([
        [1.0, np.nan, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, np.nan, 9.0]
    ])

@pytest.fixture
def X_no_nans():
    return np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])

@pytest.fixture
def X_categorical():
    return np.array([
        ['A', 1.0, 'small'],
        ['B', 2.0, 'medium'],
        ['A', 3.0, 'large'],
        ['C', 4.0, 'small']
    ], dtype=object)

# Tests for impute_missing_values function
@pytest.mark.parametrize("strategy,expected_values", [
    ('mean', [(0, 2, 11.0), (1, 1, 8.67), (3, 0, 5.0)]),
    ('median', [(0, 2, 11.0), (1, 1, 10.0), (3, 0, 5.0)]),
    ('most_frequent', None),
    ('constant', [(0, 2, 0), (1, 1, 0), (3, 0, 0)])
])
def test_impute_missing_values(X_with_nans, strategy, expected_values):
    """Test different imputation strategies."""
    X_imputed = impute_missing_values(X_with_nans, strategy=strategy)

    # Check shape preserved
    assert X_imputed.shape == X_with_nans.shape

    # Check non-NaN values unchanged
    assert X_imputed[0, 0] == 1.0
    assert X_imputed[2, 3] == 12.0

    # Check imputed values
    if expected_values:
        for row, col, expected in expected_values:
            if strategy == 'mean' and abs(expected - 8.67) < 0.1:
                # Handle floating point comparison for mean
                assert abs(X_imputed[row, col] - expected) < 0.1
            else:
                assert X_imputed[row, col] == expected

def test_impute_missing_values_most_frequent():
    """Test most_frequent imputation strategy separately due to special data needs."""
    X = np.array([
        [1.0, 2.0, 3.0],
        [1.0, np.nan, 4.0],
        [2.0, 2.0, np.nan],
        [np.nan, 5.0, 3.0]
    ])

    X_imputed = impute_missing_values(X, strategy='most_frequent')

    assert X_imputed[3, 0] == 1.0
    assert X_imputed[1, 1] == 2.0
    assert X_imputed[2, 2] == 3.0

def test_impute_missing_values_all_nans_column(X_with_all_nans_column):
    """Test imputation when an entire column is NaN."""
    X_imputed = impute_missing_values(X_with_all_nans_column, strategy='mean')

    # Check that all-NaN column was removed
    assert X_imputed.shape == (3, 2)

    # Check that first column is preserved
    np.testing.assert_array_equal(X_imputed[:, 0], np.array([1.0, 4.0, 7.0]))

    # Check that third column is now the second column
    np.testing.assert_array_equal(X_imputed[:, 1], np.array([3.0, 6.0, 9.0]))

def test_impute_missing_values_no_nans(X_no_nans):
    """Test imputation when no values are missing."""
    X_imputed = impute_missing_values(X_no_nans, strategy='mean')

    # Check that data is unchanged
    np.testing.assert_array_equal(X_imputed, X_no_nans)

# Tests for delete_rows_with_missing_values function
def test_delete_rows_with_missing_values(X_with_nans):
    """Test deletion of rows with missing values."""
    X_cleaned = delete_rows_with_missing_values(X_with_nans)

    # Only the third row (index 2) has no NaNs
    assert X_cleaned.shape == (1, 4)
    np.testing.assert_array_equal(X_cleaned[0], np.array([9.0, 10.0, 11.0, 12.0]))

def test_delete_rows_no_missing_values(X_no_nans):
    """Test deletion when no values are missing."""
    X_cleaned = delete_rows_with_missing_values(X_no_nans)

    # Check that data is unchanged
    np.testing.assert_array_equal(X_cleaned, X_no_nans)

def test_delete_rows_all_missing():
    """Test deletion when all rows have missing values."""
    X = np.array([
        [1.0, np.nan, 3.0],
        [np.nan, 5.0, 6.0],
        [7.0, 8.0, np.nan]
    ])

    X_cleaned = delete_rows_with_missing_values(X)

    # All rows have at least one NaN, should return empty array
    assert X_cleaned.shape[0] == 0

# Tests for encode_categorical function
def test_encode_categorical_onehot_single_column():
    """Test one-hot encoding with a single categorical column."""
    X = np.array([['A'], ['B'], ['A'], ['C']], dtype=object)

    X_encoded = encode_categorical(X, categorical_columns=[0], encoding='onehot')


    # Expected: one column per category (A, B, C)
    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    np.testing.assert_array_equal(X_encoded, expected)

def test_encode_categorical_onehot_multiple_columns():
    """Test one-hot encoding with multiple categorical columns."""
    X = np.array([
        ['A', 'small'],
        ['B', 'medium'],
        ['A', 'large'],
        ['C', 'small']
    ], dtype=object)

    X_encoded = encode_categorical(X, categorical_columns=[0, 1], encoding='onehot')

    # Expected: columns for A, B, C, small, medium, large
    # Categories are ordered alphabetically by default
    expected = np.array([
        [1, 0, 0, 0, 0, 1],  # A, small
        [0, 1, 0, 0, 1, 0],  # B, medium
        [1, 0, 0, 1, 0, 0],  # A, large
        [0, 0, 1, 0, 0, 1]   # C, small
    ])

    np.testing.assert_array_equal(X_encoded, expected)

def test_encode_categorical_onehot_mixed_columns(X_categorical):
    """Test one-hot encoding with mix of categorical and numeric columns."""
    X_encoded = encode_categorical(X_categorical, categorical_columns=[0, 2], encoding='onehot')

    # Expected: numeric column in the middle, followed by one-hot encoded columns
    # For columns 0 (A, B, C) and 2 (large, medium, small)
    assert X_encoded.shape == (4, 7)  # 1 numeric + 3 for col0 + 3 for col2

    # Check numeric column preserved at position 0
    np.testing.assert_array_equal(X_encoded[:, 0], np.array([1.0, 2.0, 3.0, 4.0]))

def test_encode_categorical_label(X_categorical):
    """Test label encoding."""
    X_encoded = encode_categorical(X_categorical, categorical_columns=[0, 2], encoding='label')

    # Expected:
    # Column 0: A->0, B->1, C->2
    # Column 2: large->0, medium->1, small->2 (alphabetical order)
    expected_col0 = np.array([0, 1, 0, 2])
    expected_col2 = np.array([2, 1, 0, 2])

    np.testing.assert_array_equal(X_encoded[:, 0], expected_col0)
    np.testing.assert_array_equal(X_encoded[:, 2], expected_col2)

    # Check numeric column preserved
    np.testing.assert_array_equal(X_encoded[:, 1], np.array([1.0, 2.0, 3.0, 4.0]))

def test_encode_categorical_empty_list(X_categorical):
    """Test encoding with empty categorical_columns list."""
    X_encoded = encode_categorical(X_categorical, categorical_columns=[], encoding='onehot')

    # Should return unchanged array
    np.testing.assert_array_equal(X_encoded, X_categorical)

# Configure pytest for running all tests
if __name__ == "__main__":
    pytest.main(["-v"])