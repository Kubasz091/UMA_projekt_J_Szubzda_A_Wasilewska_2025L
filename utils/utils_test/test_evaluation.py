import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation import confusion_matrix, calculate_metrics, roc_auc, cross_validate

# ------------------- Fixtures -------------------

@pytest.fixture
def binary_classification_data():
    """Binary classification data fixture."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
    y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8],
                        [0.7, 0.3], [0.4, 0.6], [0.9, 0.1], [0.3, 0.7],
                        [0.2, 0.8], [0.6, 0.4]])
    return y_true, y_pred, y_proba

@pytest.fixture
def multiclass_classification_data():
    """Multiclass classification data fixture."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 2, 1, 0, 0])
    y_proba = np.array([
        [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.6, 0.3],
        [0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7],
        [0.3, 0.3, 0.4], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4],
        [0.6, 0.2, 0.2]
    ])
    return y_true, y_pred, y_proba

@pytest.fixture
def perfect_predictions():
    """Perfect predictions fixture."""
    y_true = np.array([0, 1, 0, 1, 2])
    y_pred = np.array([0, 1, 0, 1, 2])
    return y_true, y_pred

@pytest.fixture
def single_class_data():
    """Single class data fixture."""
    y_true = np.array([1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 1])
    return y_true, y_pred

@pytest.fixture
def mock_model():
    """Mock model for cross-validation testing."""
    class MockModel:
        def __init__(self, param1=None, param2=None):
            self.param1 = param1
            self.param2 = param2
            self.fitted = False

        def fit(self, X, y):
            self.fitted = True
            self.X_train_shape = X.shape
            self.y_train_shape = y.shape
            return self

        def predict(self, X):
            # Simple mock prediction: predict class 0 for first half, 1 for second half
            predictions = np.zeros(X.shape[0])
            predictions[X.shape[0]//2:] = 1
            return predictions

    return MockModel

@pytest.fixture
def imbalanced_data():
    """Imbalanced classification data."""
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return y_true, y_pred

@pytest.fixture
def non_consecutive_labels():
    """Non-consecutive class labels."""
    y_true = np.array([1, 5, 1, 5, 10, 10])
    y_pred = np.array([5, 5, 1, 1, 1, 10])
    return y_true, y_pred

# ------------------- Tests for confusion_matrix -------------------

def test_confusion_matrix_binary(binary_classification_data):
    """Test basic confusion matrix calculation for binary classification."""
    y_true, y_pred, _ = binary_classification_data
    cm = confusion_matrix(y_true, y_pred)

    # Expected: [[3, 2], [2, 3]]
    expected = np.array([[3, 2], [2, 3]])
    np.testing.assert_array_equal(cm, expected)

    # Test shape and data type
    assert cm.shape == (2, 2)
    assert isinstance(cm, np.ndarray)

def test_confusion_matrix_multiclass(multiclass_classification_data):
    """Test confusion matrix calculation for multiclass."""
    y_true, y_pred, _ = multiclass_classification_data
    cm = confusion_matrix(y_true, y_pred)

    # Expected: [[3, 0, 1], [0, 3, 0], [1, 1, 1]]
    expected = np.array([[3, 0, 1], [0, 3, 0], [1, 1, 1]])
    np.testing.assert_array_equal(cm, expected)

    assert cm.shape == (3, 3)

def test_confusion_matrix_normalized(binary_classification_data):
    """Test normalized confusion matrix."""
    y_true, y_pred, _ = binary_classification_data
    cm = confusion_matrix(y_true, y_pred, normalize=True)

    # Expected: [[0.6, 0.4], [0.4, 0.6]]
    expected = np.array([[0.6, 0.4], [0.4, 0.6]])
    np.testing.assert_almost_equal(cm, expected)

def test_confusion_matrix_perfect(perfect_predictions):
    """Test confusion matrix with perfect predictions."""
    y_true, y_pred = perfect_predictions
    cm = confusion_matrix(y_true, y_pred)

    # Expected diagonal matrix
    expected = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    np.testing.assert_array_equal(cm, expected)

def test_confusion_matrix_single_class(single_class_data):
    """Test confusion matrix with single class."""
    y_true, y_pred = single_class_data
    cm = confusion_matrix(y_true, y_pred)

    # Single class in y_true, but y_pred contains 0 and 1
    expected = np.array([[0, 0], [1, 4]])
    np.testing.assert_array_equal(cm, expected)

def test_confusion_matrix_imbalanced(imbalanced_data):
    """Test confusion matrix with imbalanced data."""
    y_true, y_pred = imbalanced_data
    cm = confusion_matrix(y_true, y_pred)

    expected = np.array([[9, 0], [1, 0]])
    np.testing.assert_array_equal(cm, expected)

def test_confusion_matrix_non_consecutive(non_consecutive_labels):
    """Test confusion matrix with non-consecutive labels."""
    y_true, y_pred = non_consecutive_labels
    cm = confusion_matrix(y_true, y_pred)

    # Expected: labels 1, 5, 10
    expected = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 1]])
    np.testing.assert_array_equal(cm, expected)

def test_confusion_matrix_empty():
    """Test confusion matrix with empty arrays."""
    cm = confusion_matrix(np.array([]), np.array([]))

    # Should return an empty array
    assert cm.size == 0
    assert cm.shape == (0, 0)

# ------------------- Additional Confusion Matrix Edge Cases -------------------

def test_confusion_matrix_with_string_indices():
    """Test confusion matrix with string class labels."""
    y_true = np.array(["cat", "dog", "cat", "bird", "dog"])
    y_pred = np.array(["cat", "cat", "dog", "dog", "dog"])

    cm = confusion_matrix(y_true, y_pred)

    # Order should be alphabetical: bird, cat, dog
    expected = np.array([
        [0, 0, 1],  # bird -> dog
        [0, 1, 1],  # cat -> (cat=1, dog=1)
        [0, 1, 1]   # dog -> (cat=1, dog=1)
    ])
    np.testing.assert_array_equal(cm, expected)

def test_confusion_matrix_normalization_zero_rows():
    """Test normalization with rows that have zero samples."""
    # Create a case where one class has no samples in y_true
    y_true = np.array([0, 0, 0, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 2])

    # Without normalization
    cm = confusion_matrix(y_true, y_pred)
    expected = np.array([
        [1, 2, 0],  # class 0
        [0, 0, 0],  # class 1 (no samples)
        [1, 0, 1]   # class 2
    ])
    np.testing.assert_array_equal(cm, expected)

    # With normalization
    cm_norm = confusion_matrix(y_true, y_pred, normalize=True)
    expected_norm = np.array([
        [1/3, 2/3, 0],  # class 0 (3 samples)
        [0, 0, 0],       # class 1 (0 samples)
        [1/2, 0, 1/2]    # class 2 (2 samples)
    ])
    np.testing.assert_almost_equal(cm_norm, expected_norm, decimal=5)

def test_confusion_matrix_with_large_sparse_labels():
    """Test confusion matrix with large sparse label set."""
    # Labels with large gaps
    y_true = np.array([0, 100, 1000, 0, 1000])
    y_pred = np.array([0, 0, 0, 100, 1000])

    cm = confusion_matrix(y_true, y_pred)

    # Should correctly handle large label values without creating huge array
    assert cm.shape == (3, 3)
    expected = np.array([
        [1, 1, 0],    # class 0
        [1, 0, 0],    # class 100
        [1, 0, 1]     # class 1000
    ])
    np.testing.assert_array_equal(cm, expected)

# ------------------- Tests for calculate_metrics -------------------

def test_calculate_metrics_binary(binary_classification_data):
    """Test basic metric calculation for binary classification."""
    y_true, y_pred, y_proba = binary_classification_data
    metrics = calculate_metrics(y_true, y_pred, y_proba)

    # Test that all expected metrics exist
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    # Test binary-specific metrics exist
    assert "pos_class_precision" in metrics
    assert "pos_class_recall" in metrics
    assert "pos_class_f1" in metrics

    # Test accuracy calculation
    assert metrics["accuracy"] == 0.6  # 6 correct out of 10

    # Test basic metric ranges
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1

    # Verify specific values for this dataset
    # For positive class (class 1): TP=3, FP=2, FN=2
    assert abs(metrics["pos_class_precision"] - 0.6) < 0.01  # 3/5 = 0.6
    assert abs(metrics["pos_class_recall"] - 0.6) < 0.01     # 3/5 = 0.6

def test_calculate_metrics_binary_proba_shape(binary_classification_data):
    """Test metrics with different probability shapes."""
    y_true, y_pred, y_proba = binary_classification_data

    # Test with 2D probabilities
    metrics_2d = calculate_metrics(y_true, y_pred, y_proba)

    # Test with 1D probabilities (just positive class)
    metrics_1d = calculate_metrics(y_true, y_pred, y_proba[:, 1])

    # AUC should be the same regardless of probability format
    assert metrics_2d["roc_auc"] == metrics_1d["roc_auc"]

def test_calculate_metrics_multiclass(multiclass_classification_data):
    """Test metrics for multiclass classification."""
    y_true, y_pred, _ = multiclass_classification_data
    metrics = calculate_metrics(y_true, y_pred)

    # Check basic metrics exist
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    # Check multiclass doesn't have binary-specific metrics
    assert "pos_class_precision" not in metrics
    assert "roc_auc" not in metrics  # ROC AUC not calculated for multiclass without probas

    # Check basic accuracy
    assert metrics["accuracy"] == 0.7  # 7 correct out of 10

    # Check metric ranges
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1

def test_calculate_metrics_perfect(perfect_predictions):
    """Test metrics with perfect predictions."""
    y_true, y_pred = perfect_predictions
    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0

def test_calculate_metrics_imbalanced(imbalanced_data):
    """Test metrics with imbalanced data."""
    y_true, y_pred = imbalanced_data
    metrics = calculate_metrics(y_true, y_pred)

    # Basic accuracy check
    assert metrics["accuracy"] == 0.9  # 9/10 correct predictions

    # For scikit-learn implementation with weighted averaging:
    # Precision is weighted by predicted samples in each class
    assert abs(metrics["precision"] - 0.81) < 0.01

    # Recall is weighted by actual samples in each class
    # Class 0: recall 1.0 (all 9 samples found) with weight 0.9
    # Class 1: recall 0.0 (no samples found) with weight 0.1
    # Weighted recall = 1.0 * 0.9 + 0.0 * 0.1 = 0.9
    assert abs(metrics["recall"] - 0.9) < 0.01

    # F1 is derived from the harmonic mean of precision and recall
    # With the weighted versions of these metrics
    assert abs(metrics["f1"] - 0.85) < 0.01

    # Check binary-specific metrics
    assert metrics["pos_class_precision"] == 0.0  # No true positives
    assert metrics["pos_class_recall"] == 0.0     # No true positives
    assert metrics["pos_class_f1"] == 0.0         # No true positives

def test_calculate_metrics_invalid_proba():
    """Test metrics with invalid probability input."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])

    # Test with malformed probabilities (wrong shape)
    y_proba = np.array([0.1, 0.2, 0.3])  # Wrong length
    metrics = calculate_metrics(y_true, y_pred, y_proba)

    # Should still calculate basic metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert metrics["accuracy"] == 0.5

def test_calculate_metrics_no_positive_samples():
    """Test metrics when a class has no positive samples in ground truth."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 1, 1])

    metrics = calculate_metrics(y_true, y_pred)

    # Weighted metrics for this case
    assert metrics["accuracy"] == 0.5  # 2/4 correct

    # With weighted averaging:
    # Precision class 0: 2/2 = 1.0, weight = 4/4 = 1.0
    # No class 1 in ground truth
    assert metrics["precision"] == 1.0  # All predicted class 0 are correct
    assert metrics["recall"] == 0.5     # Half of all class 0 samples correctly identified

    # F1 derived from weighted precision and recall
    expected_f1 = 2*(1.0*0.5)/(1.0+0.5)  # 2*0.5/1.5 = 1.0/1.5 = 2/3
    assert abs(metrics["f1"] - expected_f1) < 0.01

    # Check specific metrics for class 1 (which doesn't exist in ground truth)
    assert "pos_class_precision" in metrics
    assert metrics["pos_class_precision"] == 0.0  # No true positives for class 1

# ------------------- Additional Edge Case Tests for calculate_metrics -------------------

def test_calculate_metrics_with_class_specific_metrics(binary_classification_data):
    """Test that class-specific metrics are properly calculated for binary case."""
    y_true, y_pred, _ = binary_classification_data
    metrics = calculate_metrics(y_true, y_pred)

    # Verify class-specific metrics exist
    assert "pos_class_precision" in metrics
    assert "pos_class_recall" in metrics
    assert "pos_class_f1" in metrics

    # Rather than checking they're different (which isn't true for this dataset),
    # check that the calculation is correct for the positive class metrics

    # Calculate expected values manually for class 1
    # From binary_classification_data fixture:
    # y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    # y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    # For class 1: TP=3, FP=2, FN=2
    expected_precision = 3/(3+2)  # 3/5 = 0.6
    expected_recall = 3/(3+2)     # 3/5 = 0.6

    assert abs(metrics["pos_class_precision"] - expected_precision) < 0.01
    assert abs(metrics["pos_class_recall"] - expected_recall) < 0.01

    # Verify that weighted metrics are calculated using all classes
    # (Though in this balanced case they happen to equal the positive class metrics)
    assert "precision" in metrics
    assert "recall" in metrics
    assert abs(metrics["precision"] - 0.6) < 0.01

def test_calculate_metrics_with_numpy_dtypes():
    """Test metrics with various numpy data types."""
    # Test with bool
    y_true_bool = np.array([True, False, True, False])
    y_pred_bool = np.array([True, False, False, True])

    metrics_bool = calculate_metrics(y_true_bool, y_pred_bool)
    assert metrics_bool["accuracy"] == 0.5

    # Test with int8, uint8, int32, int64
    dtypes = [np.int8, np.uint8, np.int32, np.int64]
    for dtype in dtypes:
        y_true_typed = np.array([0, 1, 0, 1]).astype(dtype)
        y_pred_typed = np.array([0, 1, 1, 0]).astype(dtype)

        metrics_typed = calculate_metrics(y_true_typed, y_pred_typed)
        assert metrics_typed["accuracy"] == 0.5
        assert "precision" in metrics_typed

def test_calculate_metrics_with_nan_values():
    """Test handling of NaN values in predictions."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, np.nan, 0])

    # This should handle NaN gracefully or raise a meaningful error
    try:
        metrics = calculate_metrics(y_true, y_pred)
        # If it runs, NaN should be treated as incorrect prediction
        assert metrics["accuracy"] <= 0.5
    except Exception as e:
        # Should raise a specific error about NaN values
        assert "nan" in str(e).lower() or "float" in str(e).lower()

def test_calculate_metrics_with_mixed_input_types():
    """Test metrics calculation with mixed input types."""
    # Mix of integers and strings
    y_true = np.array([0, 1, "0", "1"])
    y_pred = np.array([0, 1, "1", "0"])

    try:
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.5
    except Exception as e:
        # Should raise type error or handle it somehow
        assert "type" in str(e).lower()

def test_calculate_metrics_with_extreme_probabilities(binary_classification_data):
    """Test metrics with extremely confident but wrong predictions."""
    y_true, y_pred, _ = binary_classification_data

    # Create extreme probabilities (very confident but incorrect)
    wrong_extremes = np.zeros((len(y_true), 2))
    for i in range(len(y_true)):
        # Set probability of wrong class to 0.999
        wrong_class = 1 - y_true[i]
        wrong_extremes[i, wrong_class] = 0.999
        wrong_extremes[i, y_true[i]] = 0.001

    metrics = calculate_metrics(y_true, y_pred, wrong_extremes)

    # AUC should be very poor (close to 0)
    if "roc_auc" in metrics:
        assert metrics["roc_auc"] < 0.2

# ------------------- Tests for roc_auc -------------------

def test_roc_auc_binary(binary_classification_data):
    """Test ROC AUC calculation for binary classification."""
    y_true, _, y_proba = binary_classification_data
    auc = roc_auc(y_true, y_proba[:, 1])

    assert 0 <= auc <= 1
    assert not np.isnan(auc)

def test_roc_auc_perfect_separation():
    """Test ROC AUC with perfect separation of classes."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    auc = roc_auc(y_true, y_score)
    assert auc == 1.0

def test_roc_auc_random():
    """Test ROC AUC with random scores."""
    # Force reproducibility
    np.random.seed(42)

    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_score = np.random.random(6)  # Random scores

    auc = roc_auc(y_true, y_score)
    # For random scores, AUC should be around 0.5
    assert 0.2 <= auc <= 0.8

def test_roc_auc_single_class():
    """Test ROC AUC with single class (should return NaN)."""
    y_true = np.array([1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])

    auc = roc_auc(y_true, y_score)
    assert np.isnan(auc)

def test_roc_auc_incompatible_shapes():
    """Test ROC AUC with incompatible shapes."""
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.3])  # Wrong length

    auc = roc_auc(y_true, y_score)
    assert np.isnan(auc)

# ------------------- Additional ROC AUC Edge Cases -------------------

def test_roc_auc_with_extreme_scores():
    """Test ROC AUC with extreme probability scores (0 and 1)."""
    y_true = np.array([0, 1, 0, 1])

    # Scores are exactly 0 and 1
    y_score_perfect = np.array([0.0, 1.0, 0.0, 1.0])
    auc_perfect = roc_auc(y_true, y_score_perfect)
    assert auc_perfect == 1.0

    # All scores identical
    y_score_identical = np.array([0.5, 0.5, 0.5, 0.5])
    auc_identical = roc_auc(y_true, y_score_identical)
    # Should be 0.5 (random) or NaN
    assert np.isnan(auc_identical) or abs(auc_identical - 0.5) < 0.001

def test_roc_auc_with_pathological_predictions():
    """Test ROC AUC with pathological prediction patterns."""
    # All true labels same, predictions varied
    y_true_same = np.array([1, 1, 1, 1])
    y_pred_varied = np.array([0.1, 0.4, 0.7, 0.9])

    auc_same = roc_auc(y_true_same, y_pred_varied)
    assert np.isnan(auc_same)  # Should be NaN, can't compute AUC

    # All predictions same, true labels varied
    y_true_varied = np.array([0, 1, 0, 1])
    y_pred_same = np.array([0.5, 0.5, 0.5, 0.5])

    auc_pred_same = roc_auc(y_true_varied, y_pred_same)
    # Should be 0.5 (random classifier) or NaN
    assert np.isnan(auc_pred_same) or abs(auc_pred_same - 0.5) < 0.001

def test_roc_auc_multiclass_probability_handling(multiclass_classification_data):
    """Test how ROC AUC handles multiclass probabilities."""
    # Multiclass from fixture (passed as parameter)
    y_true, _, y_proba = multiclass_classification_data

    # Try to pass full probability matrix
    auc = roc_auc(y_true, y_proba)

    # The function should either:
    # 1. Return a valid AUC value between 0 and 1 if it supports multiclass
    # 2. Return NaN if it doesn't support multiclass
    # Either case is acceptable behavior
    assert np.isnan(auc) or (0 <= auc <= 1)

    # Try with class 0 probabilities only (should work as binary)
    auc_binary = roc_auc(y_true == 0, y_proba[:, 0])
    assert not np.isnan(auc_binary)
    assert 0 <= auc_binary <= 1


# ------------------- Tests for cross_validate -------------------

def test_cross_validate_basic(mock_model):
    """Test basic cross validation."""
    X = np.random.random((20, 5))
    y = np.random.randint(0, 2, 20)

    result = cross_validate(mock_model, X, y, n_folds=5)

    assert "accuracy_mean" in result
    assert "accuracy_std" in result
    assert "precision_mean" in result
    assert "precision_std" in result
    assert "recall_mean" in result
    assert "recall_std" in result
    assert "f1_mean" in result
    assert "f1_std" in result
    assert "fold_metrics" in result
    assert "overall" in result
    assert len(result["fold_metrics"]) == 5

def test_cross_validate_reproducibility(mock_model):
    """Test cross validation reproducibility with fixed random_state."""
    X = np.random.random((20, 5))
    y = np.random.randint(0, 2, 20)

    result1 = cross_validate(mock_model, X, y, n_folds=3, random_state=42)
    result2 = cross_validate(mock_model, X, y, n_folds=3, random_state=42)

    # Results should be identical with same random_state
    assert result1["accuracy_mean"] == result2["accuracy_mean"]
    assert result1["precision_mean"] == result2["precision_mean"]

def test_cross_validate_model_params(mock_model):
    """Test passing parameters to model in cross validation."""
    X = np.random.random((20, 5))
    y = np.random.randint(0, 2, 20)

    # Create a spy to check if parameters were correctly passed
    original_init = mock_model.__init__
    params_received = {}

    def spy_init(self, param1=None, param2=None):
        params_received["param1"] = param1
        params_received["param2"] = param2
        original_init(self, param1, param2)

    mock_model.__init__ = spy_init

    cross_validate(mock_model, X, y, n_folds=2, param1="test", param2=42)

    assert params_received["param1"] == "test"
    assert params_received["param2"] == 42

    # Restore original init
    mock_model.__init__ = original_init

def test_cross_validate_different_folds(mock_model):
    """Test cross validation with different numbers of folds."""
    X = np.random.random((30, 5))
    y = np.random.randint(0, 2, 30)

    result_3_folds = cross_validate(mock_model, X, y, n_folds=3)
    result_5_folds = cross_validate(mock_model, X, y, n_folds=5)

    assert len(result_3_folds["fold_metrics"]) == 3
    assert len(result_5_folds["fold_metrics"]) == 5

def test_cross_validate_small_dataset(mock_model):
    """Test cross validation with a very small dataset."""
    X = np.random.random((6, 5))
    y = np.random.randint(0, 2, 6)

    # With 3 folds, each fold will have just 2 samples
    result = cross_validate(mock_model, X, y, n_folds=3)

    assert len(result["fold_metrics"]) == 3
    assert "accuracy_mean" in result

def test_cross_validate_imbalanced_classes(mock_model):
    """Test cross validation with imbalanced classes."""
    X = np.random.random((20, 5))
    y = np.zeros(20)  # All zeros
    y[:2] = 1  # Just 2 ones

    result = cross_validate(mock_model, X, y, n_folds=5)

    # Should still work, but metrics might be extreme
    assert "accuracy_mean" in result
    assert "precision_mean" in result
    assert "recall_mean" in result

def test_cross_validate_error_handling():
    """Test cross validation error handling with a broken model."""
    class BrokenModel:
        def __init__(self):
            pass

        def fit(self, X, y):
            raise ValueError("Simulated error during fitting")

        def predict(self, X):
            return np.zeros(X.shape[0])

    X = np.random.random((10, 5))
    y = np.random.randint(0, 2, 10)

    # Should raise the ValueError from the fit method
    with pytest.raises(ValueError):
        cross_validate(BrokenModel, X, y, n_folds=3)

def test_cross_validate_non_numpy_inputs(mock_model):
    """Test cross validation with non-numpy inputs."""
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    y = [0, 1, 0, 1]

    # Should convert to numpy arrays internally
    result = cross_validate(mock_model, X, y, n_folds=2)

    assert "accuracy_mean" in result
    assert len(result["fold_metrics"]) == 2

# ------------------- Additional Cross Validation Edge Cases -------------------

def test_cross_validate_with_custom_splitter(mock_model):
    """Test cross-validation with a custom splitter instead of n_folds."""
    from sklearn.model_selection import KFold

    X = np.random.random((20, 5))
    y = np.random.randint(0, 2, 20)

    # Create custom CV splitter
    custom_cv = KFold(n_splits=3, shuffle=True, random_state=42)

    try:
        # Check if our function accepts a cv parameter
        result = cross_validate(mock_model, X, y, cv=custom_cv)
        assert len(result["fold_metrics"]) == 3
    except TypeError:
        # If not implemented, this should raise a TypeError about unexpected arg
        pass

def test_cross_validate_with_tiny_samples_per_class(mock_model):
    """Test cross-validation with tiny number of samples per class."""
    # Create minimal dataset with just 2 samples per class
    X = np.random.random((4, 3))
    y = np.array([0, 0, 1, 1])

    # With 2-fold CV, each fold will have just 1 sample per class
    # This tests numerical stability with extremely small validation sets
    result = cross_validate(mock_model, X, y, n_folds=2)

    assert len(result["fold_metrics"]) == 2
    assert "accuracy_mean" in result

def test_cross_validate_with_special_class_distribution(mock_model):
    """Test cross-validation with special class distributions."""
    # Test case where some folds might not have samples from all classes
    X = np.random.random((10, 3))
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8:2 imbalance

    # With 5-fold CV, some folds might not have class 1
    result = cross_validate(mock_model, X, y, n_folds=5)

    assert len(result["fold_metrics"]) == 5
    # Should handle missing classes in some folds
    assert "precision_mean" in result

def test_cross_validate_with_custom_metrics_dict(mock_model):
    """Test cross validation with a dictionary of custom metrics."""
    # Define custom metrics
    custom_metrics = {
        'tp_rate': lambda y_true, y_pred: np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_true == 1), 1),
        'tn_rate': lambda y_true, y_pred: np.sum((y_true == 0) & (y_pred == 0)) / max(np.sum(y_true == 0), 1)
    }

    X = np.random.random((20, 5))
    y = np.random.randint(0, 2, 20)

    try:
        result = cross_validate(mock_model, X, y, n_folds=3, custom_metrics=custom_metrics)
        # Check if custom metrics are included in results
        assert "tp_rate_mean" in result
        assert "tn_rate_mean" in result
    except TypeError:
        # If not implemented, this should raise TypeError about unexpected arg
        pass

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])