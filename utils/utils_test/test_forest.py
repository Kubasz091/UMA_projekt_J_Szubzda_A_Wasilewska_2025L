import pytest
import numpy as np
import sys
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ModifiedRandomForest import ModifiedRandomForest
from utils.tree import DecisionTree
from utils.sampling import uniform_distribution, normalize_weights
from utils.prediction import accuracy, weighted_majority_vote, majority_vote

# ------------------- Fixtures -------------------

@pytest.fixture
def binary_data():
    """Simple binary classification data."""
    # Create linearly separable data
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(40, 4) + np.array([2, 2, 2, 2]),
        np.random.randn(40, 4) + np.array([-2, -2, -2, -2])
    ])
    y = np.hstack([np.zeros(40), np.ones(40)])
    return X, y

@pytest.fixture
def multiclass_data():
    """Simple multi-class classification data."""
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(30, 5) + np.array([2, 2, 2, 2, 2]),
        np.random.randn(30, 5) + np.array([-2, -2, -2, -2, -2]),
        np.random.randn(30, 5) + np.array([5, 5, 5, 5, 5])
    ])
    y = np.hstack([np.zeros(30), np.ones(30), 2 * np.ones(30)])
    return X, y

@pytest.fixture
def correlated_feature_data():
    """Data with highly correlated features."""
    np.random.seed(42)
    n_samples = 80

    # Create a primary feature that determines the class
    primary_feature = np.random.randn(n_samples)

    # Create correlated features (correlated with primary_feature)
    noise = np.random.randn(n_samples, 4) * 0.1
    X = np.column_stack([
        primary_feature,
        primary_feature + noise[:, 0],
        primary_feature + noise[:, 1],
        primary_feature * 0.5 + noise[:, 2],
        primary_feature * 0.3 + noise[:, 3]
    ])

    # Class is determined by the sign of primary feature
    y = (primary_feature > 0).astype(int)

    return X, y

@pytest.fixture
def mock_tree():
    """Create a mock decision tree for testing."""
    tree = MagicMock(spec=DecisionTree)
    tree.predict.return_value = np.array([0, 1, 0, 1])
    tree.feature_importances_ = np.array([0.5, 0.3, 0.2])
    tree.to_dict.return_value = {'mock': 'tree'}
    return tree

@pytest.fixture
def simple_forest(binary_data):
    """Create a simple random forest for testing."""
    X, y = binary_data

    # Create a forest with minimal parameters for faster testing
    forest = ModifiedRandomForest(
        n_trees=5,
        max_depth=3,
        sample_fraction=0.5,
        random_state=42
    )
    forest.fit(X, y)

    return forest

# ------------------- Tests for Initialization -------------------

def test_random_forest_initialization():
    """Test ModifiedRandomForest initialization with default and custom parameters."""
    # Test default parameters
    default_forest = ModifiedRandomForest()
    assert default_forest.n_trees == 100
    assert default_forest.max_features == 'sqrt'
    assert default_forest.sample_fraction == 0.7
    assert default_forest.max_depth is None
    assert default_forest.prune == False
    assert default_forest.criterion == 'gini'
    assert default_forest.weighted_voting == False
    assert default_forest.error_weight_increase == 0.0
    assert default_forest.weighted_feature_sampling == False
    assert default_forest.min_samples_split == 2
    assert default_forest.min_samples_leaf == 1
    assert default_forest.random_state is None
    assert default_forest.forest == []
    assert default_forest.oob_accuracies == []
    assert default_forest.feature_importances_ is None

    # Test custom parameters
    custom_forest = ModifiedRandomForest(
        n_trees=50,
        max_features='log2',
        sample_fraction=0.8,
        max_depth=5,
        prune=True,
        criterion='entropy',
        weighted_voting=True,
        error_weight_increase=0.1,
        weighted_feature_sampling=True,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    assert custom_forest.n_trees == 50
    assert custom_forest.max_features == 'log2'
    assert custom_forest.sample_fraction == 0.8
    assert custom_forest.max_depth == 5
    assert custom_forest.prune == True
    assert custom_forest.criterion == 'entropy'
    assert custom_forest.weighted_voting == True
    assert custom_forest.error_weight_increase == 0.1
    assert custom_forest.weighted_feature_sampling == True
    assert custom_forest.min_samples_split == 5
    assert custom_forest.min_samples_leaf == 2
    assert custom_forest.random_state == 42

def test_get_max_features():
    """Test the _get_max_features method with different configurations."""
    forest = ModifiedRandomForest()

    # Test 'sqrt' option
    forest.max_features = 'sqrt'
    assert forest._get_max_features(16) == 4  # sqrt(16) = 4

    # Test 'log2' option
    forest.max_features = 'log2'
    assert forest._get_max_features(16) == 5  # log2(16) + 1 = 5

    # Test with float value (proportion)
    forest.max_features = 0.5
    assert forest._get_max_features(20) == 10  # 50% of 20 = 10

    # Test with integer value (number of features)
    forest.max_features = 7
    assert forest._get_max_features(10) == 7

    # Test with integer value larger than available features
    forest.max_features = 15
    assert forest._get_max_features(10) == 10  # Should be capped at n_features

    # Test with other string value (should return all features)
    forest.max_features = 'all'
    assert forest._get_max_features(10) == 10

# ------------------- Tests for Single Tree Building -------------------

def test_build_single_tree(binary_data):
    """Test the _build_single_tree method."""
    X, y = binary_data
    n_samples = X.shape[0]

    forest = ModifiedRandomForest(
        n_trees=5,
        max_depth=3,
        random_state=42
    )

    # Initialize feature importances
    forest.feature_importances_ = np.zeros(X.shape[1])

    # Create uniform sample weights
    sample_weights = uniform_distribution(n_samples)

    # Build a single tree
    tree, oob_accuracy, tree_importances = forest._build_single_tree(X, y, sample_weights, 0)

    # Check returned objects
    assert isinstance(tree, DecisionTree)
    assert isinstance(oob_accuracy, float)
    assert isinstance(tree_importances, np.ndarray)
    assert len(tree_importances) == X.shape[1]

    # Ensure OOB accuracy is between 0 and 1
    assert 0.0 <= oob_accuracy <= 1.0

def test_build_single_tree_with_weighted_sampling(binary_data):
    """Test building a tree with weighted feature sampling."""
    X, y = binary_data
    n_samples = X.shape[0]

    forest = ModifiedRandomForest(
        n_trees=5,
        max_depth=3,
        weighted_feature_sampling=True,
        random_state=42
    )

    # Initialize feature importances
    forest.feature_importances_ = np.zeros(X.shape[1])

    # Create non-uniform sample weights
    sample_weights = uniform_distribution(n_samples)
    # Make first 10 samples more important
    sample_weights[:10] *= 2
    sample_weights = normalize_weights(sample_weights)

    # Build a single tree
    tree, oob_accuracy, tree_importances = forest._build_single_tree(X, y, sample_weights, 0)

    assert isinstance(tree, DecisionTree)

def test_build_single_tree_with_pruning(binary_data):
    """Test building a tree with pruning enabled."""
    X, y = binary_data
    n_samples = X.shape[0]

    forest = ModifiedRandomForest(
        n_trees=5,
        max_depth=5,  # Deeper tree to ensure pruning has an effect
        prune=True,
        random_state=42
    )

    # Initialize feature importances
    forest.feature_importances_ = np.zeros(X.shape[1])

    # Create uniform sample weights
    sample_weights = uniform_distribution(n_samples)

    # Build a single tree
    with patch('utils.ModifiedRandomForest.prune_tree') as mock_prune:
        mock_prune.side_effect = lambda tree, X, y: tree  # Return the tree unchanged

        forest._build_single_tree(X, y, sample_weights, 0)

        # Verify prune_tree was called
        assert mock_prune.called

# ------------------- Tests for Forest Fitting -------------------

def test_random_forest_fit_basic(binary_data):
    """Test basic fitting of a random forest."""
    X, y = binary_data

    forest = ModifiedRandomForest(
        n_trees=5,
        max_depth=3,
        random_state=42
    )

    # Fit the forest
    result = forest.fit(X, y)

    # Check return value
    assert result is forest  # Should return self for chaining

    # Check that the forest was built correctly
    assert len(forest.forest) == 5  # Number of trees
    assert len(forest.oob_accuracies) == 5  # OOB accuracy for each tree
    assert forest.feature_importances_ is not None
    assert len(forest.feature_importances_) == X.shape[1]
    assert forest.n_classes_ == 2

def test_random_forest_fit_with_weighted_voting(binary_data):
    """Test fitting a forest with weighted voting."""
    X, y = binary_data

    forest = ModifiedRandomForest(
        n_trees=5,
        max_depth=3,
        weighted_voting=True,
        random_state=42
    )

    forest.fit(X, y)

    # Check OOB accuracies are set and used for weighting
    assert all(acc >= 0.0 for acc in forest.oob_accuracies)

    # Test prediction with weighted voting
    with patch('utils.ModifiedRandomForest.weighted_majority_vote') as mock_weighted_vote:
        mock_weighted_vote.return_value = np.zeros(len(X))

        forest.predict(X)

        # Verify weighted_majority_vote was called
        assert mock_weighted_vote.called

def test_random_forest_fit_with_weight_increase(binary_data):
    """Test fitting with sample weight increases for errors."""
    X, y = binary_data

    forest = ModifiedRandomForest(
        n_trees=3,
        max_depth=2,
        error_weight_increase=0.2,
        random_state=42
    )

    # Mock the predict method to always return errors for some samples
    original_predict = ModifiedRandomForest.predict

    try:
        # Mock predict to simulate consistent errors on specific samples
        def mock_predict(self, X, weighted_voting=None):
            predictions = np.ones(len(X))
            predictions[:20] = 0  # First 20 predictions match class 0
            return predictions

        ModifiedRandomForest.predict = mock_predict

        # Fit should update weights based on errors
        forest.fit(X, y)

    finally:
        # Restore the original predict method
        ModifiedRandomForest.predict = original_predict

def test_random_forest_fit_multiclass(multiclass_data):
    """Test fitting with multiclass data."""
    X, y = multiclass_data

    forest = ModifiedRandomForest(
        n_trees=5,
        max_depth=3,
        random_state=42
    )

    forest.fit(X, y)

    assert forest.n_classes_ == 3

    # Test that predictions contain all classes
    predictions = forest.predict(X)
    assert set(np.unique(predictions)) <= set([0, 1, 2])

def test_random_forest_fit_with_correlated_features(correlated_feature_data):
    """Test fitting with correlated features to check feature importance."""
    X, y = correlated_feature_data

    forest = ModifiedRandomForest(
        n_trees=10,
        max_depth=3,
        random_state=42  # Use standard random seed
    )

    forest.fit(X, y)

    # Check if feature importances are calculated
    assert forest.feature_importances_ is not None

    # Since features are correlated, check that importance is distributed among them
    # Some features should have significant importance (>0.1 total)
    assert np.sum(forest.feature_importances_) > 0
    assert np.max(forest.feature_importances_) > 0.1

    # The top 3 features should account for most of the importance
    # since they're all derived from the same primary feature
    top_features_importance = np.sort(forest.feature_importances_)[-3:].sum()
    assert top_features_importance > 0.5  # >50% of importance in top 3 features

# ------------------- Tests for Prediction -------------------

def test_random_forest_predict(simple_forest, binary_data):
    """Test basic prediction functionality."""
    X, y = binary_data

    # Make predictions
    predictions = simple_forest.predict(X)

    # Check that predictions have the right shape and values
    assert predictions.shape == (len(X),)
    assert set(np.unique(predictions)) <= set([0, 1])

    # Calculate accuracy (should be good on training data)
    acc = np.mean(predictions == y)
    assert acc > 0.8  # High accuracy expected on training data

def test_random_forest_predict_single_sample(simple_forest, binary_data):
    """Test prediction with a single sample."""
    X, _ = binary_data

    # Extract a single sample
    single_sample = X[0:1]

    # Make prediction
    pred = simple_forest.predict(single_sample)

    # Check prediction
    assert pred.shape == (1,)
    assert pred[0] in [0, 1]

def test_random_forest_predict_with_weighted_voting(binary_data):
    """Test prediction with weighted voting."""
    X, y = binary_data

    forest = ModifiedRandomForest(
        n_trees=5,
        max_depth=3,
        weighted_voting=True,
        random_state=42
    )
    forest.fit(X, y)

    # Normal prediction should use weighted voting
    with patch('utils.ModifiedRandomForest.weighted_majority_vote') as mock_weighted_vote:
        mock_weighted_vote.return_value = np.zeros(len(X))

        forest.predict(X)

        # Verify weighted_majority_vote was called
        assert mock_weighted_vote.called

    # Explicit non-weighted prediction
    with patch('utils.ModifiedRandomForest.majority_vote') as mock_majority_vote:
        mock_majority_vote.return_value = np.zeros(len(X))

        forest.predict(X, weighted_voting=False)

        # Verify majority_vote was called
        assert mock_majority_vote.called

def test_random_forest_predict_proba(simple_forest, binary_data):
    """Test predict_proba method."""
    X, _ = binary_data

    # Get probability predictions
    probas = simple_forest.predict_proba(X)

    # Check shape and values
    assert probas.shape == (len(X), 2)  # Binary classification, so 2 classes
    assert np.all(probas >= 0.0)
    assert np.all(probas <= 1.0)
    assert np.allclose(np.sum(probas, axis=1), 1.0)  # Probabilities should sum to 1

    # Check if the class with higher probability matches the class prediction
    predictions = simple_forest.predict(X)
    predicted_classes_from_proba = np.argmax(probas, axis=1)

    assert np.all(predictions == predicted_classes_from_proba)

# ------------------- Tests for Model Serialization -------------------

def test_random_forest_save_and_load_model(simple_forest, binary_data):
    """Test saving and loading the random forest model."""
    X, y = binary_data

    # Get predictions from the original model
    original_predictions = simple_forest.predict(X)

    # Save the model to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        temp_filename = tmp.name

    try:
        # Save model
        simple_forest.save_model(temp_filename)

        # Check file exists and is not empty
        assert os.path.exists(temp_filename)
        assert os.path.getsize(temp_filename) > 0

        # Load model
        loaded_forest = ModifiedRandomForest.load_model(temp_filename)

        # Check parameters
        assert loaded_forest.n_trees == simple_forest.n_trees
        assert loaded_forest.max_depth == simple_forest.max_depth
        assert loaded_forest.sample_fraction == simple_forest.sample_fraction
        assert loaded_forest.random_state == simple_forest.random_state
        assert len(loaded_forest.forest) == len(simple_forest.forest)

        # Check predictions match
        loaded_predictions = loaded_forest.predict(X)
        assert np.all(loaded_predictions == original_predictions)

    finally:
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

def test_random_forest_to_dict_conversion():
    """Test the internal tree to dictionary conversion for serialization."""
    forest = ModifiedRandomForest(n_trees=3, max_depth=2, random_state=42)

    # Mock trees
    tree1 = MagicMock()
    tree1.to_dict.return_value = {'tree': 1}

    tree2 = MagicMock()
    tree2.to_dict.return_value = {'tree': 2}

    tree3 = MagicMock()
    tree3.to_dict.return_value = {'tree': 3}

    forest.forest = [tree1, tree2, tree3]
    forest.oob_accuracies = [0.8, 0.9, 0.85]
    forest.feature_importances_ = np.array([0.3, 0.3, 0.4])
    forest.n_classes_ = 2

    # Patch open to avoid file I/O
    mock_open = MagicMock()
    with patch('builtins.open', mock_open):
        with patch('json.dump') as mock_json_dump:
            forest.save_model('dummy.json')

            # Check json.dump was called
            assert mock_json_dump.called

            # Get the dictionary that was saved
            saved_dict = mock_json_dump.call_args[0][0]

            # Check dictionary structure
            assert 'params' in saved_dict
            assert 'forest' in saved_dict
            assert 'oob_accuracies' in saved_dict
            assert 'feature_importances_' in saved_dict
            assert 'n_classes_' in saved_dict

            # Check forest conversion
            assert len(saved_dict['forest']) == 3
            assert saved_dict['forest'][0] == {'tree': 1}
            assert saved_dict['forest'][1] == {'tree': 2}
            assert saved_dict['forest'][2] == {'tree': 3}

# ------------------- Edge Cases and Corner Cases -------------------

def test_random_forest_with_empty_data():
    """Test behavior with empty data."""
    # Create empty data
    X = np.array([]).reshape(0, 3)
    y = np.array([])

    forest = ModifiedRandomForest(n_trees=5)

    # Check that it raises an appropriate error when given empty data
    with pytest.raises(IndexError) as excinfo:
        forest.fit(X, y)

    # Verify the error message is about empty data
    assert "out of bounds" in str(excinfo.value) and "size 0" in str(excinfo.value)

def test_random_forest_with_single_sample():
    """Test with a single sample."""
    X = np.array([[1, 2, 3]])
    y = np.array([0])

    forest = ModifiedRandomForest(n_trees=3)

    # This should work, but with bootstrap, all trees might be identical
    forest.fit(X, y)

    # Should still make predictions
    pred = forest.predict(X)
    assert pred[0] == y[0]

def test_random_forest_with_single_class():
    """Test with a dataset containing only one class."""
    X = np.random.randn(10, 3)
    y = np.zeros(10)  # All samples are class 0

    forest = ModifiedRandomForest(n_trees=3)
    forest.fit(X, y)

    # All predictions should be class 0
    preds = forest.predict(X)
    assert np.all(preds == 0)

    # Check n_classes_
    assert forest.n_classes_ == 1

def test_random_forest_with_list_inputs():
    """Test with list inputs instead of numpy arrays."""
    X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [0, 0, 1, 1]

    forest = ModifiedRandomForest(n_trees=3, random_state=42)
    forest.fit(X, y)

    # Should convert to numpy arrays internally
    assert forest.n_classes_ == 2

    # Predict should also accept lists
    pred = forest.predict([[1, 2]])
    assert pred[0] in [0, 1]

def test_random_forest_with_different_n_trees():
    """Test with different numbers of trees."""
    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)  # Simple binary classification

    # Build forests with different numbers of trees
    forest1 = ModifiedRandomForest(n_trees=1, random_state=42)
    forest10 = ModifiedRandomForest(n_trees=10, random_state=42)

    forest1.fit(X, y)
    forest10.fit(X, y)

    # Check forest sizes
    assert len(forest1.forest) == 1
    assert len(forest10.forest) == 10

    # Forest with more trees should typically have more stable predictions
    # But we can't test that directly in a unit test reliably

def test_random_forest_predict_with_empty_forest():
    """Test prediction behavior with an empty forest."""
    X = np.random.randn(10, 3)

    forest = ModifiedRandomForest()
    # Don't call fit, leaving forest empty

    try:
        forest.predict(X)
        assert False, "Should raise an exception with empty forest"
    except Exception as e:
        # Expect a meaningful error
        assert "not fit" in str(e).lower() or "empty" in str(e).lower() or "no forest" in str(e).lower()

# ------------------- Integration Tests -------------------

def test_random_forest_performance_comparison():
    """Compare performance of different forest configurations."""
    # Create a dataset with a clear pattern
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)  # Simple rule

    # Split into train and test
    train_idx = np.random.choice(100, 70, replace=False)
    test_idx = np.array([i for i in range(100) if i not in train_idx])

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Create different forest configurations
    basic_forest = ModifiedRandomForest(n_trees=10, random_state=42)
    weighted_forest = ModifiedRandomForest(n_trees=10, weighted_voting=True, random_state=42)
    pruned_forest = ModifiedRandomForest(n_trees=10, prune=True, random_state=42)

    # Train models
    basic_forest.fit(X_train, y_train)
    weighted_forest.fit(X_train, y_train)
    pruned_forest.fit(X_train, y_train)

    # Make predictions
    basic_preds = basic_forest.predict(X_test)
    weighted_preds = weighted_forest.predict(X_test)
    pruned_preds = pruned_forest.predict(X_test)

    # Calculate accuracies
    basic_acc = accuracy(y_test, basic_preds)
    weighted_acc = accuracy(y_test, weighted_preds)
    pruned_acc = accuracy(y_test, pruned_preds)

    # All should perform reasonably well
    assert basic_acc > 0.6
    assert weighted_acc > 0.6
    assert pruned_acc > 0.6

def test_random_forest_feature_importance():
    """Test that feature importance correctly identifies important features."""
    # Create dataset where only the first feature matters
    np.random.seed(42)
    X = np.random.randn(100, 9)
    y = (X[:, 0] > 0).astype(int)  # Only first feature matters

    forest = ModifiedRandomForest(n_trees=100, random_state=42)
    forest.fit(X, y)

    print(forest.feature_importances_)

    # First feature should have the highest importance
    assert np.argmax(forest.feature_importances_) == 0

    # First feature should be significantly more important
    assert forest.feature_importances_[0] > 2 * np.mean(forest.feature_importances_[1:])

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])