import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom implementations
from utils.ModifiedRandomForest import ModifiedRandomForest
from utils.evaluation import calculate_metrics, confusion_matrix
from data_cache import get_dataset
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Replace 'DatasetName' with the actual dataset name from data_cache.py
DATASET_NAME = 'DatasetName'  # Change this to the desired dataset name

# Load dataset from cache
print(f"Loading {DATASET_NAME} dataset from cache...")
X, y, cat_features, class_mapping = get_dataset(DATASET_NAME)
print(f"{DATASET_NAME} dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class mapping: {class_mapping}")
if cat_features:
    print(f"Categorical features: {cat_features}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Define and train ModifiedRandomForest with appropriate parameters
rf_params = {
    # Basic forest structure
    'n_trees': 100,                     # Number of trees in the forest
    'max_features': 'sqrt',             # Number of features to consider for splits
    'sample_fraction': 0.8,             # Fraction of samples to use for each tree

    # Tree growth control
    'max_depth': 10,                    # Maximum depth of each tree
    'min_samples_split': 2,             # Minimum samples required to split an internal node
    'min_samples_leaf': 1,              # Minimum samples required in a leaf node
    'criterion': 'gini',                # Split criterion ('gini' or 'entropy')

    # Post-processing
    'prune': False,                     # Whether to prune trees after construction

    # Weighting strategies
    'weighted_voting': True,            # Use accuracy-weighted voting for predictions
    'error_weight_increase': 0.1,       # Amount to increase weights on misclassified samples
    'weighted_feature_sampling': True,  # Use feature importance for feature sampling

    # Reproducibility
    'random_state': RANDOM_SEED
}

print("\nTraining ModifiedRandomForest...")
rf_model = ModifiedRandomForest(**rf_params)
rf_model.fit(X_train, y_train)

# Make predictions
print("\nGenerating predictions...")
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test) if hasattr(rf_model, 'predict_proba') else None

# Evaluate performance
metrics = calculate_metrics(y_test, y_pred, y_proba)
print("\nPerformance metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize=True)
class_names = [class_mapping.get(i, f"Class {i}") for i in range(len(np.unique(y)))] if class_mapping else [f"Class {i}" for i in range(len(np.unique(y)))]

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'ModifiedRandomForest Confusion Matrix on {DATASET_NAME}')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Add text annotations
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(f'figures/{DATASET_NAME.lower()}_rf_confusion_matrix.png', dpi=300, bbox_inches='tight')

# Show sample predictions
print("\nSample predictions (first 10 test samples):")
sample_results = pd.DataFrame({
    'True Class': [class_mapping.get(y, f"Class {y}") if class_mapping else f"Class {y}" for y in y_test[:10]],
    'Predicted Class': [class_mapping.get(y, f"Class {y}") if class_mapping else f"Class {y}" for y in y_pred[:10]]
})

if y_proba is not None:
    for i, class_name in enumerate(class_names):
        sample_results[f'Prob {class_name}'] = y_proba[:10, i]

print(sample_results)

# Try to access feature importances if available
if hasattr(rf_model, 'feature_importances_'):
    print("\nFeature importances:")
    feature_names = X.columns if hasattr(X, 'columns') else [f"Feature {i}" for i in range(X.shape[1])]
    importances = rf_model.feature_importances_

    # Print and plot feature importances
    indices = np.argsort(importances)[::-1]
    print("\nFeatures ranked by importance:")
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

plt.show()
print("\nResults saved to figures directory")