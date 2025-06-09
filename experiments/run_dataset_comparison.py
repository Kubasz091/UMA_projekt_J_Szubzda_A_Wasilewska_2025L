import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.ModifiedRandomForest import ModifiedRandomForest

# --- Configuration ---
BASE_RESULTS_DIR = os.path.join(project_root, 'experiments', 'results', 'dataset_comparison')
PLOTS_DIR = os.path.join(BASE_RESULTS_DIR, 'plots')
DATA_DIR = os.path.join(project_root, 'preprocessed_datasets')
# No longer need DATASET_LIST_PATH

os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Helper Functions ---
def load_preprocessed_data(dataset_name_from_list, data_dir):
    # Precise mapping from dataset_list.txt names to .pkl filenames from full_preprocessing.py
    # Selecting a primary version for each dataset for comparison.
    filename_map = {
        "Iris": "Iris_original.pkl",
        "Wine Quality (original)": "Wine_original.pkl",
        "Adult Census": "AdultCensus_basic_clean.pkl",
        "Bank Marketing": "BankMarketing_basic_imputed.pkl", # Using imputed version
        "MNIST": "MNIST_filters_only.pkl"
    }

    pkl_filename = filename_map.get(dataset_name_from_list)

    if not pkl_filename:
        print(f"Warning: No mapping found for dataset '{dataset_name_from_list}' in filename_map. Skipping.")
        return None, None, None

    pkl_path = os.path.join(data_dir, pkl_filename)

    if not os.path.exists(pkl_path):
        print(f"Warning: Processed data file {pkl_filename} for {dataset_name_from_list} not found in {data_dir}. Skipping.")
        return None, None, None

    data_dict = pd.read_pickle(pkl_path)

    X = data_dict['X']
    y = data_dict['y']
    # class_mapping = data_dict.get('class_mapping')

    if isinstance(X, pd.DataFrame):
        X = X.values

    if isinstance(y, pd.Series):
        y = y.values
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le

def calculate_metrics(y_true, y_pred, y_proba, num_classes, le, model_name):
    # For ROC AUC, ensure y_proba is correctly shaped
    roc_auc_val = np.nan
    if y_proba is not None:
        try:
            if num_classes == 2:
                 # Ensure y_proba is for the positive class
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    roc_auc_val = roc_auc_score(y_true, y_proba[:, 1])
                elif y_proba.ndim == 1: # If it's already proba of positive class
                     roc_auc_val = roc_auc_score(y_true, y_proba)
                else:
                    print(f"Warning: y_proba shape {y_proba.shape} not suitable for binary ROC AUC for {model_name}.")
            else: # multiclass
                if y_proba.ndim == 2 and y_proba.shape[1] == num_classes:
                    roc_auc_val = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                else:
                    print(f"Warning: y_proba shape {y_proba.shape} not suitable for multiclass ROC AUC for {model_name} (num_classes={num_classes}).")
        except ValueError as e:
            print(f"Warning: ROC AUC calculation error for {model_name}: {e}")
            roc_auc_val = np.nan

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_val
    }

def plot_comparison_metrics(all_results_df, save_dir):
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'fit_time']
    datasets = all_results_df['dataset'].unique()
    models = all_results_df['model'].unique()

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 7))

        # Create a pivot table for easier plotting
        pivot_df = all_results_df.pivot(index='dataset', columns='model', values=metric)

        if pivot_df.empty:
            print(f"Skipping plot for {metric} as no data is available.")
            continue

        pivot_df.plot(kind='bar', ax=plt.gca())

        plt.title(f'{metric.replace("_", " ").title()} Comparison Across Datasets')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xlabel('Dataset')
        plt.xticks(rotation=45, ha="right")
        plt.legend(title='Model')
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()

        plot_path = os.path.join(save_dir, f'comparison_{metric}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved comparison plot: {plot_path}")

# --- Main Experiment Logic ---
# Hardcoded dataset names (ensure these match keys in filename_map)
dataset_names = [
    "Iris",
    "Wine Quality (original)",
    "Adult Census",
    "Bank Marketing",
    "MNIST"
]

# Default parameters for ModifiedRandomForest (from main.tex Phase 1 defaults)
mrf_default_params = {
    'n_trees': 50,
    'sample_fraction': 0.7,
    'max_depth': None,
    'prune': False,
    'criterion': 'gini',
    'error_weight_increase': 0.1,
    'weighted_feature_sampling': False,
    'weighted_voting': True,
    'max_features': 'sqrt',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

all_comparison_results = []

for dataset_name in dataset_names:
    print(f"\nProcessing dataset: {dataset_name}")
    X, y, label_encoder = load_preprocessed_data(dataset_name, DATA_DIR)

    if X is None or y is None:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    n_features = X_train.shape[1]
    num_classes = len(label_encoder.classes_) if label_encoder else len(np.unique(y_train))

    print(f"  Data: X_train {X_train.shape}, y_train {y_train.shape}, n_features {n_features}, num_classes {num_classes}")

    # Resolve max_features for MRF (sqrt(n_features))
    mrf_params = mrf_default_params.copy()

    models_to_compare = {
        "ModifiedRF": ModifiedRandomForest(**mrf_params),
        "ScikitRF": RandomForestClassifier(
            n_estimators=mrf_params['n_trees'],
            max_depth=mrf_params['max_depth'],
            criterion=mrf_params['criterion'],
            max_features=mrf_params['max_features'],
            min_samples_split=mrf_params['min_samples_split'],
            min_samples_leaf=mrf_params['min_samples_leaf'],
            random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=mrf_params['n_trees'],
            random_state=42
        )
    }

    for model_name, model in models_to_compare.items():
        print(f"  Training {model_name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time

        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception as e:
                print(f"    Could not get probabilities for {model_name}: {e}")
        else:
            print(f"    predict_proba not available for {model_name}.")

        eval_metrics = calculate_metrics(y_test, y_pred, y_proba, num_classes, label_encoder, model_name)
        eval_metrics['fit_time'] = fit_time
        eval_metrics['dataset'] = dataset_name
        eval_metrics['model'] = model_name

        all_comparison_results.append(eval_metrics)
        print(f"    {model_name} Metrics: {eval_metrics}")

# Save all comparison results
comparison_df = pd.DataFrame(all_comparison_results)
results_csv_path = os.path.join(BASE_RESULTS_DIR, 'dataset_comparison_results.csv')
comparison_df.to_csv(results_csv_path, index=False)
print(f"\nAll dataset comparison results saved to {results_csv_path}")

# Plot comparison metrics
if not comparison_df.empty:
    plot_comparison_metrics(comparison_df, PLOTS_DIR)
else:
    print("No results to plot for dataset comparison.")

print("\nDataset comparison phase complete.")
