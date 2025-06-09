import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.ModifiedRandomForest import ModifiedRandomForest

BASE_RESULTS_DIR = os.path.join(project_root, 'experiments', 'results', 'parameter_search')
PLOTS_DIR = os.path.join(BASE_RESULTS_DIR, 'plots')
DATA_DIR = os.path.join(project_root, 'preprocessed_datasets')
DATASET_NAME = 'wine_quality_original'

os.makedirs(PLOTS_DIR, exist_ok=True)


def calculate_metrics(y_true, y_pred, y_proba, num_classes, le):
    if y_proba is None and num_classes > 2:
        roc_auc = np.nan
    elif y_proba is None and num_classes == 2:
         roc_auc = np.nan
    elif num_classes == 2:
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc
    }

def plot_metrics_vs_param(param_name, param_values, metrics_df, save_dir):
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    axs = axs.ravel()

    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'fit_time']

    str_param_values = [str(pv) for pv in param_values]

    for i, metric in enumerate(metric_keys):
        if metric in metrics_df.columns:
            axs[i].plot(str_param_values, metrics_df[metric], marker='o')
            axs[i].set_title(f'{metric.replace("_", " ").title()} vs {param_name}')
            axs[i].set_xlabel(param_name)
            axs[i].set_ylabel(metric.replace("_", " ").title())
            axs[i].grid(True)
            if len(str_param_values) > 5:
                 axs[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'{param_name}_performance.png')
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_path

print(f"Loading data: {DATASET_NAME}")

pkl_filename = "Wine_original.pkl"

pkl_path = os.path.join(DATA_DIR, pkl_filename)

if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"Processed data file {pkl_filename} not found in {DATA_DIR}")

data_dict = pd.read_pickle(pkl_path)

X = data_dict['X']
y = data_dict['y']

if isinstance(X, pd.DataFrame):
    X = X.values

if isinstance(y, pd.Series):
    y = y.values
if y.ndim > 1 and y.shape[1] == 1:
    y = y.ravel()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
n_features = X_train.shape[1]
num_classes = len(np.unique(y_train))
print(f"Data loaded: X_train shape {X_train.shape}, y_train shape {y_train.shape}, n_features {n_features}, num_classes {num_classes}")


default_params = {
    'n_trees': 50,
    'sample_fraction': 0.7,
    'max_depth': 10,
    'prune': False,
    'criterion': 'gini',
    'error_weight_increase': 0.3,
    'weighted_feature_sampling': False,
    'weighted_voting': False,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

param_grid = {
    'max_features_config': ['sqrt', 0.3, 0.5, 1.0],
    'sample_fraction': [0.5, 0.7, 0.9],
    'n_trees': [10, 50, 100, 200],
    'max_depth': [5, 10, None],
    'prune': [True, False],
    'criterion': ['gini', 'entropy'],
    'weighted_voting': [True, False],
    'error_weight_increase': [0.0, 0.1, 0.3, 0.5, 1.0],
    'weighted_feature_sampling': [True, False]
}

all_results = []

def resolve_max_features(config_val, num_feats):
    if config_val == 'sqrt':
        return int(np.sqrt(num_feats)) if num_feats > 0 else 1
    elif config_val == 'log2':
        return int(np.log2(num_feats)) if num_feats > 1 else 1
    elif isinstance(config_val, float) and 0 < config_val <= 1.0:
        return int(config_val * num_feats) if num_feats > 0 else 1
    elif isinstance(config_val, int):
        return config_val
    return num_feats

for param_name, param_values in param_grid.items():
    print(f"\nTuning parameter: {param_name}")
    current_param_results = []

    for value in param_values:
        params = default_params.copy()

        if param_name == 'max_features_config':
            params['max_features'] = resolve_max_features(value, n_features)
            display_value = str(value)
        else:
            params[param_name] = value
            display_value = str(value)
            if 'max_features' not in params:
                 params['max_features'] = resolve_max_features('sqrt', n_features)


        print(f"  Testing {param_name} = {display_value} (resolved max_features: {params.get('max_features')})")

        model = ModifiedRandomForest(**params)

        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time

        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception as e:
                print(f"    Could not get probabilities: {e}")
        else:
            print("    predict_proba not available for this model.")

        eval_metrics = calculate_metrics(y_test, y_pred, y_proba, num_classes, label_encoder)
        eval_metrics['fit_time'] = fit_time
        eval_metrics['param_tuned'] = param_name
        eval_metrics['param_value'] = display_value
        eval_metrics['resolved_max_features'] = params.get('max_features')

        current_param_results.append(eval_metrics)
        all_results.append(eval_metrics)

        print(f"    Metrics: {eval_metrics}")

    param_df = pd.DataFrame(current_param_results)
    actual_param_name_for_plot = 'max_features' if param_name == 'max_features_config' else param_name

    plot_path = plot_metrics_vs_param(actual_param_name_for_plot, param_df['param_value'].tolist(), param_df, PLOTS_DIR)
    print(f"    Plot saved to: {plot_path}")


all_results_df = pd.DataFrame(all_results)
results_csv_path = os.path.join(BASE_RESULTS_DIR, 'parameter_search_results.csv')
all_results_df.to_csv(results_csv_path, index=False)
print(f"\nAll parameter search results saved to {results_csv_path}")

print("\nParameter search phase complete.")
