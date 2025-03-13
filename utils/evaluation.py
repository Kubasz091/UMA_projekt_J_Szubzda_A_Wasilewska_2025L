import numpy as np
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import KFold

def confusion_matrix(y_true, y_pred, normalize=False):
    cm = sk_confusion_matrix(y_true, y_pred)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums!=0)

    return cm

def calculate_metrics(y_true, y_pred, y_proba=None):
    metrics = {}
    metrics['accuracy'] = np.mean(y_true == y_pred)

    classes = np.unique(y_true)
    n_classes = len(classes)

    # use weighted avg for all datasets
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    if n_classes <= 2:
        try:
            pos_precision = precision_score(y_true, y_pred, zero_division=0)
            pos_recall = recall_score(y_true, y_pred, zero_division=0)
            pos_f1 = f1_score(y_true, y_pred, zero_division=0)

            metrics['pos_class_precision'] = pos_precision
            metrics['pos_class_recall'] = pos_recall
            metrics['pos_class_f1'] = pos_f1
        except:
            pass

        if y_proba is not None:
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                pos_proba = y_proba[:, 1]
            else:
                pos_proba = y_proba

            try:
                metrics['roc_auc'] = roc_auc_score(y_true, pos_proba)
            except:
                metrics['roc_auc'] = float('nan')
    return metrics

def roc_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return float('nan')

def cross_validate(model_class, X, y, n_folds=5, random_state=None, **model_params):
    X = np.asarray(X)
    y = np.asarray(y)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    all_metrics = []
    all_predictions = []
    all_true_values = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        all_predictions.append(y_pred)
        all_true_values.append(y_test)

        fold_metrics = calculate_metrics(y_test, y_pred)
        fold_metrics['fold'] = fold_idx
        all_metrics.append(fold_metrics)

    metrics_keys = all_metrics[0].keys()
    aggregated_metrics = {}

    for key in metrics_keys:
        if key != 'fold':
            values = [m[key] for m in all_metrics]
            aggregated_metrics[key + '_mean'] = np.mean(values)
            aggregated_metrics[key + '_std'] = np.std(values)

    aggregated_metrics['fold_metrics'] = all_metrics

    all_y_pred = np.concatenate(all_predictions)
    all_y_true = np.concatenate(all_true_values)

    overall_metrics = calculate_metrics(all_y_true, all_y_pred)
    aggregated_metrics['overall'] = overall_metrics

    return aggregated_metrics