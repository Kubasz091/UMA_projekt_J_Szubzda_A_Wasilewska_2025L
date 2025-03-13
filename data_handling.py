import os
import pickle
import time
from pathlib import Path
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml

SAVE_DIR = './datasets_cache'
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

def save_dataset(name, X, y, cat_features=None, class_mapping=None, base_dir=SAVE_DIR, suffix=''):
    if suffix:
        name = f"{name}_{suffix}"
    dataset_path = os.path.join(base_dir, f"{name}.pkl")

    dataset_dict = {
        'X': X,
        'y': y,
        'cat_features': cat_features,
        'class_mapping': class_mapping
    }

    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset_dict, f)

    print(f"Saved {name} dataset ({X.shape[0]} samples, {X.shape[1]} features)")
    return dataset_path

def load_dataset(name, base_dir=SAVE_DIR):
    dataset_path = os.path.join(base_dir, f"{name}.pkl")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{name}' not found in cache at {dataset_path}")

    with open(dataset_path, 'rb') as f:
        dataset_dict = pickle.load(f)

    X = dataset_dict['X']
    y = dataset_dict['y']
    cat_features = dataset_dict['cat_features']
    class_mapping = dataset_dict['class_mapping']

    print(f"Loaded {name} dataset from cache ({X.shape[0]} samples, {X.shape[1]} features)")
    return X, y, cat_features, class_mapping


##
def download_iris(base_dir=SAVE_DIR):
    """Download the Iris dataset directly from UCI - no transformations"""
    print("Downloading Iris dataset...")
    iris = fetch_ucirepo(id=53)

    X = iris.data.features
    y = iris.data.targets.iloc[:, 0]

    save_dataset('Iris', X, y, None, None, base_dir)
    return X, y, None, None

def download_wine(base_dir=SAVE_DIR):
    print("Downloading Wine dataset...")
    wine = fetch_ucirepo(id=186)

    X = wine.data.features
    y = wine.data.targets.iloc[:, 0]

    save_dataset('Wine', X, y, None, None, base_dir)
    return X, y, None, None

def download_adult_census(base_dir=SAVE_DIR):
    print("Downloading Adult Census dataset...")
    adult = fetch_ucirepo(id=2)

    X = adult.data.features
    y = adult.data.targets.iloc[:, 0]
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    save_dataset('Adult_Census', X, y, cat_cols, None, base_dir)
    return X, y, cat_cols, None

def download_bank_marketing(base_dir=SAVE_DIR):
    print("Downloading Bank Marketing dataset...")
    bank = fetch_ucirepo(id=222)

    X = bank.data.features
    y = bank.data.targets.iloc[:, 0]

    bank_cats = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

    save_dataset('Bank_Marketing', X, y, bank_cats, None, base_dir)
    return X, y, bank_cats, None

def download_mnist(base_dir=SAVE_DIR):
    print("Downloading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')

    X = mnist.data
    y = mnist.target

    save_dataset('MNIST', X, y, None, None, base_dir)
    return X, y, None, None
##


if __name__ == "__main__":
    print(f"Downloading and saving datasets to {SAVE_DIR}...")
    start_time = time.time()

    download_iris(SAVE_DIR)
    download_wine(SAVE_DIR)
    download_adult_census(SAVE_DIR)
    download_bank_marketing(SAVE_DIR)
    download_mnist(SAVE_DIR)

    elapsed_time = time.time() - start_time
    print(f"All datasets saved successfully in {elapsed_time:.2f} seconds.")