import os
import pickle
import time
from pathlib import Path

# Import the existing dataset loading function
from data_analysis import load_datasets

def save_datasets(base_dir='./datasets_cache'):
    """
    Load datasets using the existing function and save them to disk.

    Args:
        base_dir: Directory to save the cached datasets
    """
    # Create cache directory if it doesn't exist
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading datasets and saving to {base_dir}...")
    start_time = time.time()

    # Load all datasets using existing function
    datasets = load_datasets()

    # Save each dataset to disk
    for name, (X, y, cat_features, class_mapping) in datasets.items():
        dataset_path = os.path.join(base_dir, f"{name}.pkl")

        # Create a dictionary with all components
        dataset_dict = {
            'X': X,
            'y': y,
            'cat_features': cat_features,
            'class_mapping': class_mapping
        }

        # Save to pickle file
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset_dict, f)

        print(f"Saved {name} dataset ({X.shape[0]} samples, {X.shape[1]} features)")

    elapsed_time = time.time() - start_time
    print(f"All datasets saved successfully in {elapsed_time:.2f} seconds.")

def load_cached_datasets(names=None, base_dir='./datasets_cache'):
    """
    Load datasets from cache if available.

    Args:
        names: List of dataset names to load (None for all)
        base_dir: Directory where cached datasets are stored

    Returns:
        Dictionary of datasets in the same format as load_datasets()
    """
    cached_datasets = {}

    # Get all available datasets if names not specified
    if names is None:
        pickle_files = [f for f in os.listdir(base_dir) if f.endswith('.pkl')]
        names = [os.path.splitext(f)[0] for f in pickle_files]

    print(f"Loading datasets from cache ({', '.join(names)})...")
    start_time = time.time()

    for name in names:
        dataset_path = os.path.join(base_dir, f"{name}.pkl")

        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                dataset_dict = pickle.load(f)

            # Extract components
            X = dataset_dict['X']
            y = dataset_dict['y']
            cat_features = dataset_dict['cat_features']
            class_mapping = dataset_dict['class_mapping']

            cached_datasets[name] = (X, y, cat_features, class_mapping)
            print(f"Loaded {name} dataset from cache ({X.shape[0]} samples, {X.shape[1]} features)")
        else:
            print(f"Warning: {name} dataset not found in cache")

    elapsed_time = time.time() - start_time
    print(f"Loaded {len(cached_datasets)} datasets in {elapsed_time:.2f} seconds.")

    return cached_datasets

def get_dataset(name, base_dir='./datasets_cache'):
    """
    Load a single dataset from cache or download if necessary.

    Args:
        name: Name of the dataset to load
        base_dir: Cache directory

    Returns:
        Tuple of (X, y, cat_features, class_mapping)
    """
    dataset_path = os.path.join(base_dir, f"{name}.pkl")

    # Check if dataset exists in cache
    if os.path.exists(dataset_path):
        print(f"Loading {name} dataset from cache...")
        with open(dataset_path, 'rb') as f:
            dataset_dict = pickle.load(f)

        X = dataset_dict['X']
        y = dataset_dict['y']
        cat_features = dataset_dict['cat_features']
        class_mapping = dataset_dict['class_mapping']
        print(f"Loaded {name} dataset ({X.shape[0]} samples, {X.shape[1]} features)")

    else:
        print(f"{name} not found in cache, downloading...")
        # Create directory if it doesn't exist
        Path(base_dir).mkdir(parents=True, exist_ok=True)

        # Load all datasets (inefficient but simpler than modifying original function)
        datasets = load_datasets()

        if name in datasets:
            X, y, cat_features, class_mapping = datasets[name]

            # Save for future use
            dataset_dict = {
                'X': X,
                'y': y,
                'cat_features': cat_features,
                'class_mapping': class_mapping
            }

            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset_dict, f)

            print(f"Downloaded and cached {name} dataset")
        else:
            raise ValueError(f"Dataset '{name}' not found")

    return X, y, cat_features, class_mapping

if __name__ == "__main__":
    # When run as a script, cache all datasets
    save_datasets()