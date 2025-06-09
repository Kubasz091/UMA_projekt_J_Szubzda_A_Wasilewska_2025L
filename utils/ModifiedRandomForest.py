import numpy as np
from utils.sampling import (uniform_distribution, normalize_weights,
    sample_with_replacement, weighted_random_selection, random_selection)
from utils.tree import DecisionTree
from utils.pruning import prune_tree
from utils.prediction import accuracy, weighted_majority_vote, majority_vote
from utils.split import calculate_information_gain

class ModifiedRandomForest:
    def __init__(self, n_trees=100, max_features='sqrt', sample_fraction=0.7,
                 max_depth=None, prune=False, criterion='gini',
                 weighted_voting=False, error_weight_increase=0.0,
                 weighted_feature_sampling=False, min_samples_split=2,
                 min_samples_leaf=1, random_state=None):

        self.n_trees = n_trees
        self.max_features = max_features
        self.sample_fraction = sample_fraction
        self.max_depth = max_depth
        self.prune = prune
        self.criterion = criterion
        self.weighted_voting = weighted_voting
        self.error_weight_increase = error_weight_increase
        self.weighted_feature_sampling = weighted_feature_sampling
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.forest = []
        self.oob_accuracies = []
        self.feature_importances_ = None
        self.class_to_idx_ = None
        self.idx_to_class_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def _get_max_features(self, n_features):
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                return int(np.log2(n_features) + 1)
            else:
                return n_features
        elif isinstance(self.max_features, float) and 0.0 < self.max_features <= 1.0:
            return int(self.max_features * n_features)
        else:
            return min(int(self.max_features), n_features)

    def setup_class_mapping(self, y):
        unique_classes = np.unique(y)
        self.n_classes_ = len(unique_classes)

        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class_ = {idx: cls for idx, cls in enumerate(unique_classes)}

        y_mapped = np.array([self.class_to_idx_[cls] for cls in y])

        print(f"Set up class mapping with {self.n_classes_} classes")
        return y_mapped

    def _build_single_tree(self, X, y, sample_weights, tree_idx):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)

        sample_size = int(self.sample_fraction * n_samples)
        indices = sample_with_replacement(
            np.arange(n_samples),
            size=sample_size,
            p=sample_weights
        )

        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[indices] = False
        X_train, y_train = X[indices], y[indices]
        X_oob, y_oob = X[oob_mask], y[oob_mask]

        if self.weighted_feature_sampling:
            ig = calculate_information_gain(X_train, y_train)
            feature_indices = weighted_random_selection(max_features, ig)
        else:
            feature_indices = random_selection(n_features, max_features)

        tree = DecisionTree(
            max_depth=self.max_depth,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=None # random state messes things up
        )
        tree.fit(X_train, y_train, feature_indices)

        if self.prune:
            if len(X_oob) == 0:
                tree = prune_tree(tree, X_train, y_train)
            else:
                tree = prune_tree(tree, X_oob, y_oob)

        oob_accuracy = 0.0
        if len(X_oob) > 0:
            oob_predictions = tree.predict(X_oob)
            oob_accuracy = accuracy(y_oob, oob_predictions)

        return tree, oob_accuracy, tree.feature_importances_

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        y = self.setup_class_mapping(y)

        n_samples, n_features = X.shape
        self.n_classes_ = len(np.unique(y))

        sample_weights = uniform_distribution(n_samples)
        self.forest = []
        self.oob_accuracies = []
        self.feature_importances_ = np.zeros(n_features)

        min_weight = sample_weights[0] * 1e-4

        for i in range(self.n_trees):
            tree, oob_acc, tree_importances = self._build_single_tree(X, y, sample_weights, i)

            self.forest.append(tree)
            self.oob_accuracies.append(oob_acc)
            self.feature_importances_ += tree_importances

            if self.error_weight_increase > 0:
                current_pred = self.predict(X, weighted_voting=self.weighted_voting)
                errors = current_pred != y

                for j in range(n_samples):
                    if errors[j]:
                        sample_weights[j] *= (1.0 + self.error_weight_increase)

                sample_weights = normalize_weights(sample_weights, min_weight=min_weight)

        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = normalize_weights(self.feature_importances_)

        return self

    def predict(self, X, weighted_voting=None):
        X = np.asarray(X)
        if weighted_voting is None:
            weighted_voting = self.weighted_voting

        n_samples = X.shape[0]
        predictions = np.zeros((len(self.forest), n_samples), dtype=np.int32)

        for i, tree in enumerate(self.forest):
            predictions[i] = tree.predict(X)

        if weighted_voting:
            weights = normalize_weights(np.array(self.oob_accuracies), min_weight=1e-10)
            indices = weighted_majority_vote(predictions, weights, self.n_classes_)
        else:
            indices = majority_vote(predictions)

        if self.idx_to_class_ is not None:
            return np.array([self.idx_to_class_[int(idx)] for idx in indices])
        else:
            return indices

    def predict_proba(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.n_classes_))

        for tree in self.forest:
            preds = tree.predict(X)
            for i, pred in enumerate(preds):
                probas[i, int(pred)] += 1

        probas /= np.sum(probas, axis=1, keepdims=True)
        return probas

    def print_mapping(self):
        if self.class_to_idx_ is None or self.idx_to_class_ is None:
            print("No class mapping available. The model may not be fitted yet.")
            return

        max_class_width = max(len(str(cls)) for cls in self.class_to_idx_.keys())
        max_idx_width = max(len(str(idx)) for idx in self.idx_to_class_.keys())
        max_width = max(max_class_width, max_idx_width)

        print("\n" + "="*50)
        print(f"CLASS MAPPING ({self.n_classes_} classes)")
        print("="*50)

        print("\nORIGINAL CLASS → INDEX:")
        print("-" * 30)
        for cls, idx in sorted(self.class_to_idx_.items(), key=lambda x: x[1]):
            print(f"{str(cls):<{max_width}} → {idx}")

        print("\nINDEX → ORIGINAL CLASS:")
        print("-" * 30)
        for idx, cls in sorted(self.idx_to_class_.items()):
            print(f"{idx:<{max_width}} → {cls}")

        print("\n" + "="*50 + "\n")

    def save_model(self, filename):
        import json

        forest_dict = []
        for tree in self.forest:
            forest_dict.append(tree.to_dict())

        class_to_idx_serialized = {}
        idx_to_class_serialized = {}

        if self.class_to_idx_ is not None:
            for cls, idx in self.class_to_idx_.items():
                if isinstance(cls, int):
                    key_type = "int"
                    cls_str = str(cls)
                elif isinstance(cls, float):
                    key_type = "float"
                    cls_str = str(cls)
                    if cls_str.endswith('.0'):
                        cls_str = str(int(cls))
                        key_type = "float_int"
                elif isinstance(cls, bool):
                    key_type = "bool"
                    cls_str = str(cls)
                else:
                    key_type = "str"
                    cls_str = str(cls)
                class_to_idx_serialized[cls_str] = {"type": key_type, "value": idx}

        if self.idx_to_class_ is not None:
            for idx, cls in self.idx_to_class_.items():
                if isinstance(cls, int):
                    val_type = "int"
                    val_str = str(cls)
                elif isinstance(cls, float):
                    val_type = "float"
                    val_str = str(cls)
                    if cls.is_integer():
                        val_type = "float_int"
                elif isinstance(cls, bool):
                    val_type = "bool"
                    val_str = str(cls)
                else:
                    val_type = "str"
                    val_str = str(cls)
                idx_to_class_serialized[str(idx)] = {"type": val_type, "value": val_str}

        model_dict = {
            'params': {
                'n_trees': self.n_trees,
                'max_features': self.max_features,
                'sample_fraction': self.sample_fraction,
                'max_depth': self.max_depth,
                'prune': self.prune,
                'criterion': self.criterion,
                'weighted_voting': self.weighted_voting,
                'error_weight_increase': self.error_weight_increase,
                'weighted_feature_sampling': self.weighted_feature_sampling,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'random_state': self.random_state
            },
            'forest': forest_dict,
            'oob_accuracies': self.oob_accuracies,
            'feature_importances_': self.feature_importances_.tolist(),
            'n_classes_': self.n_classes_,
            'class_mapping': {
                'class_to_idx': class_to_idx_serialized,
                'idx_to_class': idx_to_class_serialized
            }
        }

        with open(filename, 'w') as f:
            json.dump(model_dict, f)

        print(f"Model saved to {filename}")

    @classmethod
    def load_model(cls, filename):
        import json
        from utils.tree import DecisionTree

        with open(filename, 'r') as f:
            model_dict = json.load(f)

        model = cls(**model_dict['params'])

        model.forest = []
        for tree_dict in model_dict['forest']:
            model.forest.append(DecisionTree.from_dict(tree_dict))

        model.oob_accuracies = model_dict['oob_accuracies']
        model.feature_importances_ = np.array(model_dict['feature_importances_'])
        model.n_classes_ = model_dict['n_classes_']

        if 'class_mapping' in model_dict:
            model.class_to_idx_ = {}
            for cls_str, data in model_dict['class_mapping']['class_to_idx'].items():
                key_type = data['type']

                if key_type == 'int':
                    key = int(cls_str)
                elif key_type == 'float' or key_type == 'float_int':
                    key = float(cls_str)
                elif key_type == 'bool':
                    key = cls_str.lower() == 'true'
                else:
                    key = cls_str

                model.class_to_idx_[key] = data['value']

            model.idx_to_class_ = {}
            for idx_str, data in model_dict['class_mapping']['idx_to_class'].items():
                idx = int(idx_str)
                val_type = data['type']
                val_str = data['value']

                if val_type == 'int':
                    val = int(val_str)
                elif val_type == 'float':
                    val = float(val_str)
                elif val_type == 'float_int':
                    val = float(val_str)
                elif val_type == 'bool':
                    val = val_str.lower() == 'true'
                else:
                    val = val_str

                model.idx_to_class_[idx] = val

        print(f"Model loaded from {filename}")
        if model.class_to_idx_ is not None:
            print(f"Loaded class mapping with {model.n_classes_} classes")

        return model