import numpy as np
from utils.sampling import (
    uniform_distribution, normalize_weights,
    sample_with_replacement, weighted_random_selection, random_selection
)
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
            return min(self.max_features, n_features)

    def _build_single_tree(self, X, y, sample_weights, tree_idx):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)

        # Bootstrap sample with weight
        sample_size = int(self.sample_fraction * n_samples)
        indices = sample_with_replacement(
            np.arange(n_samples),
            size=sample_size,
            p=sample_weights
        )

        # Split training and OOB samples
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[indices] = False
        X_train, y_train = X[indices], y[indices]
        X_oob, y_oob = X[oob_mask], y[oob_mask]

        # Select features
        if self.weighted_feature_sampling:
            ig = calculate_information_gain(X_train, y_train)
            feature_indices = weighted_random_selection(max_features, ig)
        else:
            feature_indices = random_selection(n_features, max_features)

        # Create and fit tree
        tree = DecisionTree(
            max_depth=self.max_depth,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=None # if random state is given here it will make the same tree over and over again
        )
        tree.fit(X_train, y_train, feature_indices)

        # Prune if needed
        if self.prune:
            if len(X_oob) == 0:
                tree = prune_tree(tree, X_train, y_train)
            else:
                tree = prune_tree(tree, X_oob, y_oob)

        # Calculate OOB accuracy for weighted voting
        oob_accuracy = 0.0
        if len(X_oob) > 0:
            oob_predictions = tree.predict(X_oob)
            oob_accuracy = accuracy(y_oob, oob_predictions)

        return tree, oob_accuracy, tree.feature_importances_

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.n_classes_ = len(np.unique(y))

        # Initialize sample weights and storage
        sample_weights = uniform_distribution(n_samples)
        self.forest = []
        self.oob_accuracies = []
        self.feature_importances_ = np.zeros(n_features)

        minimal_sample_weight = sample_weights[0] * 1e-4

        for i in range(self.n_trees):
            # Build single tree
            tree, oob_acc, tree_importances = self._build_single_tree(X, y, sample_weights, i)

            # Store tree and OOB accuracy
            self.forest.append(tree)
            self.oob_accuracies.append(oob_acc)
            self.feature_importances_ += tree_importances

            # Update sample weights based on current forest predictions
            if self.error_weight_increase > 0:
                # Get predictions from current forest
                current_pred = self.predict(X, weighted_voting=self.weighted_voting)
                errors = current_pred != y

                # Update weights for misclassified samples
                for j in range(n_samples):
                    if errors[j]:
                        sample_weights[j] *= (1.0 + self.error_weight_increase)

                # Normalize weights
                sample_weights = normalize_weights(sample_weights, min_weight=minimal_sample_weight)

        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = normalize_weights(self.feature_importances_)

        return self

    def predict(self, X, weighted_voting=None):
        X = np.asarray(X)

        if weighted_voting is None:
            weighted_voting = self.weighted_voting

        n_samples = X.shape[0]
        predictions = np.zeros((len(self.forest), n_samples), dtype=np.int32)

        # Get predictions from each tree
        for i, tree in enumerate(self.forest):
            predictions[i] = tree.predict(X)

        # Combine predictions
        if weighted_voting:
            weights = normalize_weights(np.array(self.oob_accuracies), min_weight=1e-10)
            return weighted_majority_vote(predictions, weights, self.n_classes_)
        else:
            return majority_vote(predictions)

    def predict_proba(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]

        # Initialize probabilities array
        probas = np.zeros((n_samples, self.n_classes_))

        # Get raw votes
        for tree in self.forest:
            preds = tree.predict(X)
            for i, pred in enumerate(preds):
                probas[i, int(pred)] += 1

        # Normalize to get probabilities
        probas /= np.sum(probas, axis=1, keepdims=True)
        return probas

    def save_model(self, filename):
        """Save model to file"""
        import json

        # Convert trees to serializable format
        forest_dict = []
        for tree in self.forest:
            forest_dict.append(tree.to_dict())

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
            'n_classes_': self.n_classes_
        }

        with open(filename, 'w') as f:
            json.dump(model_dict, f)

    @classmethod
    def load_model(cls, filename):
        """Load model from file"""
        import json
        from utils.tree import DecisionTree

        with open(filename, 'r') as f:
            model_dict = json.load(f)

        # Create model with parameters
        model = cls(**model_dict['params'])

        # Load trees
        model.forest = []
        for tree_dict in model_dict['forest']:
            model.forest.append(DecisionTree.from_dict(tree_dict))

        # Load other attributes
        model.oob_accuracies = model_dict['oob_accuracies']
        model.feature_importances_ = np.array(model_dict['feature_importances_'])
        model.n_classes_ = model_dict['n_classes_']

        return model