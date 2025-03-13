import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def impute_missing_values(X, strategy='mean'):
    # Use scikit-learn's optimized imputer
    imputer = SimpleImputer(strategy=strategy)
    return imputer.fit_transform(X)

def delete_rows_with_missing_values(X):
    return X[~np.isnan(X).any(axis=1)]

def encode_categorical(X, categorical_columns, encoding='onehot'):
    X = X.copy()

    if encoding == 'onehot':
        # One-hot encoding (for multi-class features)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        if len(categorical_columns) > 0:
            encoded_features = encoder.fit_transform(X[:, categorical_columns])

            # Replace original categorical columns with encoded features
            numeric_columns = [i for i in range(X.shape[1])
                              if i not in categorical_columns]

            # Combine numeric and encoded features
            if numeric_columns:
                X_numeric = X[:, numeric_columns]
                X = np.hstack([X_numeric, encoded_features])
            else:
                X = encoded_features

    elif encoding == 'label':
        # Label encoding (for binary or ordinal features)
        for col in categorical_columns:
            encoder = LabelEncoder()
            X[:, col] = encoder.fit_transform(X[:, col])

    return X