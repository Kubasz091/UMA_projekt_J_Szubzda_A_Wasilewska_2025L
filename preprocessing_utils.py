import pandas as pd
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder

PREPROCESSED_DIR = './preprocessed_datasets'
Path(PREPROCESSED_DIR).mkdir(parents=True, exist_ok=True)

def adaptive_robust_binning(df, num_cols, y=None, min_adaptive_bins=10, max_adaptive_bins=20):
    df_result = df.copy()

    if y is not None:
        features_to_bin = df[num_cols].values
        mi_scores = mutual_info_classif(features_to_bin, y)

        if max(mi_scores) == min(mi_scores):
            bin_counts = {col: int((max_adaptive_bins + min_adaptive_bins) / 2) for col in num_cols}
        else:
            scaled_bins = min_adaptive_bins + ((mi_scores - min(mi_scores)) /
                           (max(mi_scores) - min(mi_scores)) *
                           (max_adaptive_bins - min_adaptive_bins))
            bin_counts = {col: max(2, int(scaled_bins[i])) for i, col in enumerate(num_cols)}
    else:
        bin_counts = {col: min_adaptive_bins for col in num_cols}

    for col in num_cols:
        if col not in df.columns:
            continue

        if df[col].nunique() <= 1:
            df_result[col] = 0
            continue

        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        if lower_bound == upper_bound:
            lower_bound = df[col].min()
            upper_bound = df[col].max()

        values_to_bin = df[col].clip(lower_bound, upper_bound)

        try:
            df_result[col] = pd.qcut(
                values_to_bin,
                q=bin_counts[col],
                labels=False,
                duplicates='drop'
            )

            median_bin = int(df_result[col].median())
            df_result[col] = df_result[col].fillna(median_bin)
        except:
            try:
                df_result[col] = pd.cut(
                    values_to_bin,
                    bins=bin_counts[col],
                    labels=False,
                    include_lowest=True
                )
                median_bin = int(df_result[col].median())
                df_result[col] = df_result[col].fillna(median_bin)
            except:
                df_result[col] = df[col]
    return df_result

def one_hot_encode(df, cat_features):
    df_result = df.copy()
    valid_cat_cols = [col for col in cat_features if col in df.columns]
    if not valid_cat_cols:
        return df_result

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[valid_cat_cols])
    feature_names = encoder.get_feature_names_out(valid_cat_cols)

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=feature_names,
        index=df.index
    )

    df_result = df_result.drop(columns=valid_cat_cols)
    df_result = pd.concat([df_result, encoded_df], axis=1)
    return df_result