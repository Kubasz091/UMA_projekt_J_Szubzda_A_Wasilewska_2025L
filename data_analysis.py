import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
import matplotlib.gridspec as gridspec
import os
import warnings

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)

def information_gain(X, y, feature_idx):
    # Get entropy before split
    y_counts = np.bincount(y)
    y_probs = y_counts / len(y)
    total_entropy = entropy(y_probs, base=2)

    # Split on feature values
    feature_values = X[:, feature_idx]
    unique_vals = np.unique(feature_values)

    # Calculate weighted entropy after split
    w_entropy = 0
    for val in unique_vals:
        indices = feature_values == val
        subset = y[indices]
        if len(subset) == 0:
            continue

        weight = len(subset) / len(y)
        counts = np.bincount(subset, minlength=len(y_counts))
        probs = counts / len(subset)
        w_entropy += weight * entropy(probs, base=2)

    return total_entropy - w_entropy

def analyze_dataset(X, y, name, categorical_features=None, class_mapping=None):
    print(f"Analyzing dataset: {name}")

    # Convert inputs to pandas if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    df = X.copy()
    df['target'] = y

    is_mnist = name.lower() == 'mnist'

    # Get basic stats
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    class_counts = y.value_counts()
    imbalance = class_counts.max() / class_counts.min()
    missing_pct = X.isnull().sum().sum() / (X.shape[0] * X.shape[1]) * 100

    summary = pd.DataFrame({
        'Metric': ['Samples', 'Features', 'Classes', 'Imbalance Ratio', 'Missing Values (%)'],
        'Value': [n_samples, n_features, n_classes, round(imbalance, 2), round(missing_pct, 2)]
    })

    if is_mnist:
        # MNIST gets special treatment due to high dimensionality
        plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(2, 2)

        # Dataset summary table
        ax0 = plt.subplot(gs[0, 0])
        ax0.axis('tight')
        ax0.axis('off')
        table = ax0.table(cellText=summary.values,
                        colLabels=summary.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.4, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax0.set_title(f"Dataset Summary: {name}", fontsize=16)

        # Class distribution
        ax1 = plt.subplot(gs[0, 1])
        class_counts = y.value_counts().sort_index()
        class_counts.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Class Distribution', fontsize=14)
        ax1.set_ylabel('Count')
        ax1.set_xlabel('Class')

        for i, v in enumerate(class_counts):
            ax1.text(i, v + 0.1, str(v), ha='center', fontsize=10)

        # Correlation matrix for pixels
        ax2 = plt.subplot(gs[1, :])
        pixel_range = list(range(0, 200, 10))
        corr_sample = X.iloc[:, :200].corr()
        sns.heatmap(corr_sample, cmap='coolwarm', linewidths=0.1, ax=ax2,
                cbar_kws={"shrink": .8},
                xticklabels=[f"p{i}" if i in pixel_range else "" for i in range(200)],
                yticklabels=[f"p{i}" if i in pixel_range else "" for i in range(200)])

        ax2.set_title('Correlation Matrix (First 200 Pixels)', fontsize=14)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

    else:
        # For standard datasets with more plots
        plt.figure(figsize=(18, 20))
        gs = gridspec.GridSpec(4, 2)

        # Dataset summary
        ax0 = plt.subplot(gs[0, 0])
        ax0.axis('tight')
        ax0.axis('off')
        table = ax0.table(cellText=summary.values,
                        colLabels=summary.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.4, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax0.set_title(f"Dataset Summary: {name}", fontsize=16)

        # Class distribution - Update to use class mapping for x-tick labels
        ax1 = plt.subplot(gs[0, 1])
        class_counts = y.value_counts().sort_index()
        class_counts.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Class Distribution', fontsize=14)
        ax1.set_ylabel('Count')
        ax1.set_xlabel('Class')

        # If we have a class mapping, use it for the x-tick labels
        if class_mapping:
            plt.xticks(range(len(class_counts)),
                    [class_mapping.get(i, i) for i in sorted(class_counts.index)],
                    rotation=45, ha='right')

        for i, v in enumerate(class_counts):
            ax1.text(i, v + 0.1, str(v), ha='center', fontsize=10)

        # Missing values analysis
        ax2 = plt.subplot(gs[1, 0])
        missing = X.isnull().sum() / len(X) * 100
        missing = missing[missing > 0]
        if not missing.empty:
            missing = missing.sort_values(ascending=False)
            sns.heatmap(X[missing.index].isnull().sample(min(1000, len(X))),
                       cmap='viridis', ax=ax2)
            ax2.set_title('Missing Values Heatmap (Sample)', fontsize=14)
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax2.set_title('Missing Values Analysis', fontsize=14)
            ax2.axis('off')

        # Correlation matrix
        ax3 = plt.subplot(gs[1, 1])
        X_corr = X.copy()

        # Handle categorical features for correlation
        if categorical_features:
            for cat in categorical_features:
                if cat in X_corr.columns:
                    try:
                        le = LabelEncoder()
                        X_corr[cat] = le.fit_transform(X_corr[cat].astype(str))
                    except:
                        X_corr = X_corr.drop(cat, axis=1)

        if X_corr.shape[1] > 1:
            try:
                # Get numeric columns only
                X_corr = X_corr.select_dtypes(include=[np.number])
                if X_corr.shape[1] > 1:
                    corr = X_corr.corr()
                    mask = np.zeros_like(corr)
                    mask[np.triu_indices_from(mask)] = True
                    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True,
                               fmt=".2f", linewidths=0.5, ax=ax3,
                               cbar_kws={"shrink": .8})
                    ax3.set_title('Correlation Matrix', fontsize=14)
                else:
                    ax3.text(0.5, 0.5, 'Not enough numeric features for correlation',
                            ha='center', va='center', fontsize=14)
                    ax3.axis('off')
            except:
                ax3.axis('off')
        else:
            ax3.axis('off')

        # Feature distributions
        ax4 = plt.subplot(gs[2, 0])

        # Get numeric columns
        num_cols = X.select_dtypes(include=np.number).columns

        # Fix for Adult Census dataset
        if name == 'Adult_Census' and len(num_cols) == 0:
            numeric_candidates = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                                'capital-loss', 'hours-per-week']
            for col in numeric_candidates:
                if col in X.columns:
                    X[col] = pd.to_numeric(X[col].astype(str).str.strip(), errors='coerce')
            num_cols = X.select_dtypes(include=np.number).columns

        if len(num_cols) > 0:
            # Set up the grid for boxplots
            ax4.axis('off')
            plot_cols = num_cols[:min(6, len(num_cols))]
            n_cols = min(3, len(plot_cols))
            n_rows = (len(plot_cols) + n_cols - 1) // n_cols

            pos = ax4.get_position()
            fig = plt.gcf()
            fig.text(pos.x0 + pos.width/2, pos.y0 + pos.height + 0.02,
                    'Feature Distributions', ha='center', fontsize=14)

            # Create individual boxplots
            for i, feat in enumerate(plot_cols):
                row = i // n_cols
                col = i % n_cols

                # Position the subplot
                ax_pos = [
                    pos.x0 + col * (pos.width / n_cols),
                    pos.y0 + (n_rows - 1 - row) * (pos.height / n_rows),
                    pos.width / n_cols * 0.85,
                    pos.height / n_rows * 0.85
                ]
                feat_ax = fig.add_axes(ax_pos)

                # Plot the boxplot
                data = X[feat].dropna()
                if len(data) > 0:
                    color = plt.cm.tab10.colors[i % len(plt.cm.tab10.colors)]
                    bp = feat_ax.boxplot(data, vert=True, patch_artist=True)

                    for patch in bp['boxes']:
                        patch.set_facecolor(color)
                        patch.set_edgecolor('black')

                    feat_ax.set_title(feat, fontsize=10)
                    feat_ax.tick_params(axis='y', labelsize=8)

                    # Clean up appearance
                    feat_ax.spines['top'].set_visible(False)
                    feat_ax.spines['right'].set_visible(False)
                    feat_ax.spines['bottom'].set_visible(False)
                    feat_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            ax4.text(0.5, 0.5, 'No numeric features available',
                    ha='center', va='center', fontsize=14)
            ax4.axis('off')

        # Mutual Information with target
        ax5 = plt.subplot(gs[2, 1])
        try:
            X_proc = X.copy()

            # Process categorical features
            if categorical_features:
                for cat in categorical_features:
                    if cat in X_proc.columns:
                        try:
                            le = LabelEncoder()
                            X_proc[cat] = le.fit_transform(X_proc[cat].astype(str))
                        except:
                            X_proc = X_proc.drop(cat, axis=1)

            X_proc = X_proc.select_dtypes(include=[np.number])
            X_proc = X_proc.fillna(X_proc.median())

            if X_proc.shape[1] > 0:
                X_np = X_proc.values
                y_np = y.values.astype(int)

                # For big datasets, sample
                if len(X_np) > 10000:
                    idx = np.random.choice(len(X_np), 10000, replace=False)
                    X_sample = X_np[idx]
                    y_sample = y_np[idx]
                    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
                else:
                    mi_scores = mutual_info_classif(X_np, y_np, random_state=42)

                mi = pd.Series(mi_scores, index=X_proc.columns).sort_values(ascending=False)
                mi = mi[:min(15, len(mi))]

                mi.plot(kind='bar', ax=ax5)
                ax5.set_title('Mutual Information with Target', fontsize=14)
                ax5.set_ylabel('MI Score')
                plt.xticks(rotation=45, ha='right')
            else:
                ax5.axis('off')
        except Exception as e:
            print(f"MI calculation error: {e}")
            ax5.axis('off')

        # Information Gain
        ax6 = plt.subplot(gs[3, 0])
        try:
            if 'X_proc' in locals() and X_proc.shape[1] > 0:
                X_np = X_proc.values
                y_np = y.values.astype(int)

                # Sample for large datasets
                if len(X_np) > 10000:
                    idx = np.random.choice(len(X_np), 10000, replace=False)
                    X_sample = X_np[idx]
                    y_sample = y_np[idx]

                    ig_scores = [information_gain(X_sample, y_sample, i)
                               for i in range(X_sample.shape[1])]
                else:
                    ig_scores = [information_gain(X_np, y_np, i)
                               for i in range(X_np.shape[1])]

                ig = pd.Series(ig_scores, index=X_proc.columns).sort_values(ascending=False)
                ig = ig[:min(15, len(ig))]
                ig.plot(kind='bar', ax=ax6)
                ax6.set_title('Information Gain for Features', fontsize=14)
                ax6.set_ylabel('Information Gain')
                plt.xticks(rotation=45, ha='right')
            else:
                ax6.axis('off')
        except:
            ax6.axis('off')

        # Feature Variability
        ax7 = plt.subplot(gs[3, 1])
        try:
            num_cols = X.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                # Coefficient of variation
                cv = X[num_cols].std() / X[num_cols].mean().replace(0, np.nan)
                cv = cv.dropna().sort_values(ascending=False)
                cv = cv[:min(15, len(cv))]

                cv.plot(kind='bar', ax=ax7)
                ax7.set_title('Feature Variability (CV)', fontsize=14)
                ax7.set_ylabel('CV (std/mean)')
                plt.xticks(rotation=45, ha='right')
            else:
                ax7.axis('off')
        except:
            ax7.axis('off')

        # Create separate plot for categorical features
        if categorical_features:
            try:
                n_cats = min(5, len(categorical_features))
                plt.figure(figsize=(20, 5*n_cats))

                for i, cat in enumerate(categorical_features[:n_cats]):
                    if cat in X.columns:
                        plt.subplot(n_cats, 1, i+1)

                        counts = X[cat].value_counts()
                        if len(counts) > 30:
                            # Too many categories, group smaller ones
                            top_cats = counts.nlargest(29).index
                            grouped_counts = pd.Series({
                                **{c: counts[c] for c in top_cats},
                                'Other': counts.drop(top_cats).sum()
                            })
                            grouped_counts.plot(kind='bar')
                        else:
                            counts.plot(kind='bar')

                        plt.title(f'Distribution of {cat}')
                        plt.ylabel('Count')
                        plt.xticks(rotation=45, ha='right')

                plt.tight_layout()
                plt.savefig(f'figures/{name}_categorical_distributions.png', dpi=300, bbox_inches='tight')
                plt.close()
            except:
                print(f"Failed to plot categorical features for {name}")

    # Save the main figure
    plt.tight_layout()
    plt.savefig(f'figures/{name}_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create pairplot for small datasets
    if not is_mnist and n_features <= 10 and n_samples <= 10000:
        try:
            plt.figure(figsize=(12, 10))
            plot_df = X.copy()
            plot_df['target'] = y

            num_df = plot_df.select_dtypes(include=np.number)

            if len(num_df.columns) > 2:
                sns.pairplot(num_df, hue='target', corner=True, diag_kind='kde')
                plt.suptitle(f'{name} - Feature Relationships', y=1.02, fontsize=16)
                plt.tight_layout()
                plt.savefig(f'figures/{name}_pairplot.png', dpi=300, bbox_inches='tight')
            plt.close()
        except:
            print(f"Couldn't create pairplot for {name}")

    return {
        'name': name,
        'samples': n_samples,
        'features': n_features,
        'classes': n_classes,
        'imbalance_ratio': round(imbalance, 2),
        'missing_values_pct': round(missing_pct, 2)
    }

def load_datasets(base_dir='.'):
    datasets = {}
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Iris dataset - ID: 53
    print("Loading Iris dataset...")
    iris = fetch_ucirepo(id=53)
    X_iris = iris.data.features
    y_iris = iris.data.targets.iloc[:, 0]

    # Handle string class labels by encoding them
    class_mapping = None
    if y_iris.dtype == object:  # Check if y contains strings
        print("Converting Iris class labels to numeric...")
        original_classes = y_iris.unique()
        le = LabelEncoder()
        y_iris_encoded = le.fit_transform(y_iris)
        # Create mapping from numeric to original class names
        class_mapping = dict(zip(range(len(original_classes)), original_classes))
        y_iris = pd.Series(y_iris_encoded)
        print(f"Mapped classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

    datasets['Iris'] = (X_iris, y_iris, None, class_mapping)

    # Wine dataset
    print("Loading Wine dataset...")
    wine = fetch_ucirepo(id=186)
    X_wine = wine.data.features
    y_wine = wine.data.targets.iloc[:, 0]
    datasets['Wine'] = (X_wine, y_wine, None, None)

    # Adult Census Income
    print("Loading Adult Census dataset...")
    adult = fetch_ucirepo(id=2)
    X_adult = adult.data.features
    y_adult = adult.data.targets.iloc[:, 0]

    # Simple binary mapping for clearer labels
    income_mapping = {0: '<=50K', 1: '>50K'}
    y_adult = (y_adult == '>50K').astype(int)

    cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']

    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain',
               'capital-loss', 'hours-per-week']

    for col in num_cols:
        if col in X_adult.columns:
            X_adult[col] = pd.to_numeric(X_adult[col].astype(str).str.strip(), errors='coerce')

    datasets['Adult_Census'] = (X_adult, y_adult, cat_cols, income_mapping)

    # Bank Marketing
    print("Loading Bank Marketing dataset...")
    bank = fetch_ucirepo(id=222)
    X_bank = bank.data.features
    y_bank = (bank.data.targets.iloc[:, 0] == 'yes').astype(int)

    bank_mapping = {0: 'no', 1: 'yes'}
    bank_cats = ['job', 'marital', 'education', 'default', 'housing',
                'loan', 'contact', 'month', 'day_of_week', 'poutcome']

    datasets['Bank_Marketing'] = (X_bank, y_bank, bank_cats, bank_mapping)

    # MNIST dataset
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')
    X_mnist = mnist.data
    y_mnist = mnist.target

    # Convert target to numeric if it's not already
    if y_mnist.dtype == object:
        y_mnist = y_mnist.astype(int)

    datasets['MNIST'] = (X_mnist, y_mnist, None, None)

    return datasets

if __name__ == "__main__":
    datasets = load_datasets()

    stats = []
    for name, dataset_info in datasets.items():
        if len(dataset_info) == 4:  # Check if we have class mapping
            X, y, cat_features, class_mapping = dataset_info
            stats.append(analyze_dataset(X, y, name, cat_features, class_mapping))
        else:
            X, y, cat_features = dataset_info
            stats.append(analyze_dataset(X, y, name, cat_features))

    # Create summary table with proper formatting
    stats_df = pd.DataFrame(stats)

    # Format numbers: integers as int, floats with 2 decimal places but no trailing zeros
    for col in ['samples', 'features', 'classes']:
        stats_df[col] = stats_df[col].astype(int)

    # Custom format function to remove trailing zeros
    def format_float(x):
        if x == int(x):
            return int(x)
        else:
            # Format as string with 2 decimal places, then strip trailing zeros
            s = f"{x:.2f}".rstrip('0').rstrip('.')
            return s

    for col in ['imbalance_ratio', 'missing_values_pct']:
        stats_df[col] = stats_df[col].apply(format_float)

    # Clean column names for better LaTeX formatting
    clean_cols = {
        'name': 'Dataset',
        'samples': 'Samples',
        'features': 'Features',
        'classes': 'Classes',
        'imbalance_ratio': 'Imbalance Ratio',
        'missing_values_pct': 'Missing Values (%)'
    }

    stats_df = stats_df.rename(columns=clean_cols)

    # Generate LaTeX table
    latex = stats_df.to_latex(
        index=False,
        caption='Comparative Analysis of Datasets',
        label='tab:dataset_comparison',
        column_format='lccccc'
    )

    with open('figures/dataset_comparison_table.tex', 'w') as f:
        f.write(latex)

    print("\nAnalysis complete! Results saved in 'figures' directory.")
    print("LaTeX table saved as 'dataset_comparison_table.tex'")