import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Function to calculate Cramér's V for categorical-categorical correlations
def cramers_v(confusion_matrix):
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.to_numpy().sum()
    if n == 0:
        return 0
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

# Function to calculate Point Biserial correlation for numeric-categorical correlations
def numeric_categorical_corr(numeric_series, categorical_series):
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(categorical_series)
    corr, _ = pointbiserialr(numeric_series, encoded)
    return corr

# Load dataset
file_path = "..\Different_classification_algorithms\customer_churn.csv"  # Replace with your dataset path
df = pd.read_csv(file_path)

# Separate numeric and categorical columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(exclude=[np.number]).columns

# Initialize correlation matrix
all_columns = df.columns
correlation_matrix = pd.DataFrame(np.zeros((len(all_columns), len(all_columns))), index=all_columns, columns=all_columns)

# Calculate correlations
for col1 in all_columns:
    for col2 in all_columns:
        if col1 == col2:
            correlation_matrix.loc[col1, col2] = 1.0
        elif col1 in numeric_columns and col2 in numeric_columns:
            correlation_matrix.loc[col1, col2] = df[col1].corr(df[col2])
        elif col1 in categorical_columns and col2 in categorical_columns:
            confusion_mat = pd.crosstab(df[col1], df[col2])
            correlation_matrix.loc[col1, col2] = cramers_v(confusion_mat)
        elif col1 in numeric_columns and col2 in categorical_columns:
            correlation_matrix.loc[col1, col2] = numeric_categorical_corr(df[col1], df[col2])
        elif col1 in categorical_columns and col2 in numeric_columns:
            correlation_matrix.loc[col1, col2] = numeric_categorical_corr(df[col2], df[col1])

# Save analysis and plots to a PDF
output_pdf = "detailed_analysis_fixed.pdf"
with PdfPages(output_pdf) as pdf:
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap="coolwarm", xticklabels=True, yticklabels=True)
    plt.title("Correlation Heatmap (Numeric and Categorical Features)")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Descriptive statistics
    desc_stats = df.describe(include="all").transpose()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    tbl = pd.plotting.table(ax, desc_stats, loc='center', colWidths=[0.2] * len(desc_stats.columns))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.2)
    plt.title("Descriptive Statistics Table")
    pdf.savefig()
    plt.close()

    # Numeric column distributions
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, bins=30, color='blue')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Boxplot for outlier detection
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f"Boxplot for {col}")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Categorical column distributions
    for col in categorical_columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=df[col], order=df[col].value_counts().index, color="blue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Pairplot for numeric columns
    if len(numeric_columns) <= 5:
        sns.pairplot(df[numeric_columns])
        plt.suptitle("Pairplot of Numeric Features", y=1.02)
        pdf.savefig()
        plt.close()

    # Relationships between categorical and numeric columns
    for cat_col in categorical_columns:
        for num_col in numeric_columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=df[cat_col], y=df[num_col], color="skyblue")
            plt.title(f"Boxplot of {num_col} by {cat_col}")
            plt.xlabel(cat_col)
            plt.ylabel(num_col)
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    # Heatmap for missing data
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Data Heatmap")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print(f"Detailed analysis saved in {output_pdf}.")
