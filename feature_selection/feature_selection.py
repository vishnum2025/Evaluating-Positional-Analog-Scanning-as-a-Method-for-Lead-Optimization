####################################    Removing Low variance features   #############################


# import pandas as pd
# from sklearn.feature_selection import VarianceThreshold

# # Load the final dataset
# data = pd.read_csv("/Users/vishnu/Desktop/viji_files/sem8/BTP_3/mol_3d_descriptors_final.csv")

# #print(data.shape)


# # Drop non-feature columns (SMILES, labels, categories)
# non_feature_cols = [
#     "canonical_smiles_1", "canonical_smiles_2",
#     "Potency_Change", "Potency_Change_Category", "Potency_Change_Label"
# ]
# X = data.drop(columns=non_feature_cols)

# # Convert all columns to numeric (ignore errors, then drop columns with non-numeric values)
# X = X.apply(pd.to_numeric, errors="coerce")
# X = X.dropna(axis=1, how="all")  # Drop columns that couldn’t be converted

# # Initialize VarianceThreshold (remove features with >95% constant values)
# selector = VarianceThreshold(threshold=0.05 * (1 - 0.05))
# X_high_variance = selector.fit_transform(X)

# # Get retained feature names
# retained_features = X.columns[selector.get_support()]

# # Create a DataFrame with high-variance features
# data_high_variance = pd.DataFrame(X_high_variance, columns=retained_features)

# # Add back non-feature columns (SMILES, labels, etc.)
# data_high_variance = pd.concat([
#     data[non_feature_cols],  # Include metadata
#     data_high_variance
# ], axis=1)

# # Save the reduced dataset
# data_high_variance.to_csv(
#     "/Users/vishnu/Desktop/viji_files/sem8/BTP_3/ft_select_mol_3d_descriptors_reduced.csv",
#     index=False
# )




# data = pd.read_csv("/Users/vishnu/Desktop/viji_files/sem8/BTP_3/variance_filter_mol_3d_descriptors_reduced.csv")

# print(data.shape)


#################################   statistical testing  ######################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, probplot

# ------------------------------
# 1. Load and Prepare the Data
# ------------------------------

# Update the file path as needed
file_path = "/Users/vishnu/BTP_3/mol_3d_descriptors_final.csv"
data = pd.read_csv(file_path)

# Define non-feature columns (metadata, labels, etc.)
non_feature_cols = [
    "canonical_smiles_1", "canonical_smiles_2",
    "Potency_Change", "Potency_Change_Category", "Potency_Change_Label"
]

# Separate features and target
X = data.drop(columns=non_feature_cols)
y = data["Potency_Change_Label"]

# Convert all feature columns to numeric (ignoring conversion errors) and drop columns that are completely NaN
X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, how="all")

# ------------------------------
# 2. Statistical Testing
# ------------------------------

# List of feature names
features = X.columns

# Sample a subset of features (here, 50 or fewer if not enough features exist)
n_features_sample = min(50, len(features))
sample_features = np.random.choice(features, size=n_features_sample, replace=False)

# Get unique class labels from y
unique_labels = y.unique()

# Lists to store the results
normality_results = []  # (feature, average Shapiro-Wilk p-value across classes)
variance_results = []   # (feature, Levene's test p-value)

# Loop over each sampled feature
for feature in sample_features:
    shapiro_p_values = []
    
    # For each class, perform the Shapiro-Wilk test after dropping missing values
    for label in unique_labels:
        # Extract data for the current feature and class, and drop NaN values
        data_subset = X[feature][y == label].dropna()
        
        # Proceed only if there are enough observations
        if len(data_subset) < 3:
            continue
        # Check if the data is constant (in which case the test isn't applicable)
        if data_subset.nunique() == 1:
            shapiro_p_values.append(np.nan)
        else:
            stat, p_value = shapiro(data_subset)
            shapiro_p_values.append(p_value)
    
    # Compute the average p-value for normality (ignoring NaNs)
    if shapiro_p_values:
        avg_p_value = np.nanmean(shapiro_p_values)
    else:
        avg_p_value = np.nan
    normality_results.append((feature, avg_p_value))
    
    # For Levene's test, collect groups for each class after dropping NaN values
    groups = [X[feature][y == label].dropna() for label in unique_labels if len(X[feature][y == label].dropna()) > 0]
    if len(groups) > 1:
        stat_levene, p_levene = levene(*groups)
    else:
        p_levene = np.nan
    variance_results.append((feature, p_levene))

# Convert results to DataFrames for easier viewing
normality_df = pd.DataFrame(normality_results, columns=["Feature", "Avg_P_Normality"])
variance_df = pd.DataFrame(variance_results, columns=["Feature", "P_Variance"])

# Sort the DataFrames so that features with the lowest p-values (i.e., most likely to deviate) come first
normality_df_sorted = normality_df.sort_values("Avg_P_Normality")
variance_df_sorted = variance_df.sort_values("P_Variance")

print("Top 10 features with the lowest average p-value for normality (indicating possible deviation from normality):")
print(normality_df_sorted.head(10))

print("\nTop 10 features with the lowest p-value for variance homogeneity (indicating heteroscedasticity):")
print(variance_df_sorted.head(10))

