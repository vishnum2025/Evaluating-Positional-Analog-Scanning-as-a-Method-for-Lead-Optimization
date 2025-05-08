import h2o
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from h2o.estimators.random_forest import H2ORandomForestEstimator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# # --------------------------------------  Select Top 500 Features after difference of common mol_1 and mol_2 features ----------------------------------

# # 1. Initialize H2O
# h2o.init(max_mem_size="6G")  # Adjust memory as needed
# # 2. Import the Precomputed Difference Features CSV using pandas
# print("Loading data...")
# diff_csv_path = "/Users/vishnu/BTP_3/difference_features_df.csv"
# diff_df = pd.read_csv(diff_csv_path)
# print(diff_df.head())
# print(f"Difference features loaded: {diff_df.shape[0]} rows and {diff_df.shape[1]} columns.")

# target = "Potency_Change_Label"

# # 3. Convert the Difference DataFrame to an H2OFrame

# data_diff = h2o.H2OFrame(diff_df)
# # Convert the target column to a factor for classification
# data_diff[target] = data_diff[target].asfactor()
# print("Converted difference features to H2OFrame:")
# print(data_diff.head())


# # 4. Train a Preliminary Random Forest Model for Feature Importance

# # Define the list of feature names (all columns except the target)
# diff_features = [col for col in data_diff.columns if col != target]
# print(f"Training Random Forest on {len(diff_features)} difference features...")

# rf_model = H2ORandomForestEstimator(ntrees=50, seed=1)
# rf_model.train(x=diff_features, y=target, training_frame=data_diff)

# # Obtain variable importance as a Pandas DataFrame
# varimp_df = rf_model.varimp(use_pandas=True)
# print("Top 10 features by variable importance:")
# print(varimp_df.head(10))


# # 5. Export Results

# top_n = 500
# top_features_df = varimp_df.head(top_n)
# output_path = "/Users/vishnu/Desktop/viji_files/sem8/BTP_3/feature_imp_h2o_diff_top_500.csv"
# top_features_df.to_csv(output_path, index=False)
# print(f"Top {top_n} feature importances exported to {output_path}")


# ----------------------------------------------------------  Descriptive stats of top 500 features ------------------------------------------------------




diff_csv_path = "/Users/vishnu/BTP_3/difference_features_df.csv"
diff_df = pd.read_csv(diff_csv_path, low_memory=False)
print(f"Full difference features data: {diff_df.shape[0]} rows and {diff_df.shape[1]} columns.")
print("Sample of the data:")
print(diff_df.head())


# Load the Top 500 Selected Feature Names CSV

top_features_csv = "/Users/vishnu/Desktop/viji_files/sem8/BTP_3/feature_imp_h2o_diff_top_500.csv"
top_features_df = pd.read_csv(top_features_csv)
# Assumption: the CSV has a column named "variable" with the feature names.
selected_features = top_features_df["variable"].tolist()
print(f"Number of selected features: {len(selected_features)}")
print("Selected features sample:")
print(selected_features[:10])


# Subset the Full Difference Dataset to the Selected Features

target = "Potency_Change_Label"
if target in diff_df.columns:
    subset_df = diff_df[selected_features + [target]]
else:
    subset_df = diff_df[selected_features]

print(f"Subset data has: {subset_df.shape[0]} rows and {subset_df.shape[1]} columns.")


# Convert Selected Feature Columns to Numeric

# Ensure that the selected feature columns are numeric (they should be, but in case of errors, we coerce).
for col in selected_features:
    subset_df[col] = pd.to_numeric(subset_df[col], errors='coerce')

# Optionally, you could drop rows with NaN values if many are missing:
# subset_df = subset_df.dropna()


# Descriptive Statistics and Visualization

# Exclude target from numeric analysis if present
if target in subset_df.columns:
    X = subset_df.drop(columns=[target])
else:
    X = subset_df.copy()

print("Summary statistics:")
print(X.describe())

# Plot histograms for a subset of features (e.g., top 10 columns)
top10_features = X.columns[:10]
X[top10_features].hist(bins=30, figsize=(15, 10))
plt.suptitle("Histograms of Top 10 Selected Difference Features")
plt.show()

# Compute and display a correlation matrix heatmap for the selected features
corr_matrix = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True,
            cbar_kws={"shrink": 0.5}, xticklabels=False, yticklabels=False)
plt.title("Correlation Matrix Heatmap of Top 500 Selected Difference Features")
plt.show()

# ----------------------------------
# 6. Principal Component Analysis (PCA)
# ----------------------------------
# Standardize the features before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Impute missing values in the standardized data (PCA cannot handle NaNs)
imputer = SimpleImputer(strategy="mean")
X_scaled_imputed = imputer.fit_transform(X_scaled)

# Perform PCA on the imputed, scaled data (compute top 10 components)
pca = PCA(n_components=10)
pca_result = pca.fit_transform(X_scaled_imputed)
explained_variance = pca.explained_variance_ratio_

print("Explained variance ratio for the top 10 principal components:")
for i, ratio in enumerate(explained_variance, start=1):
    print(f"PC{i}: {ratio:.4f}")

plt.figure(figsize=(8, 6))
plt.bar(range(1, 11), explained_variance, alpha=0.7, align='center')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Principal Components")
plt.xticks(range(1, 11))
plt.show()
