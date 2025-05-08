import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from sklearn.impute import SimpleImputer
from tqdm import tqdm

# ----------------------------------
# 1. Load and Prepare the Data
# ----------------------------------
print("Loading data...")

# Updated file path
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

# Convert features to numeric (non-convertible entries become NaN) and drop columns that are completely NaN
X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, how="all")

print(f"Data loaded: {len(data)} samples and {X.shape[1]} features before imputation.")

# ----------------------------------
# 2. Impute Missing Values
# ----------------------------------
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print(f"Data after imputation: {X_imputed.shape[1]} features with no missing values.")

# ----------------------------------
# 3. Compute Kendall's tau for Each Feature
# ----------------------------------
print("Computing Kendall's tau correlations for each feature...")
kendall_scores = []
feature_names = X_imputed.columns.tolist()

# Loop through each feature with progress tracking
for feature in tqdm(feature_names, desc="Kendall tau computation"):
    # Compute Kendall tau between the feature and the target
    tau, p_value = kendalltau(X_imputed[feature], y)
    # Store the tau value (you can also store p_value if desired)
    kendall_scores.append(tau)

kendall_scores = np.array(kendall_scores)

# Create a DataFrame for the results
kendall_df = pd.DataFrame({
    "Feature": feature_names,
    "Kendall_tau": kendall_scores
})

# Add a column for the absolute value (to rank by strength regardless of direction)
kendall_df["abs_tau"] = np.abs(kendall_df["Kendall_tau"])

# Sort by absolute Kendall tau in descending order
kendall_df_sorted = kendall_df.sort_values("abs_tau", ascending=False)

print("\nTop 10 features by absolute Kendall tau correlation:")
print(kendall_df_sorted.head(10))

# ----------------------------------
# 4. Select Top-k Features Based on Kendall tau
# ----------------------------------
# For example, select the top 500 features
k = 500
selected_features = kendall_df_sorted.head(k)["Feature"].tolist()
X_kendall = X_imputed[selected_features]

print(f"\nNumber of features selected by Kendall tau: {len(selected_features)}")
print("Selected features:")
print(selected_features)

# ----------------------------------
# 5. (Optional) Visualize the Kendall tau Scores
# ----------------------------------
plt.figure(figsize=(12, 6))
plt.bar(range(len(kendall_scores)), kendall_scores, color='skyblue')
plt.xlabel("Feature Index")
plt.ylabel("Kendall tau")
plt.title("Kendall tau Correlation Scores for All Features")
plt.show()

# Now you can use X_kendall as your feature set for further modeling.
