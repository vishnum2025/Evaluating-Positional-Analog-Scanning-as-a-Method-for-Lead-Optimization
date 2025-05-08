################################################################ RFE ################################################

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import RFE, VarianceThreshold
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import StratifiedKFold
# from tqdm import tqdm

# # ----------------------------------
# # 1. Load and Prepare the Data
# # ----------------------------------
# print("Loading data...")

# # Updated file path
# file_path = "/Users/vishnu/BTP_3/mol_3d_descriptors_final.csv"
# data = pd.read_csv(file_path)

# # Define non-feature columns (metadata, labels, etc.)
# non_feature_cols = [
#     "canonical_smiles_1", "canonical_smiles_2",
#     "Potency_Change", "Potency_Change_Category", "Potency_Change_Label"
# ]

# # Separate features and target
# X = data.drop(columns=non_feature_cols)
# y = data["Potency_Change_Label"]

# # Convert features to numeric (non-convertible entries become NaN) and drop columns that are completely NaN
# X = X.apply(pd.to_numeric, errors="coerce")
# X = X.dropna(axis=1, how="all")

# print(f"Data loaded: {len(data)} samples and {X.shape[1]} features before imputation.")

# # ----------------------------------
# # 2. Impute Missing Values
# # ----------------------------------
# print("Imputing missing values...")
# imputer = SimpleImputer(strategy='mean')
# X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
# print(f"Data after imputation: {X_imputed.shape[1]} features with no missing values.")

# # ----------------------------------
# # 3. Combine Corresponding Features for mol_1 and mol_2
# # ----------------------------------
# print("Combining corresponding mol_1 and mol_2 features...")

# # Assuming features are prefixed with 'mol_1_' and 'mol_2_'
# mol1_cols = [col for col in X_imputed.columns if col.startswith("mol_1_")]
# mol2_cols = [col for col in X_imputed.columns if col.startswith("mol_2_")]

# # Sort the lists to ensure correspondence (if column order is not already guaranteed)
# mol1_cols.sort()
# mol2_cols.sort()

# # Check that the number of features is the same
# if len(mol1_cols) != len(mol2_cols):
#     raise ValueError("The number of mol_1 and mol_2 features do not match!")

# # Compute the difference using the underlying NumPy arrays to force element-wise subtraction.
# # This avoids pandas aligning the DataFrames by column name.
# diff_array = X_imputed[mol1_cols].values - X_imputed[mol2_cols].values

# # Now create new column names.
# # For example, remove "mol_1_" prefix and append "_diff"
# new_col_names = [col.replace("mol_1_", "") + "_diff" for col in mol1_cols]

# # Convert the resulting NumPy array back to a DataFrame with the new column names.
# X_diff = pd.DataFrame(diff_array, columns=new_col_names)

# print(f"Combined feature matrix created with shape: {X_diff.shape}")
# # Now, X_diff has one column per descriptor (approximately half the total columns in X_imputed).

# # ----------------------------------
# # 4. Define the Pipeline with RFE Using the Combined Features
# # ----------------------------------
# pipeline = Pipeline([
#     ('variance_threshold', VarianceThreshold()),  # Remove constant/near-constant features
#     ('scaling', StandardScaler()),
#     ('feature_selection', RFE(
#         estimator=RandomForestClassifier(n_estimators=100, random_state=42),
#         n_features_to_select=500,  # Adjust this number based on your data
#         step=50,                   # Number of features to remove at each iteration
#         verbose=1                  # Enable verbose output for RFE progress
#     )),
#     ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
# ])

# # ----------------------------------
# # 5. Evaluate the Pipeline Using Manual Cross-Validation with Progress Tracking
# # ----------------------------------
# print("Evaluating pipeline with 5-fold cross-validation...")

# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = []

# # Loop manually over CV folds with a progress bar
# for train_idx, test_idx in tqdm(cv.split(X_diff, y), total=5, desc="CV Folds"):
#     X_train, X_test = X_diff.iloc[train_idx], X_diff.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
#     pipeline.fit(X_train, y_train)
#     score = pipeline.score(X_test, y_test)
#     cv_scores.append(score)

# print(f"Cross-validated accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")



################################################################ Tree based methods ################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
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

# Convert all feature columns to numeric (non-convertible entries become NaN)
X = X.apply(pd.to_numeric, errors="coerce")
# Drop columns that are completely NaN
X = X.dropna(axis=1, how="all")

print(f"Data loaded: {len(data)} samples and {X.shape[1]} features before imputation.")

# ----------------------------------
# 2. Impute Missing Values
# ----------------------------------
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print(f"Data after imputation: {X_imputed.shape[1]} features with no missing values.")

# Note: Here, X_imputed still contains both sets of features separately (e.g., mol_1_xyz and mol_2_xyz)

# ----------------------------------
# 3. Define the Pipeline with Tree-Based Feature Selection
# ----------------------------------
# We use a tree-based method (ExtraTreesClassifier) to select important features.
pipeline_tree = Pipeline([
    ('variance_threshold', VarianceThreshold()),  # Remove constant/near-constant features
    ('scaling', StandardScaler()),
    ('feature_selection', SelectFromModel(
        ExtraTreesClassifier(n_estimators=100, random_state=42),
        threshold='median'  # Select features with importance above the median importance
    )),
    ('classifier', ExtraTreesClassifier(n_estimators=100, random_state=42))
])

# ----------------------------------
# 4. Evaluate the Pipeline Using Manual Cross-Validation with Progress Tracking
# ----------------------------------
print("Evaluating pipeline with 5-fold cross-validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

# Loop manually over CV folds with a progress bar
for train_idx, test_idx in tqdm(cv.split(X_imputed, y), total=5, desc="CV Folds"):
    X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    pipeline_tree.fit(X_train, y_train)
    score = pipeline_tree.score(X_test, y_test)
    cv_scores.append(score)

print(f"Cross-validated accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
