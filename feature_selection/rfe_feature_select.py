import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
# 3. Define the Pipeline with RFE
# ----------------------------------
pipeline = Pipeline([
    ('variance_threshold', VarianceThreshold()),  # Remove constant/near-constant features
    ('scaling', StandardScaler()),
    ('feature_selection', RFE(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        n_features_to_select=500,  # Adjust this number based on your data
        step=50,                   # Number of features to remove at each iteration
        verbose=1                  # Print progress messages from RFE
    )),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# ----------------------------------
# 4. Evaluate the Pipeline using Manual Cross-Validation with Progress Tracking
# ----------------------------------
print("Evaluating pipeline with 5-fold cross-validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

# Loop manually over CV folds with a progress bar
for train_idx, test_idx in tqdm(cv.split(X_imputed, y), total=5, desc="CV Folds"):
    X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    cv_scores.append(score)

print(f"Cross-validated accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

# Optionally, fit the pipeline on the entire dataset
print("Fitting pipeline on the entire dataset...")
pipeline.fit(X_imputed, y)
print("Pipeline training complete.")

# ----------------------------------
# 5. (Optional) Extract the Selected Features
# ----------------------------------
# The 'feature_selection' step is RFE, which stores a boolean mask in the attribute 'support_'
rfe_step = pipeline.named_steps['feature_selection']
selected_feature_mask = rfe_step.support_
selected_features = X_imputed.columns[selected_feature_mask]
print(f"Number of features selected by RFE: {len(selected_features)}")
print("Selected features:")
print(selected_features.tolist())

# ----------------------------------
# 6. (Optional) Visualize the Selected Feature Importance
# ----------------------------------
# If you want to visualize feature importances from the final classifier, you can do so:
classifier = pipeline.named_steps['classifier']
importances = classifier.feature_importances_
# Plot the importances of the selected features
plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances, color='skyblue')
plt.xlabel("Selected Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importances from RandomForest (on RFE-selected features)")
plt.show()
