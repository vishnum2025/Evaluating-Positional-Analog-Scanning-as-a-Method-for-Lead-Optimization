import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
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
# 3. Feature Selection using Mutual Information
#     with Progress Tracking
# ----------------------------------

# Set the number of top features you want to select (e.g., top 500 features)
k = 500
print(f"Calculating Mutual Information scores for each feature (selecting top {k})...")

# List to store MI scores for each feature
mi_scores = []
feature_names = X_imputed.columns.tolist()

# Loop through each feature with a progress bar
for feature in tqdm(feature_names, desc="MI Score Calculation"):
    # Compute the mutual information score for this single feature
    score = mutual_info_classif(X_imputed[[feature]], y, random_state=42)
    mi_scores.append(score[0])

# Convert the list of scores to a NumPy array
mi_scores = np.array(mi_scores)

# Select the indices of the top k features
top_k_indices = np.argsort(mi_scores)[-k:]
selected_features = X_imputed.columns[top_k_indices]

# Create a reduced feature matrix containing only the selected features
X_mi = X_imputed[selected_features]

print(f"\nNumber of features selected by Mutual Information: {len(selected_features)}")
print("Selected features:")
print(selected_features.tolist())

# Optionally, display the top 10 features by MI score in a sorted table
mi_df = pd.DataFrame({"Feature": feature_names, "MI_Score": mi_scores})
mi_df_sorted = mi_df.sort_values("MI_Score", ascending=False)
print("\nTop 10 features by Mutual Information Score:")
print(mi_df_sorted.head(10))

# ----------------------------------
# 4. Evaluate the Selected Features
# ----------------------------------
print("\nEvaluating the selected features using RandomForestClassifier with 5-fold cross-validation...")

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Evaluate model performance using 5-fold cross-validation
scores = cross_val_score(rf, X_mi, y, cv=5, scoring='accuracy')
print(f"Random Forest accuracy using MI-selected features: {scores.mean():.3f} ± {scores.std():.3f}")


# Initialize the XGBoost classifier
# Note: use_label_encoder=False and setting an eval_metric avoids warnings.
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Evaluate model performance using 5-fold cross-validation
scores = cross_val_score(xgb_model, X_mi, y, cv=5, scoring='accuracy')
print(f"XGBoost accuracy using MI-selected features: {scores.mean():.3f} ± {scores.std():.3f}")








# # ----------------------------------
# # 5. Optional: Plot MI Scores for All Features
# # ----------------------------------
# plt.figure(figsize=(12, 6))
# plt.bar(range(len(mi_scores)), mi_scores, color='skyblue')
# plt.xlabel("Feature Index")
# plt.ylabel("Mutual Information Score")
# plt.title("Mutual Information Scores for All Features")
# plt.show()
