# import h2o
# from h2o.automl import H2OAutoML
# from h2o.estimators.random_forest import H2ORandomForestEstimator

# # Initialize H2O with increased memory allocation (adjust as needed)
# h2o.init(max_mem_size="4G")  # Increase this value if your system allows

# # ----------------------------------
# # 1. Import the Data
# # ----------------------------------
# data_path = "/Users/vishnu/BTP_3/mol_3d_descriptors_final.csv"
# data = h2o.import_file(data_path)

# # Create a subset of roughly 10,000 rows using split_frame
# subset_ratio = 10000 / data.nrows
# subsets = data.split_frame(ratios=[subset_ratio], seed=1234)
# subset = subsets[0]  # Use the first split as your subset

# # Define non-feature columns (metadata, labels, etc.)
# non_feature_cols = [
#     "canonical_smiles_1", "canonical_smiles_2", 
#     "Potency_Change", "Potency_Change_Category", "Potency_Change_Label"
# ]

# # Specify target and features. Remove non-feature columns.
# target = "Potency_Change_Label"
# features = [col for col in subset.columns if col not in non_feature_cols]

# # Ensure the target is treated as a factor (categorical) for classification
# subset[target] = subset[target].asfactor()

# print(f"Data imported: {data.nrows} rows and {data.ncols} columns.")
# print(f"Subset created: {subset.nrows} rows.")

# # ----------------------------------
# # 2. Preliminary Feature Selection using H2O Random Forest
# # ----------------------------------
# print("Training a preliminary Random Forest model for feature importance...")
# rf_model = H2ORandomForestEstimator(ntrees=50, seed=1)
# rf_model.train(x=features, y=target, training_frame=subset)

# # Obtain variable importance as a Pandas DataFrame
# varimp_df = rf_model.varimp(use_pandas=True)
# print("Top features by variable importance:")
# print(varimp_df.head(10))

# # Choosing the top N features
# top_n = 100
# selected_features = varimp_df['variable'].head(top_n).tolist()
# print(f"Selected top {top_n} features for AutoML:")
# print(selected_features)

# # ----------------------------------
# # 3. Run H2O AutoML using the Selected Features
# # ----------------------------------
# aml = H2OAutoML(max_models=20, max_runtime_secs=7200, seed=1)
# aml.train(x=selected_features, y=target, training_frame=subset)

# # ----------------------------------
# # 4. Review the Leaderboard and Best Model
# # ----------------------------------
# lb = aml.leaderboard
# print("Leaderboard:")
# print(lb.head(rows=lb.nrows))

# best_model = aml.leader
# print("Best Model:")
# print(best_model)

# # ----------------------------------
# # 5. Export Results to CSV and Text File
# # ----------------------------------
# # Export the leaderboard to CSV
# lb_df = aml.leaderboard.as_data_frame()
# lb_df.to_csv("/Users/vishnu/Desktop/viji_files/sem8/BTP_3/automl_leaderboard.csv", index=False)
# print("Leaderboard exported to automl_leaderboard.csv")

# # Export best model summary to a text file.
# # best_model.summary() returns a string summary if as_html is False.
# best_model_summary = best_model.summary(as_html=False)
# with open("/Users/vishnu/Desktop/viji_files/sem8/BTP_3/best_model_summary.txt", "w") as f:
#     f.write(best_model_summary)
# print("Best model summary exported to best_model_summary.txt")




######################################################### Feature importance  #######################################
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.random_forest import H2ORandomForestEstimator
import pandas as pd
from tqdm import tqdm

# Initialize H2O with increased memory allocation (adjust as needed)
h2o.init(max_mem_size="6G")  # Increase this value if your system allows

# ----------------------------------
# 1. Import the Data
# ----------------------------------
data_path = "/Users/vishnu/BTP_3/mol_3d_descriptors_final.csv"
data = h2o.import_file(data_path)

# Convert the H2OFrame to a pandas DataFrame for easier manipulation, using multi-threading if available
df = data.as_data_frame(use_multi_thread=True)

# Define non-feature columns (metadata, labels, etc.)
non_feature_cols = [
    "canonical_smiles_1", "canonical_smiles_2", 
    "Potency_Change", "Potency_Change_Category", "Potency_Change_Label"
]

# Specify target column
target = "Potency_Change_Label"

print(f"Data imported: {df.shape[0]} rows and {df.shape[1]} columns.")

# ----------------------------------
# 2. Compute Difference Features for Common Descriptors
# ----------------------------------
# Identify mol_1 and mol_2 columns based on their prefixes
mol1_cols = [col for col in df.columns if col.startswith("mol_1_")]
mol2_cols = [col for col in df.columns if col.startswith("mol_2_")]

# Create dictionaries mapping the descriptor (without prefix) to the full column name
mol1_dict = {col.replace("mol_1_", ""): col for col in mol1_cols}
mol2_dict = {col.replace("mol_2_", ""): col for col in mol2_cols}

# Find common descriptor keys
common_keys = sorted(set(mol1_dict.keys()) & set(mol2_dict.keys()))
print(f"Found {len(common_keys)} common descriptors.")

# Instead of inserting columns one-by-one, build a dictionary of difference series.
diff_dict = {}
for key in tqdm(common_keys, desc="Computing differences"):
    col1 = mol1_dict[key]
    col2 = mol2_dict[key]
    diff_dict[key + "_diff"] = df[col1] - df[col2]

# Create a DataFrame from the dictionary of difference features
diff_df = pd.DataFrame(diff_dict)

# Include the target column in the new DataFrame
diff_df[target] = df[target]

print(f"Difference feature DataFrame shape: {diff_df.shape}")

# Convert the pandas DataFrame back to an H2OFrame
data_diff = h2o.H2OFrame(diff_df)
data_diff[target] = data_diff[target].asfactor()

# Define the list of difference features (exclude the target)
diff_features = [col for col in data_diff.columns if col != target]

# ----------------------------------
# 3. Preliminary Feature Importance using H2O Random Forest
# ----------------------------------
print("Training a preliminary Random Forest model on difference features...")
rf_model = H2ORandomForestEstimator(ntrees=50, seed=1)
rf_model.train(x=diff_features, y=target, training_frame=data_diff)

# Obtain variable importance as a Pandas DataFrame
varimp_df = rf_model.varimp(use_pandas=True)
print("Top features by variable importance (difference features):")
print(varimp_df.head(10))

# ----------------------------------
# 4. Select Top-N Features and Export Results
# ----------------------------------
# Set the number of top features you want to export (e.g., 500)
top_n = 500
print(f"Selecting top {top_n} features based on importance...")

top_features_df = varimp_df.head(top_n)

# (Optional) Process each selected feature with progress tracking
print("Processing selected features:")
for idx, row in tqdm(top_features_df.iterrows(), total=top_features_df.shape[0], desc="Processing Features"):
    # Custom processing can be added here if needed.
    pass

# Export the selected top features to a CSV file
output_path = f"/Users/vishnu/Desktop/viji_files/sem8/BTP_3/feature_imp_h2o_diff_top_{top_n}.csv"
top_features_df.to_csv(output_path, index=False)
print(f"Feature importance exported to {output_path}")
