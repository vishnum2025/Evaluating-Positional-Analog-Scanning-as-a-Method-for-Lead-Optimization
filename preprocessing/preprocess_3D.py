import pandas as pd
# Load the dataset
data = pd.read_csv("/Users/vishnu/Desktop/viji_files/sem8/BTP_3/mol_3d_descriptors.csv")
print(f"Dataset shape: {data.shape}")


# Calculate rows with NaN
rows_with_nan = data.isna().any(axis=1).sum()

# Calculate columns with NaN
columns_with_nan = data.isna().any(axis=0).sum()
print(f"Number of rows with NaN values: {rows_with_nan}")
print(f"Number of columns with NaN values: {columns_with_nan}")

## How many NaN values each column has:
# missing_values_per_column = data.isna().sum()
# print(missing_values_per_column)

# Identify columns with all NaN values
completely_empty_columns = data.columns[data.isna().all()]

# Count the number of completely empty columns
num_completely_empty_columns = len(completely_empty_columns)

print(f"Number of completely empty columns (all NaN): {num_completely_empty_columns}")
# print("List of completely empty columns:")
# print(completely_empty_columns.tolist())


missing_percentage = (data.isna().sum() / len(data)) * 100
high_missing_columns = missing_percentage[missing_percentage > 50]  # Adjust threshold if needed
print(f"Columns with >50% missing values: {len(high_missing_columns)}")
print(high_missing_columns.sort_values(ascending=False))
