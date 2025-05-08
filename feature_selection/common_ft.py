import pandas as pd

# Load the CSV files into DataFrames
df1 = pd.read_csv('/Users/vishnu/Desktop/viji_files/sem8/BTP_3/kaggle ouptut files/selected_numeric_features.csv')
df2 = pd.read_csv('/Users/vishnu/Desktop/viji_files/sem8/BTP_3/kaggle ouptut files/selected_variance_0.8_features.csv')

# Extract the "Features" column as sets for each DataFrame
features1 = set(df1['Features'])
features2 = set(df2['Features'])

# Find the common features using set intersection
common_features = features1.intersection(features2)

# Print the common features and their count
print("Common Features:", common_features)
print("Number of common features:", len(common_features))

# Convert the set of common features to a DataFrame
common_features_df = pd.DataFrame(list(common_features), columns=['Features'])

# Save the common features to a CSV file
common_features_df.to_csv('/Users/vishnu/Desktop/viji_files/sem8/BTP_3/kaggle ouptut files/common_features.csv', index=False)
