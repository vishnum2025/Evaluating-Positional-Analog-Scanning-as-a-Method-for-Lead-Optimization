# import pandas as pd

# # Load the previously saved output CSV file
# file_path = '/Users/vishnu/single_atom_additions_2.csv'
# df = pd.read_csv(file_path)

# # Filter for rows where 'Single_Addition' is True
# df_true_additions = df[df['Single_Addition'] == True]

# # Optional: Save the filtered DataFrame to a new CSV file if needed
# output_filtered_path = '/Users/vishnu/true_single_atom_additions.csv'
# df_true_additions.to_csv(output_filtered_path, index=False)

# # Print some information to check
# print(f"Filtered DataFrame with 'Single_Addition' == True saved to {output_filtered_path}")
# print("Sample of the filtered DataFrame:")
# print(df_true_additions.head())
# print(len(df_true_additions))





import pandas as pd

# Load the previously saved output CSV file
file_path = '/Users/vishnu/Desktop/viji files/sem 6/BTP/identifying additions/single_atom_additions_2.csv'
df = pd.read_csv(file_path)

# Filter for rows where 'Single_Addition' is True
df_true_additions = df[df['Single_Addition'] == True]

# Remove duplicates based on specific columns (adjust columns as needed)
df_unique = df_true_additions.drop_duplicates(subset=['Molecule1_ID', 'Molecule2_ID', 'Common_Scaffold'])

# Optional: Save the unique DataFrame to a new CSV file if needed
output_filtered_path = '/Users/vishnu/Desktop/viji files/sem 6/BTP/identifying additions/unique_single_atom_additions.csv'
df_unique.to_csv(output_filtered_path, index=False)

# Print some information to check
print(f"Filtered DataFrame with unique 'Single_Addition' == True saved to {output_filtered_path}")
print("Sample of the filtered DataFrame:")
print(df_unique.head())
print("Number of rows:", len(df_unique))
