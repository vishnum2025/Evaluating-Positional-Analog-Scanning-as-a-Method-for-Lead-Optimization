import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# Load data
csv_file_path = '/Users/vishnu/characterized_modifications.csv'
df_filtered = pd.read_csv(csv_file_path)
df_filtered['Molecule1_ID'] = df_filtered['Molecule1_ID'].str.strip()
df_filtered['Molecule2_ID'] = df_filtered['Molecule2_ID'].str.strip()

activity_df = pd.read_csv('/Users/vishnu/filtered_output_Ro5.csv', sep="\t")
activity_df['molecule_chembl_id'] = activity_df['molecule_chembl_id'].str.strip()
activity_df = activity_df[activity_df['standard_value'] > 0]
activity_df['log_standard_value'] = np.log10(activity_df['standard_value'])

# Assuming df_pairs contains Molecule1_ID and Molecule2_ID and activity_df contains activity details

# Filter for EC50 and IC50
activity_ic50 = activity_df[activity_df['standard_type'] == 'IC50']
print(activity_ic50.shape[0]) #1085819

activity_ec50 = activity_df[activity_df['standard_type'] == 'EC50']
print(activity_ec50.shape[0])  #188839

inversion_only_df = df_filtered[(df_filtered['Distance'] == 0) & 
                                 (df_filtered['Modifications'].str.contains("Inversion of stereocenters detected"))].copy()
print(inversion_only_df.shape[0])   #61221



# Merge IC50
merged_ic50 = inversion_only_df.merge(
    activity_ic50[['molecule_chembl_id', 'log_standard_value']],
    left_on='Molecule1_ID', right_on='molecule_chembl_id', suffixes=('', '_1')
).merge(
    activity_ic50[['molecule_chembl_id', 'log_standard_value']],
    left_on='Molecule2_ID', right_on='molecule_chembl_id', suffixes=('_1', '_2')
)
print("done 1, IC50:")
print(merged_ic50.head())
print(merged_ic50.shape[0]) #3263165
# Merge EC50 
merged_ec50 = inversion_only_df.merge(
    activity_ec50[['molecule_chembl_id', 'log_standard_value']],
    left_on='Molecule1_ID', right_on='molecule_chembl_id', suffixes=('', '_1')
).merge(
    activity_ec50[['molecule_chembl_id', 'log_standard_value']],
    left_on='Molecule2_ID', right_on='molecule_chembl_id', suffixes=('_1', '_2')
)
print("done 2, EC50:")
print(merged_ec50.head())
print(merged_ec50.shape[0]) #153742




# Calculate the activity ratios
merged_ic50['IC50_Ratio'] = merged_ic50.apply(
    lambda x: max(x['log_standard_value_1'], x['log_standard_value_2']) /
              min(x['log_standard_value_1'], x['log_standard_value_2']) if min(x['log_standard_value_1'], x['log_standard_value_2']) > 0 else np.nan,
    axis=1
)
print("done 3, IC50 ratios:")
print(merged_ic50['IC50_Ratio'])

merged_ec50['EC50_Ratio'] = merged_ec50.apply(
    lambda x: max(x['log_standard_value_1'], x['log_standard_value_2']) /
              min(x['log_standard_value_1'], x['log_standard_value_2']) if min(x['log_standard_value_1'], x['log_standard_value_2']) > 0 else np.nan,
    axis=1
)
print("done 4, EC50 ratios:")
print(merged_ec50['EC50_Ratio'])




# Chunk processing for the final merge
chunk_size = 100000  
num_chunks = len(merged_ic50) // chunk_size + 1
print(num_chunks)

df_combined_list = []

for i in range(num_chunks):
    ic50_chunk = merged_ic50.iloc[i*chunk_size:(i+1)*chunk_size]
    ec50_chunk = merged_ec50.iloc[i*chunk_size:(i+1)*chunk_size]
    
    combined_chunk = ic50_chunk[['Molecule1_ID', 'Molecule2_ID', 'IC50_Ratio']].merge(
        ec50_chunk[['Molecule1_ID', 'Molecule2_ID', 'EC50_Ratio']],
        on=['Molecule1_ID', 'Molecule2_ID'],
        how='inner'
    )
    
    df_combined_list.append(combined_chunk)

df_combined = pd.concat(df_combined_list, ignore_index=True)
print("done 5")

# Calculating the relative EC50/IC50 ratio
df_combined['Relative_EC50_IC50'] = df_combined['EC50_Ratio'] / df_combined['IC50_Ratio']

# Plotting to visualize the distribution of Relative EC50/IC50
plt.figure(figsize=(10, 6))
sns.histplot(df_combined['Relative_EC50_IC50'].dropna(), kde=True)
plt.title('Distribution of Relative EC50/IC50 Ratios')
plt.xlabel('Relative EC50/IC50 Ratio')
plt.ylabel('Frequency')
plt.show()



