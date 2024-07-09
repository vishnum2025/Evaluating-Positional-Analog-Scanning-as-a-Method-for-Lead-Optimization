import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# def get_activity(molecule_id):
#     activities = activity_client.filter(molecule_chembl_id=molecule_id).filter(standard_type="IC50")
#     for act in activities:
#         if 'standard_value' in act and 'standard_units' in act and act['standard_units'] == 'nM':
#             return float(act['standard_value'])
#     return None

#Loading the data
csv_file_path = '/Users/vishnu/characterized_modifications.csv'
df_filtered = pd.read_csv(csv_file_path)
df_filtered['Molecule1_ID'] = df_filtered['Molecule1_ID'].str.strip()
df_filtered['Molecule2_ID'] = df_filtered['Molecule2_ID'].str.strip()
activity_df = pd.read_csv('/Users/vishnu/filtered_output_Ro5.csv', sep="\t")
activity_df['molecule_chembl_id'] = activity_df['molecule_chembl_id'].str.strip()
activity_df = activity_df[activity_df['standard_value'] > 0]
activity_df['log_standard_value'] = np.log10(activity_df['standard_value'])
print(activity_df['log_standard_value'].describe())

inversion_only_df = df_filtered[(df_filtered['Distance'] == 0) & 
                                (df_filtered['Modifications'].str.contains("Inversion of stereocenters detected"))].copy()


def plot_activity_ratios(standard_type):
    filtered_activity_df = activity_df[activity_df['standard_type'] == standard_type]

    # Merge the activity data for Molecule1
    merged_df_1 = inversion_only_df.merge(
        filtered_activity_df[['molecule_chembl_id', 'log_standard_value']],
        left_on='Molecule1_ID',
        right_on='molecule_chembl_id',
        how='left'
    )

    # Merge the activity data for Molecule2
    merged_df_final = merged_df_1.merge(
        filtered_activity_df[['molecule_chembl_id', 'log_standard_value']],
        left_on='Molecule2_ID',
        right_on='molecule_chembl_id',
        suffixes=('_1', '_2'),
        how='left'
    )

    # Calculate the activity ratio and plot
    merged_df_final['ActivityRatio'] = merged_df_final['log_standard_value_1'] - merged_df_final['log_standard_value_2']
    print(merged_df_final['ActivityRatio'].describe())
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_df_final['ActivityRatio'].dropna(), kde=True, bins=30, log_scale=True)
    plt.title(f'Distribution of {standard_type} Activity Ratios for Molecule Pairs with Inversion of Stereocenters')
    plt.xlabel('Activity Ratio')
    plt.ylabel('Frequency')
    plt.show()

# Plot for each standard type
for st_type in ['IC50', 'EC50', 'Ki']:
    plot_activity_ratios(st_type)


