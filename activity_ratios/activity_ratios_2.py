# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from tqdm.auto import tqdm

# # Load the data
# csv_file_path = '/Users/vishnu/characterized_modifications.csv'
# df_filtered = pd.read_csv(csv_file_path)
# df_filtered['Molecule1_ID'] = df_filtered['Molecule1_ID'].str.strip()
# df_filtered['Molecule2_ID'] = df_filtered['Molecule2_ID'].str.strip()

# activity_df = pd.read_csv('/Users/vishnu/filtered_output_Ro5.csv', sep="\t")
# activity_df['molecule_chembl_id'] = activity_df['molecule_chembl_id'].str.strip()
# activity_df = activity_df[activity_df['standard_value'] > 0]
# activity_df['log_standard_value'] = np.log10(activity_df['standard_value'])

# inversion_only_df = df_filtered[(df_filtered['Distance'] == 0) & 
#                                 (df_filtered['Modifications'].str.contains("Inversion of stereocenters detected"))].copy()

# def plot_activity_ratios(standard_types):
#     plt.figure(figsize=(10, 6))
#     for standard_type in standard_types:
#         filtered_activity_df = activity_df[activity_df['standard_type'] == standard_type]

#         # Merge the activity data for Molecule1
#         merged_df_1 = inversion_only_df.merge(
#             filtered_activity_df[['molecule_chembl_id', 'log_standard_value']],
#             left_on='Molecule1_ID',
#             right_on='molecule_chembl_id',
#             how='left'
#         )

#         # Merge the activity data for Molecule2
#         merged_df_final = merged_df_1.merge(
#             filtered_activity_df[['molecule_chembl_id', 'log_standard_value']],
#             left_on='Molecule2_ID',
#             right_on='molecule_chembl_id',
#             suffixes=('_1', '_2'),
#             how='left'
#         )

#         # Calculate the activity ratio ensuring it's never less than 1
#         merged_df_final['ActivityRatio'] = merged_df_final.apply(
#             lambda row: max(row['log_standard_value_1'], row['log_standard_value_2']) / 
#                         min(row['log_standard_value_1'], row['log_standard_value_2']) if min(row['log_standard_value_1'], row['log_standard_value_2']) > 0 else np.nan, 
#             axis=1
#         )

#         # Plotting
#         sns.histplot(merged_df_final['ActivityRatio'].dropna(), kde=True, bins=30, log_scale=True, label=standard_type)

#     plt.title('Distribution of Activity Ratios for Molecule Pairs with Inversion of Stereocenters')
#     plt.xlabel('Activity Ratio')
#     plt.ylabel('Frequency')
#     plt.legend(title='Standard Type')
#     plt.show()


# def print_activity_ratios(standard_type):
#     filtered_activity_df = activity_df[activity_df['standard_type'] == standard_type]

#     # Merge the activity data for Molecule1
#     merged_df_1 = inversion_only_df.merge(
#         filtered_activity_df[['molecule_chembl_id', 'log_standard_value']],
#         left_on='Molecule1_ID',
#         right_on='molecule_chembl_id',
#         how='left'
#     )

#     # Merge the activity data for Molecule2
#     merged_df_final = merged_df_1.merge(
#         filtered_activity_df[['molecule_chembl_id', 'log_standard_value']],
#         left_on='Molecule2_ID',
#         right_on='molecule_chembl_id',
#         suffixes=('_1', '_2'),
#         how='left'
#     )

#     # Calculate the activity ratio ensuring it's never less than 1
#     merged_df_final['ActivityRatio'] = merged_df_final.apply(
#         lambda row: max(row['log_standard_value_1'], row['log_standard_value_2']) / 
#                     min(row['log_standard_value_1'], row['log_standard_value_2']) if min(row['log_standard_value_1'], row['log_standard_value_2']) > 0 else np.nan, 
#         axis=1
#     )

#     print(f"Activity Ratios for {standard_type}:")
#     print(merged_df_final['ActivityRatio'].describe())




# for st_type in ['IC50', 'EC50', 'Ki']:
#     print_activity_ratios(st_type)
# plot_activity_ratios(['IC50', 'EC50', 'Ki'])

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import anderson_ksamp
from scipy.stats import mannwhitneyu
from tqdm.auto import tqdm

# Load data
csv_file_path = '/Users/vishnu/characterized_modifications.csv'
df_filtered = pd.read_csv(csv_file_path)
df_filtered['Molecule1_ID'] = df_filtered['Molecule1_ID'].str.strip()
df_filtered['Molecule2_ID'] = df_filtered['Molecule2_ID'].str.strip()

activity_df = pd.read_csv('/Users/vishnu/filtered_output_Ro5.csv', sep="\t")
activity_df['molecule_chembl_id'] = activity_df['molecule_chembl_id'].str.strip()
# activity_df = activity_df[activity_df['standard_value'] > 0]
# activity_df['log_standard_value'] = np.log10(activity_df['standard_value'])
MIN_THRESHOLD = 1


# Filter out non-positive and NaN values
activity_df = activity_df[(activity_df['standard_value'] > 0) & (~activity_df['standard_value'].isna())]

# Apply log transformation only to values greater than the minimum threshold
activity_df['log_standard_value'] = activity_df['standard_value'].apply(lambda x: np.log10(x) if x >= MIN_THRESHOLD else np.log10(MIN_THRESHOLD))

non_positive_log_values = activity_df[activity_df['log_standard_value'] <= 0]

# Display non-positive log_standard_value rows
if not non_positive_log_values.empty:
    print("Non-positive log_standard_value rows:")
    print(non_positive_log_values[['molecule_chembl_id', 'standard_value', 'log_standard_value']])
else:
    print("No non-positive log_standard_value found. The data is clean!")


inversion_only_df = df_filtered[(df_filtered['Distance'] == 0) & 
                                (df_filtered['Modifications'].str.contains("Inversion of stereocenters detected"))].copy()


                                
ic50_data = activity_df[(activity_df['standard_type'] == 'IC50') & (activity_df['log_standard_value'] > 0)]['log_standard_value']
ec50_data = activity_df[(activity_df['standard_type'] == 'EC50') & (activity_df['log_standard_value'] > 0)]['log_standard_value']
ki_data = activity_df[(activity_df['standard_type'] == 'Ki') & (activity_df['log_standard_value'] > 0)]['log_standard_value']

# Plot distributions for visual inspection
plt.figure(figsize=(12, 8))
sns.kdeplot(ic50_data, label='IC50', color='blue', linestyle='--')
sns.kdeplot(ec50_data, label='EC50', color='red', linestyle=':')
sns.kdeplot(ki_data, label='Ki', color='green', linestyle='-.')

plt.title('Distribution of log_standard_value for IC50, EC50, and Ki')
plt.xlabel('log_standard_value')
plt.ylabel('Density')
plt.legend()
plt.show()



# Perform Kolmogorov-Smirnov tests
ks_ic50_ec50 = ks_2samp(ic50_data, ec50_data)
ks_ic50_ki = ks_2samp(ic50_data, ki_data)
ks_ec50_ki = ks_2samp(ec50_data, ki_data)

# Display the results
print(f"IC50 vs EC50: KS statistic = {ks_ic50_ec50.statistic:.4f}, p-value = {ks_ic50_ec50.pvalue:.4g}")
print(f"IC50 vs Ki: KS statistic = {ks_ic50_ki.statistic:.4f}, p-value = {ks_ic50_ki.pvalue:.4g}")
print(f"EC50 vs Ki: KS statistic = {ks_ec50_ki.statistic:.4f}, p-value = {ks_ec50_ki.pvalue:.4g}")




###### CODE FOR RELATIVE FREQUENCY ################

# def plot_activity_ratios(standard_types):
#     plt.figure(figsize=(10, 6))
#     data_for_tests = []
    
#     for standard_type in standard_types:
#         filtered_activity_df = activity_df[activity_df['standard_type'] == standard_type]

#         merged_df_1 = inversion_only_df.merge(
#             filtered_activity_df[['molecule_chembl_id', 'log_standard_value']],
#             left_on='Molecule1_ID',
#             right_on='molecule_chembl_id',
#             how='left'
#         )

#         merged_df_final = merged_df_1.merge(
#             filtered_activity_df[['molecule_chembl_id', 'log_standard_value']],
#             left_on='Molecule2_ID',
#             right_on='molecule_chembl_id',
#             suffixes=('_1', '_2'),
#             how='left'
#         )

#         merged_df_final['ActivityRatio'] = merged_df_final.apply(
#             lambda row: max(row['log_standard_value_1'], row['log_standard_value_2']) /
#                         min(row['log_standard_value_1'], row['log_standard_value_2']) if min(row['log_standard_value_1'], row['log_standard_value_2']) > 0 else np.nan,
#             axis=1
#         )

#         sns.histplot(merged_df_final['ActivityRatio'].dropna(), kde=True, bins=30, log_scale=True, label=standard_type, stat='density')

#         data_for_tests.append(merged_df_final['ActivityRatio'].dropna())

#     plt.title('Normalized Distribution of Activity Ratios for Molecule Pairs with Inversion of Stereocenters')
#     plt.xlabel('Activity Ratio')
#     plt.ylabel('Density')
#     plt.legend(title='Standard Type')
#     plt.show()


# plot_activity_ratios(['IC50', 'EC50', 'Ki'])



