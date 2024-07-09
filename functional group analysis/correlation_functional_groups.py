import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, f_oneway, shapiro, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

# File paths
file_path = '/Users/vishnu/Desktop/viji files/sem 6/BTP/identifying additions/unique_single_atom_additions.csv'
activity_file_path = '/Users/vishnu/filtered_output_Ro5_logP.csv'

# Load data
df_unique = pd.read_csv(file_path)
activity_df = pd.read_csv(activity_file_path, sep=",")
activity_df['molecule_chembl_id'] = activity_df['molecule_chembl_id'].str.strip()

# Merge activity data for the first molecule
df_merged_1 = df_unique.merge(
    activity_df[['molecule_chembl_id', 'standard_value', 'logP', 'canonical_smiles']],
    left_on='Molecule1_ID', right_on='molecule_chembl_id'
).rename(columns={
    'standard_value': 'standard_value_1',
    'logP': 'logP_1',
    'canonical_smiles': 'canonical_smiles_1'
}).drop(columns=['molecule_chembl_id'])

# Merge activity data for the second molecule
df_merged_2 = df_merged_1.merge(
    activity_df[['molecule_chembl_id', 'standard_value', 'logP', 'canonical_smiles']],
    left_on='Molecule2_ID', right_on='molecule_chembl_id'
).rename(columns={
    'standard_value': 'standard_value_2',
    'logP': 'logP_2',
    'canonical_smiles': 'canonical_smiles_2'
}).drop(columns=['molecule_chembl_id'])

df_merged_2 = df_merged_2.loc[:, ~df_merged_2.columns.duplicated()]

# Ensure canonical_smiles columns are string type
df_merged_2['canonical_smiles_1'] = df_merged_2['canonical_smiles_1'].astype(str)
df_merged_2['canonical_smiles_2'] = df_merged_2['canonical_smiles_2'].astype(str)

# Calculate potency change
df_merged_2['Potency_Change'] = df_merged_2['standard_value_2'] - df_merged_2['standard_value_1']
df_merged_2 = df_merged_2[df_merged_2['Potency_Change'].abs() < 1e6]

# Define Functional Groups with correct SMARTS patterns
functional_groups = {
    'Methyl': '[CX4H3]',
    'Aromatic': 'c1ccccc1',
    'Carbonyl': 'C=O',
    'Chloride': '[Cl]',
    'Hydroxyl': '[OH]',
    'Amine': '[NX3;H2]',
    'Ether': '[C-O-C]',
    'Carboxyl': '[CX3](=O)[OX2H1]',
    'Sulfonyl': '[$([S](=O)(=O))]',
    'Aldehyde': '[CX3H1](=O)',
    'Ester': '[CX3](=O)[OX2]',
    'Nitro': '[N+](=O)[O-]',
    'Thiol': '[SH]',
    'Imine': '[CX2]=[NX1]',
    'Amide': '[NX3][CX3](=[OX1])[#6]',
    'Phosphate': '[PX4](=O)(O)(O)(O)'
}

# Identify functional groups in the molecules
def identify_functional_groups(smiles, functional_groups):
    mol = Chem.MolFromSmiles(smiles)
    found_groups = []
    if mol:
        for group_name, smarts in functional_groups.items():
            smarts_mol = Chem.MolFromSmarts(smarts)
            if smarts_mol and mol.HasSubstructMatch(smarts_mol):
                found_groups.append(group_name)
    return found_groups

# Identify functional groups in the molecules
df_merged_2['Functional_Groups_1'] = df_merged_2['canonical_smiles_1'].apply(identify_functional_groups, args=(functional_groups,))
df_merged_2['Functional_Groups_2'] = df_merged_2['canonical_smiles_2'].apply(identify_functional_groups, args=(functional_groups,))


# Add functional group presence as binary columns   
functional_groups_list = list(functional_groups.keys())
for group in functional_groups_list:
    df_merged_2[f'FG_1_{group}'] = df_merged_2['Functional_Groups_1'].apply(lambda x: 1 if group in x else 0)
    df_merged_2[f'FG_2_{group}'] = df_merged_2['Functional_Groups_2'].apply(lambda x: 1 if group in x else 0)


# Identify addition types using the extended list of functional groups
def identify_addition_type(smiles_2, smiles_1, functional_groups):
    mol1 = Chem.MolFromSmiles(smiles_1)
    mol2 = Chem.MolFromSmiles(smiles_2)
    if not mol1 or not mol2:
        return 'Unknown'
    
    for group_name, smarts in functional_groups.items():
        try:
            smarts_mol = Chem.MolFromSmarts(smarts)
            if smarts_mol and mol2.HasSubstructMatch(smarts_mol) and not mol1.HasSubstructMatch(smarts_mol):
                return group_name
        except Exception as e:
            print(f"Error parsing SMARTS '{smarts}' for group '{group_name}': {e}")
    return 'Other'

# Identify addition types using the extended list of functional groups
df_merged_2['Addition_Type_Extended'] = df_merged_2.apply(
    lambda row: identify_addition_type(row['canonical_smiles_2'], row['canonical_smiles_1'], functional_groups), axis=1
)

# Step 1: Detailed Breakdown of "Other" Category
df_other = df_merged_2[df_merged_2['Addition_Type_Extended'] == 'Other']

# Display the first few rows for manual inspection
print(df_other[['canonical_smiles_1', 'canonical_smiles_2']].head())

# Analyze and visualize the frequency of functional groups in the "Other" category
df_other['Functional_Groups_Other'] = df_other['canonical_smiles_2'].apply(identify_functional_groups, args=(functional_groups,))
all_groups_other = [group for sublist in df_other['Functional_Groups_Other'] for group in sublist]
group_counts_other = Counter(all_groups_other)
print("Functional Group Counts in 'Other' Category:", group_counts_other)

# Combine the counts for visualization
combined_counts = group_counts_other

# Convert to DataFrame for plotting
df_group_counts_other = pd.DataFrame(combined_counts.items(), columns=['Functional Group', 'Count'])

# Plot the results for the "Other" category
plt.figure(figsize=(12, 6))
sns.barplot(x='Functional Group', y='Count', data=df_group_counts_other)
plt.title('Frequency of Functional Groups in "Other" Category Molecules')
plt.xlabel('Functional Group')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()



###################################    Perform Normality check  ###################################


# def check_normality(data):
#     if len(data) < 3:  # Shapiro-Wilk test requires at least 3 data points
#         return False
#     stat, p = shapiro(data)
#     return p > 0.05  # If p-value > 0.05, we fail to reject the null hypothesis that the data is normally distributed

# # Check normality for each functional group
# normality_results = {}
# for group in functional_groups_list:
#     fg_col_2 = f'FG_2_{group}'
#     group_data = df_merged_2[df_merged_2[fg_col_2] == 1]['Potency_Change']
#     if len(group_data) < 3:
#         normality_results[group] = False
#     else:
#         normality_results[group] = check_normality(group_data)


# # Display normality test results
# for group, is_normal in normality_results.items():
#     print(f"Normality Test for {group}: {'Normally distributed' if is_normal else 'Not normally distributed'}")




#################################### Mann-Whitney U test for all functional groups   ###################################


stat_test_results = {}
for group in functional_groups_list:
    fg_col_2 = f'FG_2_{group}'
    group_0 = df_merged_2[df_merged_2[fg_col_2] == 0]['Potency_Change']
    group_1 = df_merged_2[df_merged_2[fg_col_2] == 1]['Potency_Change']
    try:
        u_stat, p_val = mannwhitneyu(group_0, group_1)
    except ValueError:
        u_stat, p_val = np.nan, np.nan
    stat_test_results[group] = {'Test': 'Mann-Whitney U', 'Statistic': u_stat, 'p_value': p_val}

# Display Mann-Whitney U test results
for group, values in stat_test_results.items():
    print(f"{values['Test']} for {group}:")
    print(f"  U-statistic: {values['Statistic']:.2f}, p-value: {values['p_value']:.2e}")



# # Correlation Analysis -  only 1 group
# # Perform correlation analysis between the addition of specific functional groups and changes in activity metrics (potency change)
# correlation_results = {}
# for group in functional_groups_list:
#     fg_col_2 = f'FG_2_{group}'
#     corr, p_val = pearsonr(df_merged_2[fg_col_2], df_merged_2['Potency_Change'])
#     correlation_results[group] = {'Correlation': corr, 'p_value': p_val}

# # Display correlation results
# for group, values in correlation_results.items():
#     print(f"Correlation for {group}:")
#     print(f"  Correlation coefficient: {values['Correlation']:.2f}, p-value: {values['p_value']:.2e}")



output_analyzed_path = '/Users/vishnu/Desktop/viji files/sem 6/BTP/functional group analysis/analyzed_single_atom_additions_with_tests.csv'
df_merged_2.to_csv(output_analyzed_path, index=False)
print(f"Analyzed results saved to {output_analyzed_path}")


#####################################  Correlation Analysis  ###################################


# functional_groups_list = list(functional_groups.keys())

# # Add functional group presence as binary columns
# for group in functional_groups_list:
#     df_merged_2[f'FG_{group}'] = df_merged_2.apply(lambda row: 1 if group in row['Functional_Groups_1'] or group in row['Functional_Groups_2'] else 0, axis=1)

# # Correlation analysis between functional groups and potency change
# correlation_results = {}
# for group in functional_groups_list:
#     fg_col = f'FG_{group}'
#     corr, p_val = pearsonr(df_merged_2[fg_col], df_merged_2['Potency_Change'])
#     correlation_results[group] = {'Correlation': corr, 'p_value': p_val}

# # Display correlation results
# for group, values in correlation_results.items():
#     print(f"{group}:")
#     print(f"  Correlation: {values['Correlation']:.2f}, p-value: {values['p_value']:.2e}")



##################################### Clustering Analysis   #######################################
# # Normalize the data
# scaler = StandardScaler()
# df_merged_2_scaled = scaler.fit_transform(df_merged_2[['logP_1', 'logP_2', 'Potency_Change']])

# # Perform K-means clustering
# kmeans = KMeans(n_clusters=5, random_state=0)
# clusters = kmeans.fit_predict(df_merged_2_scaled)

# # Add cluster labels to the dataframe
# df_merged_2['Cluster'] = clusters

# # Visualize the clusters using a 3D scatter plot
# import plotly.express as px

# fig = px.scatter_3d(df_merged_2, x='logP_1', y='logP_2', z='Potency_Change', color='Cluster', 
#                     title='3D Scatter Plot of Clusters', opacity=0.7)
# fig.show()




