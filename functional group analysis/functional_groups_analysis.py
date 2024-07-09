# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import Descriptors, rdMolDescriptors
# from rdkit.Chem import rdMolChemicalFeatures
# from rdkit.Chem import ChemicalFeatures
# from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
# from rdkit import RDConfig
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Load your dataset
# file_path = '/Users/vishnu/Desktop/viji files/sem 6/BTP/identifying additions/analyzed_single_atom_additions.csv'
# df = pd.read_csv(file_path)

# # Define functional groups as SMARTS patterns
# functional_groups = {
#     'Methyl': '[CX4H3]',
#     'Chloride': '[Cl]',
#     'Hydroxyl': '[OH]',
#     'Aromatic': 'a',
#     'Carbonyl': 'C=O',
#     'Amine': '[NH2]',
# }

# # Function to check for the presence of functional groups
# def check_functional_groups(smiles, functional_groups):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return {group: False for group in functional_groups}
#     group_presence = {}
#     for group, smarts in functional_groups.items():
#         patt = Chem.MolFromSmarts(smarts)
#         group_presence[group] = mol.HasSubstructMatch(patt)
#     return group_presence

# # Add functional group information to the dataframe
# functional_group_data = df['canonical_smiles_1'].apply(lambda smi: check_functional_groups(smi, functional_groups))
# functional_group_df = pd.DataFrame(list(functional_group_data))

# df = pd.concat([df, functional_group_df], axis=1)

# # Visualize the frequency of functional groups
# group_counts = functional_group_df.sum().sort_values(ascending=False)
# plt.figure(figsize=(12, 6))
# sns.barplot(x=group_counts.index, y=group_counts.values)
# plt.title('Frequency of Functional Groups in Molecules')
# plt.xlabel('Functional Group')
# plt.ylabel('Count')
# plt.show()


# import plotly.express as px

# # 3D Scatter plot of Potency Change vs logP and Heavy Atom Count
# fig = px.scatter_3d(
#     df, 
#     x='logP_1', 
#     y='Heavy_Atoms_1', 
#     z='Potency_Change', 
#     color='Addition_Type',
#     title='3D Scatter Plot: Potency Change vs logP and Heavy Atom Count',
#     labels={'logP_1': 'logP', 'Heavy_Atoms_1': 'Heavy Atom Count', 'Potency_Change': 'Potency Change'},
#     log_y=True,
#     log_z=True
# )
# fig.show()

# # 3D Scatter plot of Potency Change vs logP Change and HAC Change
# df['logP_Change'] = df['logP_2'] - df['logP_1']
# df['HAC_Change'] = df['Heavy_Atoms_2'] - df['Heavy_Atoms_1']

# fig = px.scatter_3d(
#     df, 
#     x='logP_Change', 
#     y='HAC_Change', 
#     z='Potency_Change', 
#     color='Addition_Type',
#     title='3D Scatter Plot: Potency Change vs logP Change and HAC Change',
#     labels={'logP_Change': 'logP Change', 'HAC_Change': 'Heavy Atom Count Change', 'Potency_Change': 'Potency Change'},
#     log_y=True,
#     log_z=True
# )
# fig.show()


# 33333

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import Counter


file_path = '/Users/vishnu/Desktop/viji files/sem 6/BTP/identifying additions/unique_single_atom_additions.csv'
activity_file_path = '/Users/vishnu/filtered_output_Ro5_logP.csv'
df_unique = pd.read_csv(file_path)
activity_df = pd.read_csv(activity_file_path, sep=",")
activity_df['molecule_chembl_id'] = activity_df['molecule_chembl_id'].str.strip()

df_merged_1 = df_unique.merge(
    activity_df[['molecule_chembl_id', 'standard_value', 'logP', 'canonical_smiles']],
    left_on='Molecule1_ID', right_on='molecule_chembl_id'
).rename(columns={
    'standard_value': 'standard_value_1',
    'logP': 'logP_1',
    'canonical_smiles': 'canonical_smiles_1'
}).drop(columns=['molecule_chembl_id'])

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

# Step 1: Define Functional Groups with correct SMARTS patterns
functional_groups = {
    'Methyl': '[CX4H3]',
    'Aromatic': 'c1ccccc1',
    'Carbonyl': 'C=O',
    'Chloride': '[Cl]',
    'Hydroxyl': '[OH]',
    'Amine': '[NH2]',
    'Ether': 'C-O-C',
    'Carboxyl': 'C(=O)[OH]',  
    'Sulfonyl': '[SX4](=O)(=O)',
    'Aldehyde': '[CX3H1](=O)',
    'Ester': 'C(=O)O',
    'Nitro': '[N+](=O)[O-]',
    'Thiol': '[SH]',
    'Imine': '[N]=[C]',
    'Amide': 'C(=O)[N]',
    'Phosphate': '[PX4](=O)(O)(O)(O)'
}

# Function to identify functional groups in molecules
def identify_functional_groups(smiles, functional_groups):
    mol = Chem.MolFromSmiles(smiles)
    found_groups = []
    if mol:
        for group_name, smarts in functional_groups.items():
            try:
                smarts_mol = Chem.MolFromSmarts(smarts)
                if smarts_mol and mol.HasSubstructMatch(smarts_mol):
                    found_groups.append(group_name)
            except Exception as e:
                print(f"Error parsing SMARTS '{smarts}' for group '{group_name}': {e}")
    return found_groups

# Identify functional groups in the molecules
df_merged_2['Functional_Groups_1'] = df_merged_2['canonical_smiles_1'].apply(identify_functional_groups, args=(functional_groups,))
df_merged_2['Functional_Groups_2'] = df_merged_2['canonical_smiles_2'].apply(identify_functional_groups, args=(functional_groups,))

# Flatten the lists and count the occurrences
all_groups_1 = [group for sublist in df_merged_2['Functional_Groups_1'] for group in sublist]
all_groups_2 = [group for sublist in df_merged_2['Functional_Groups_2'] for group in sublist]

group_counts_1 = Counter(all_groups_1)
group_counts_2 = Counter(all_groups_2)

print("Functional Group Counts in Molecule 1:", group_counts_1)
print("Functional Group Counts in Molecule 2:", group_counts_2)

# Combine the counts for visualization
combined_counts = group_counts_1 + group_counts_2

# Convert to DataFrame for plotting
df_group_counts = pd.DataFrame(combined_counts.items(), columns=['Functional Group', 'Count'])

# Plot the results
plt.figure(figsize=(12, 6))
sns.barplot(x='Functional Group', y='Count', data=df_group_counts)
plt.title('Frequency of Functional Groups in Molecules')
plt.xlabel('Functional Group')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# Step 3: Analyze and Visualize the Results
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

# Step 4: Update Potency Change Analysis
# Plotting potency change by extended addition type
plt.figure(figsize=(12, 6))
sns.boxplot(x='Addition_Type_Extended', y='Potency_Change', data=df_merged_2)
plt.title('Potency Change by Extended Addition Type')
plt.xlabel('Addition Type')
plt.ylabel('Potency Change')
plt.yscale('log')
plt.xticks(rotation=90)
plt.show()

