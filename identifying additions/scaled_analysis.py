import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

file_path = '/Users/vishnu/Desktop/viji files/sem 6/BTP/identifying additions/unique_single_atom_additions.csv'
df_unique = pd.read_csv(file_path)

activity_file_path = '/Users/vishnu/filtered_output_Ro5_logP.csv'
activity_df = pd.read_csv(activity_file_path, sep=",")
activity_df['molecule_chembl_id'] = activity_df['molecule_chembl_id'].str.strip()

# Merging 1
df_merged_1 = df_unique.merge(
    activity_df[['molecule_chembl_id', 'standard_value', 'logP', 'canonical_smiles']],
    left_on='Molecule1_ID', right_on='molecule_chembl_id'
).rename(columns={
    'standard_value': 'standard_value_1',
    'logP': 'logP_1',
    'canonical_smiles': 'canonical_smiles_1'
}).drop(columns=['molecule_chembl_id'])

# Merging 2
df_merged_2 = df_merged_1.merge(
    activity_df[['molecule_chembl_id', 'standard_value', 'logP', 'canonical_smiles']],
    left_on='Molecule2_ID', right_on='molecule_chembl_id'
).rename(columns={
    'standard_value': 'standard_value_2',
    'logP': 'logP_2',
    'canonical_smiles': 'canonical_smiles_2'
}).drop(columns=['molecule_chembl_id'])

df_merged_2 = df_merged_2.loc[:, ~df_merged_2.columns.duplicated()]


df_merged_2['canonical_smiles_1'] = df_merged_2['canonical_smiles_1'].astype(str)
df_merged_2['canonical_smiles_2'] = df_merged_2['canonical_smiles_2'].astype(str)

# potency change
df_merged_2['Potency_Change'] = df_merged_2['standard_value_2'] - df_merged_2['standard_value_1']
df_merged_2 = df_merged_2[df_merged_2['Potency_Change'].abs() < 1e6]

# Calculating heavy atom count for the original molecule
def count_heavy_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.HeavyAtomCount(mol) if mol else None

df_merged_2['Heavy_Atoms_1'] = df_merged_2['canonical_smiles_1'].apply(count_heavy_atoms)
df_merged_2['Heavy_Atoms_2'] = df_merged_2['canonical_smiles_2'].apply(count_heavy_atoms)

# Calculating change in logP and heavy atom count
df_merged_2['logP_Change'] = df_merged_2['logP_2'] - df_merged_2['logP_1']
df_merged_2['HAC_Change'] = df_merged_2['Heavy_Atoms_2'] - df_merged_2['Heavy_Atoms_1']

# Analysis: Distribution of Potency Change for Different logP Bins
df_merged_2['logP_Bin'] = pd.cut(df_merged_2['logP_1'], bins=np.arange(-5, 6, 1))
plt.figure(figsize=(12, 6))
sns.boxplot(x='logP_Bin', y='Potency_Change', data=df_merged_2)
plt.title('Potency Change by logP Bins')
plt.xlabel('logP Bin')
plt.ylabel('Potency Change')
plt.yscale('log')  
plt.show()

# Analysis: Distribution of Potency Change for Different HAC Bins
df_merged_2['HAC_Bin'] = pd.cut(df_merged_2['Heavy_Atoms_1'], bins=np.arange(0, 40, 5))
plt.figure(figsize=(12, 6))
sns.boxplot(x='HAC_Bin', y='Potency_Change', data=df_merged_2)
plt.title('Potency Change by HAC Bins')
plt.xlabel('Heavy Atom Count Bin')
plt.ylabel('Potency Change')
plt.yscale('log')  
plt.show()

# Analysis: Correlation between Change in logP and Potency Change
logP_corr, logP_pval = pearsonr(df_merged_2['logP_Change'], df_merged_2['Potency_Change'])
print(f"Pearson correlation between logP Change and Potency Change: {logP_corr:.2f}, p-value: {logP_pval:.2e}")

plt.figure(figsize=(12, 6))
sns.scatterplot(x='logP_Change', y='Potency_Change', data=df_merged_2)
plt.title('Potency Change vs. logP Change')
plt.xlabel('logP Change')
plt.ylabel('Potency Change')
plt.yscale('log')  
plt.show()

# Analysis: Correlation between Change in HAC and Potency Change
hac_corr, hac_pval = pearsonr(df_merged_2['HAC_Change'], df_merged_2['Potency_Change'])
print(f"Pearson correlation between HAC Change and Potency Change: {hac_corr:.2f}, p-value: {hac_pval:.2e}")

plt.figure(figsize=(12, 6))
sns.scatterplot(x='HAC_Change', y='Potency_Change', data=df_merged_2)
plt.title('Potency Change vs. HAC Change')
plt.xlabel('HAC Change')
plt.ylabel('Potency Change')
plt.yscale('log')  
plt.show()
