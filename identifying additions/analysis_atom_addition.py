import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


file_path = '/Users/vishnu/Desktop/viji files/sem 6/BTP/identifying additions/unique_single_atom_additions.csv'
df_unique = pd.read_csv(file_path)


activity_file_path = '/Users/vishnu/filtered_output_Ro5_logP.csv'
activity_df = pd.read_csv(activity_file_path, sep=",")
print(activity_df.columns)
print(activity_df["logP"])
activity_df['molecule_chembl_id'] = activity_df['molecule_chembl_id'].str.strip()
# # Calculate logP for each molecule
# def calculate_logP(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     return Descriptors.MolLogP(mol) if mol else None

# merging to get activity data for the first molecule
df_merged_1 = df_unique.merge(
    activity_df[['molecule_chembl_id', 'standard_value', 'logP', 'canonical_smiles']],
    left_on='Molecule1_ID', right_on='molecule_chembl_id'
).rename(columns={
    'standard_value': 'standard_value_1',
    'logP': 'logP_1',
    'canonical_smiles': 'canonical_smiles_1'
}).drop(columns=['molecule_chembl_id'])

# merging to get activity data for the second molecule
df_merged_2 = df_merged_1.merge(
    activity_df[['molecule_chembl_id', 'standard_value', 'logP', 'canonical_smiles']],
    left_on='Molecule2_ID', right_on='molecule_chembl_id'
).rename(columns={
    'standard_value': 'standard_value_2',
    'logP': 'logP_2',
    'canonical_smiles': 'canonical_smiles_2'
}).drop(columns=['molecule_chembl_id'])


df_merged_2 = df_merged_2.loc[:, ~df_merged_2.columns.duplicated()]


print(df_merged_2.dtypes)
print(df_merged_2[['canonical_smiles_1', 'canonical_smiles_2']].head())

# ensuring 'canonical_smiles_1' and 'canonical_smiles_2' are of string type
df_merged_2['canonical_smiles_1'] = df_merged_2['canonical_smiles_1'].astype(str)
df_merged_2['canonical_smiles_2'] = df_merged_2['canonical_smiles_2'].astype(str)

print("done 2")

# potency change
df_merged_2['Potency_Change'] = df_merged_2['standard_value_2'] - df_merged_2['standard_value_1']

df_merged_2 = df_merged_2[df_merged_2['Potency_Change'].abs() < 1e6]

print("Summary of Potency_Change:\n", df_merged_2['Potency_Change'].describe())
print("Rows with Potency_Change > 1000 or Potency_Change < -1000:\n", df_merged_2[(df_merged_2['Potency_Change'] > 1000) | (df_merged_2['Potency_Change'] < -1000)])

# # Visualize the distribution of Potency Change
# plt.figure(figsize=(12, 6))
# sns.histplot(df_merged_2['Potency_Change'], bins=50, kde=True)
# plt.title('Distribution of Potency Change')
# plt.xlabel('Potency Change')
# plt.ylabel('Frequency')
# plt.show()

# Function to count heavy atoms in a molecule
def count_heavy_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.HeavyAtomCount(mol) if mol else None

# Calculate heavy atom count for the original molecule
df_merged_2['Heavy_Atoms_1'] = df_merged_2['canonical_smiles_1'].apply(count_heavy_atoms)

# Analysis 1: Effects of different additions
def identify_addition_type(smiles_2, smiles_1):
    mol1 = Chem.MolFromSmiles(smiles_1)
    mol2 = Chem.MolFromSmiles(smiles_2)
    if not mol1 or not mol2:
        return 'Unknown'
    if mol2.HasSubstructMatch(Chem.MolFromSmarts('[CH3]')) and not mol1.HasSubstructMatch(Chem.MolFromSmarts('[CH3]')):
        return 'Methyl'
    if mol2.HasSubstructMatch(Chem.MolFromSmarts('[Cl]')) and not mol1.HasSubstructMatch(Chem.MolFromSmarts('[Cl]')):
        return 'Chloride'
    if mol2.HasSubstructMatch(Chem.MolFromSmarts('[OH]')) and not mol1.HasSubstructMatch(Chem.MolFromSmarts('[OH]')):
        return 'Hydroxyl'
    return 'Other'

df_merged_2['Addition_Type'] = df_merged_2.apply(
    lambda row: identify_addition_type(row['canonical_smiles_2'], row['canonical_smiles_1']), axis=1
)

# Plotting potency change by addition type
plt.figure(figsize=(12, 6))
sns.boxplot(x='Addition_Type', y='Potency_Change', data=df_merged_2)
plt.title('Potency Change by Addition Type')
plt.xlabel('Addition Type')
plt.ylabel('Potency Change')
plt.yscale('log')  
plt.show()

# Analysis 2: Dependence on the size of the original molecule
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Heavy_Atoms_1', y='Potency_Change', data=df_merged_2)
plt.title('Potency Change vs. Number of Heavy Atoms in Original Molecule')
plt.xlabel('Number of Heavy Atoms in Original Molecule')
plt.ylabel('Potency Change')
plt.yscale('log')  
plt.show()

# Analysis 3: Correlation with logP
plt.figure(figsize=(12, 6))
sns.scatterplot(x='logP_1', y='Potency_Change', data=df_merged_2)
plt.title('Potency Change vs. logP of Original Molecule')
plt.xlabel('logP of Original Molecule')
plt.ylabel('Potency Change')
plt.yscale('log') 
plt.show()


# Save the merged DataFrame for further analysis if needed
output_analyzed_path = '/Users/vishnu/Desktop/viji files/sem 6/BTP/identifying additions/analyzed_single_atom_additions.csv'
df_merged_2.to_csv(output_analyzed_path, index=False)
print(f"Analyzed results saved to {output_analyzed_path}")



# Check correlation
logP_correlation, logP_p_value = pearsonr(df_merged_2['logP_1'], df_merged_2['Potency_Change'])
hac_correlation, hac_p_value = pearsonr(df_merged_2['Heavy_Atoms_1'], df_merged_2['Potency_Change'])

print(f"Pearson correlation between logP and Potency Change: {logP_correlation:.2f}, p-value: {logP_p_value:.2e}")
print(f"Pearson correlation between HAC and Potency Change: {hac_correlation:.2f}, p-value: {hac_p_value:.2e}")
# Plotting Hexbin plots
plt.figure(figsize=(12, 6))
plt.hexbin(df_merged_2['logP_1'], df_merged_2['Potency_Change'], gridsize=50, cmap='Blues', mincnt=1)
plt.colorbar(label='Count')
plt.xlabel('logP of Original Molecule')
plt.ylabel('Potency Change')
plt.yscale('log')
plt.title('Hexbin Plot: Potency Change vs. logP of Original Molecule')
plt.show()

plt.figure(figsize=(12, 6))
plt.hexbin(df_merged_2['Heavy_Atoms_1'], df_merged_2['Potency_Change'], gridsize=50, cmap='Blues', mincnt=1)
plt.colorbar(label='Count')
plt.xlabel('Number of Heavy Atoms in Original Molecule')
plt.ylabel('Potency Change')
plt.yscale('log')
plt.title('Hexbin Plot: Potency Change vs. Number of Heavy Atoms in Original Molecule')
plt.show()

# Pair plots to visualize relationships
sns.pairplot(df_merged_2[['logP_1', 'Heavy_Atoms_1', 'Potency_Change']], plot_kws={'alpha': 0.1})
plt.show()



