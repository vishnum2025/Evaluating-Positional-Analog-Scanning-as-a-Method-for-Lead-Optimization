import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu, shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
for group in functional_groups.keys():
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

# Additional Molecular Features
df_merged_2['MolWeight_1'] = df_merged_2['canonical_smiles_1'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))
df_merged_2['MolWeight_2'] = df_merged_2['canonical_smiles_2'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))
df_merged_2['HBondDonors_1'] = df_merged_2['canonical_smiles_1'].apply(lambda x: Descriptors.NumHDonors(Chem.MolFromSmiles(x)))
df_merged_2['HBondDonors_2'] = df_merged_2['canonical_smiles_2'].apply(lambda x: Descriptors.NumHDonors(Chem.MolFromSmiles(x)))
df_merged_2['HBondAcceptors_1'] = df_merged_2['canonical_smiles_1'].apply(lambda x: Descriptors.NumHAcceptors(Chem.MolFromSmiles(x)))
df_merged_2['HBondAcceptors_2'] = df_merged_2['canonical_smiles_2'].apply(lambda x: Descriptors.NumHAcceptors(Chem.MolFromSmiles(x)))
df_merged_2['TPSA_1'] = df_merged_2['canonical_smiles_1'].apply(lambda x: Descriptors.TPSA(Chem.MolFromSmiles(x))) #topological polar surface area
df_merged_2['TPSA_2'] = df_merged_2['canonical_smiles_2'].apply(lambda x: Descriptors.TPSA(Chem.MolFromSmiles(x)))






# Correlation Analysis for Additional Features
additional_features = ['MolWeight', 'HBondDonors', 'HBondAcceptors', 'TPSA']

correlation_results = {}
for feature in additional_features:
    feature_1 = f'{feature}_1'
    feature_2 = f'{feature}_2'
    corr_1, p_val_1 = pearsonr(df_merged_2[feature_1], df_merged_2['Potency_Change'])
    corr_2, p_val_2 = pearsonr(df_merged_2[feature_2], df_merged_2['Potency_Change'])
    correlation_results[feature] = {'Correlation_1': corr_1, 'p_value_1': p_val_1, 'Correlation_2': corr_2, 'p_value_2': p_val_2}

# Display correlation results
for feature, values in correlation_results.items():
    print(f"{feature}:")
    print(f"  Correlation with Molecule 1: {values['Correlation_1']:.2f}, p-value: {values['p_value_1']:.2e}")
    print(f"  Correlation with Molecule 2: {values['Correlation_2']:.2f}, p-value: {values['p_value_2']:.2e}")

# Normality Test
def check_normality(data):
    if len(data) < 3:
        return 'Insufficient data'
    stat, p = shapiro(data)
    return 'Normally distributed' if p > 0.05 else 'Not normally distributed'

# Check normality for each functional group
normality_results = {}
for group in functional_groups.keys():
    fg_col_2 = f'FG_2_{group}'
    group_data = df_merged_2[df_merged_2[fg_col_2] == 1]['Potency_Change']
    normality_results[group] = check_normality(group_data)

# Display normality test results
for group, result in normality_results.items():
    print(f"Normality Test for {group}: {result}")





# Mann-Whitney U Test
mann_whitney_results = {}
for group in functional_groups.keys():
    fg_col_2 = f'FG_2_{group}'
    group_data = df_merged_2[df_merged_2[fg_col_2] == 1]['Potency_Change']
    other_data = df_merged_2[df_merged_2[fg_col_2] == 0]['Potency_Change']
    
    if len(group_data) > 0 and len(other_data) > 0:
        u_stat, p_val = mannwhitneyu(group_data, other_data, alternative='two-sided')
        mann_whitney_results[group] = {'U_statistic': u_stat, 'p_value': p_val}
    else:
        mann_whitney_results[group] = {'U_statistic': np.nan, 'p_value': np.nan}

# Display Mann-Whitney U test results
for group, values in mann_whitney_results.items():
    print(f"Mann-Whitney U for {group}:")
    print(f"  U-statistic: {values['U_statistic']:.2f}, p-value: {values['p_value']:.2e}")





