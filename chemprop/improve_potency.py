import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import rdMolDescriptors


# file_path = '/Users/vishnu/Desktop/viji_files/sem6/BTP/identifying additions/unique_single_atom_additions.csv'
# activity_file_path = '/Users/vishnu/filtered_output_Ro5_logP.csv'
# df_unique = pd.read_csv(file_path)
# activity_df = pd.read_csv(activity_file_path, sep=",")
# activity_df['molecule_chembl_id'] = activity_df['molecule_chembl_id'].str.strip()

# df_merged_1 = df_unique.merge(
#     activity_df[['molecule_chembl_id', 'standard_value', 'logP', 'canonical_smiles']],
#     left_on='Molecule1_ID', right_on='molecule_chembl_id'
# ).rename(columns={
#     'standard_value': 'standard_value_1',
#     'logP': 'logP_1',
#     'canonical_smiles': 'canonical_smiles_1'
# }).drop(columns=['molecule_chembl_id'])

# df_merged_2 = df_merged_1.merge(
#     activity_df[['molecule_chembl_id', 'standard_value', 'logP', 'canonical_smiles']],
#     left_on='Molecule2_ID', right_on='molecule_chembl_id'
# ).rename(columns={
#     'standard_value': 'standard_value_2',
#     'logP': 'logP_2',
#     'canonical_smiles': 'canonical_smiles_2'
# }).drop(columns=['molecule_chembl_id'])

# df_merged_2 = df_merged_2.loc[:, ~df_merged_2.columns.duplicated()]

# # Ensure canonical_smiles columns are string type
# df_merged_2['canonical_smiles_1'] = df_merged_2['canonical_smiles_1'].astype(str)
# df_merged_2['canonical_smiles_2'] = df_merged_2['canonical_smiles_2'].astype(str)

# # Calculate potency change
# df_merged_2['Potency_Change'] = df_merged_2['standard_value_2'] - df_merged_2['standard_value_1']
# df_merged_2 = df_merged_2[df_merged_2['Potency_Change'].abs() < 1e6]

# df_merged_2.to_csv("/Users/vishnu/Desktop/viji_files/sem7/BTP_2/df_merged_with_potency_change.csv", index=False)



#####################################################################################################


# file_path = '/Users/vishnu/Desktop/viji_files/sem7/BTP_2/df_merged_with_potency_change.csv'
# df_potency = pd.read_csv(file_path)
# print(df_potency.head())
# print(len(df_potency))
# print(df_potency.columns)
# print(df_potency["Common_Scaffold"])

# chemprop_df = df_potency[['canonical_smiles_1', 'canonical_smiles_2', 'Potency_Change']]
# print(chemprop_df.head())
# chemprop_df.to_csv("/Users/vishnu/Desktop/viji_files/sem7/BTP_2/chemprop_data.csv", index=False)


#####################################################################################################

chemprop_path = "/Users/vishnu/Desktop/viji_files/sem7/BTP_2/chemprop_data.csv"
df_chemprop = pd.read_csv(chemprop_path)



from sklearn.model_selection import train_test_split


X = df_chemprop[['canonical_smiles_1', 'canonical_smiles_2']]  # Features
y = df_chemprop['Potency_Change']  # Target variable

# Combine features and target back into a single DataFrame for easy saving
data = pd.concat([X, y], axis=1)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the training and testing data into separate CSV files
train_data.to_csv('/Users/vishnu/Desktop/viji_files/sem7/BTP_2/train_data_chemprop.csv', index=False)
test_data.to_csv('/Users/vishnu/Desktop/viji_files/sem7/BTP_2/test_data_chemprop.csv', index=False)

print("Training and testing data saved")



data = pd.read_csv('/Users/vishnu/Desktop/viji_files/sem7/BTP_2/chemprop_data_fingerprintss.csv')
print(data.head())
print(data.columns)
print(data.shape)

