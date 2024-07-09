import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem
from rdkit.Chem import rdMolTransforms
from rdkit import DataStructs
import numpy as np
import os
from tqdm import tqdm
import requests
from multiprocessing import Pool, cpu_count




def get_stereocenter_info(mol):
    """Extracts stereocenters and their configurations from a molecule."""
    stereocenters = []
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    for atom in mol.GetAtoms():
        if atom.HasProp('_ChiralityPossible'):
            cip_code = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else 'unassigned'
            stereocenters.append((atom.GetIdx(), cip_code))
    return stereocenters

def compare_stereocenters(centers1, centers2):
    """Compares stereocenters between two molecules for inversion."""
    if len(centers1) != len(centers2):
        return False  # Different number of stereocenters
    inversion_detected = False
    for center1, center2 in zip(centers1, centers2):
        idx1, cip1 = center1
        idx2, cip2 = center2
        if idx1 == idx2 and cip1 != cip2:
            inversion_detected = True
            break
    return inversion_detected

def check_inversion_of_stereocenters(mol1, mol2):
    """Checks for inversion of stereocenters between two molecules."""
    centers1 = get_stereocenter_info(mol1)
    centers2 = get_stereocenter_info(mol2)
    return compare_stereocenters(centers1, centers2)


def process_batch(batch_args):
    chembl_smiles_dict, batch = batch_args
    results = []
    for index, row in enumerate(batch.itertuples(index=False)):
        print(f"Processing row {index+1}/{len(batch)}")  # Debug print
        result = characterize_modifications((row, chembl_smiles_dict))
        results.append(result)
    return results





def characterize_modifications(args):
    row, chembl_smiles_dict = args
    mol1_smiles = chembl_smiles_dict.get(row.Molecule1_ID.strip(), "SMILES Not Found")
    mol2_smiles = chembl_smiles_dict.get(row.Molecule2_ID.strip(), "SMILES Not Found")
    
    if mol1_smiles == "SMILES Not Found" or mol2_smiles == "SMILES Not Found":
        return "SMILES not found for one or both molecules"
    
    mol1 = Chem.MolFromSmiles(mol1_smiles)
    mol2 = Chem.MolFromSmiles(mol2_smiles)
    
    modifications_detected = []
    
    if check_inversion_of_stereocenters(mol1, mol2):
        modifications_detected.append("Inversion of stereocenters detected")
    
    return '; '.join(modifications_detected) if modifications_detected else "No significant modifications detected"


def create_batches(df, batch_size):
    total_rows = len(df)
    return [df[i:i+batch_size] for i in range(0, total_rows, batch_size)]





csv_file_path = '/Users/vishnu/combined.csv'
df_pairs = pd.read_csv(csv_file_path, header=None, names=['AssayChEMBLID', 'Molecule1_ID', 'Molecule2_ID', 'Distance'])
df_filtered = df_pairs[(df_pairs['Distance'] == 0) | (df_pairs['Distance'] == 1)].copy()
df_smiles = pd.read_csv("/Users/vishnu/output.csv", sep="\t", low_memory=False)
df_smiles['molecule_chembl_id'] = df_smiles['molecule_chembl_id'].str.strip()
df_smiles['canonical_smiles'] = df_smiles['canonical_smiles'].str.strip()
chembl_smiles_dict = pd.Series(df_smiles.canonical_smiles.values, index=df_smiles.molecule_chembl_id).to_dict()



# Use a small subset for debugging
debug_batch = df_filtered.head(855697)
debug_results = process_batch((chembl_smiles_dict, debug_batch))
print(debug_results)
# debug_results_df = pd.DataFrame(debug_results)
# df_filtered.update(debug_results_df)


# batch_size = 100  # Adjust based on your system's capabilities
# batches = create_batches(df_filtered, batch_size)
# batch_args = [(chembl_smiles_dict, batch) for batch in batches]
# Process batches with multiprocessing
# with Pool(cpu_count()) as pool:
#     batch_results = list(tqdm(pool.imap(process_batch, batch_args), total=len(batch_args)))
# # Flatten list of lists
# results = [item for sublist in batch_results for item in sublist]
# # Storing results back into the DataFrame


df_filtered['Modifications'] = debug_results



# Save the results to a CSV file
output_csv_path = '/Users/vishnu/characterized_modifications.csv'

df_filtered.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")


