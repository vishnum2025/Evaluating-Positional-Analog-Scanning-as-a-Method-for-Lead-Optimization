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

 

csv_file_path = '/Users/vishnu/combined.csv'

# Load the dataset
df_pairs = pd.read_csv(csv_file_path, header=None, names=['AssayChEMBLID', 'Molecule1_ID', 'Molecule2_ID', 'Distance'])
df_filtered = df_pairs[(df_pairs['Distance'] == 0) | (df_pairs['Distance'] == 1)].copy()
#print(df_filtered.head())
df_smiles = pd.read_csv("/Users/vishnu/output.csv", sep="\t", low_memory=False)

df_smiles['molecule_chembl_id'] = df_smiles['molecule_chembl_id'].str.strip()
df_smiles['canonical_smiles'] = df_smiles['canonical_smiles'].str.strip()

# Creating a dictionary to map chembl ids and their respective smiles, instead of iterating through the csv everytime
chembl_smiles_dict = pd.Series(df_smiles.canonical_smiles.values, index=df_smiles.molecule_chembl_id).to_dict()


def fetch_smiles_from_chembl(chembl_id):
    return chembl_smiles_dict.get(chembl_id.strip(), "SMILES Not Found")



def get_atom_map(mol):
    """Returns a dictionary mapping atom indices to atom symbols."""
    return {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}

def get_bond_map(mol):
    """Returns a set of tuples representing bonds (by atom indices)."""
    return {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in mol.GetBonds()}


def identify_stereocenters(mol):
    """Identify stereocenters in the molecule."""
    stereocenters = []
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    for atom in mol.GetAtoms():
        if atom.HasProp('_ChiralityPossible'):
            stereocenters.append(atom.GetIdx())
    return stereocenters

def check_stereocenter_differences(mol1, mol2):
    """Check for stereocenter differences in two molecules."""
    stereocenters_mol1 = set(identify_stereocenters(mol1))
    stereocenters_mol2 = set(identify_stereocenters(mol2))
    return stereocenters_mol1 != stereocenters_mol2

def get_functional_groups(mol):
    """Identify functional groups in the molecule."""
    # Define SMARTS for functional groups of interest
    functional_groups = {
        'carboxylic_acid': '[CX3](=O)[OX2H1]',
        'amine': '[NX3;H2,H1;!$(NC=O)]',
        'ketone': '[#6][CX3](=O)[#6]',
        'aldehyde': '[CX3H1](=O)[#6]',
        'alcohol': '[OX2H]',
        'ether': '[#6]-O-[#6]',
        'alkene': 'C=C',
        'alkyne': 'C#C',
        'nitrile': '[#6]C#N',
        'nitro': '[NX3](=O)=O',
        'sulfonamide': 'S(=O)(=O)N',
        'phosphate': '[PX4](=O)([OX2H][#6])[OX2][OX2][#6]',
        'halide': '[F,Cl,Br,I]',
    }
    groups_found = {}
    for name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        count = len(mol.GetSubstructMatches(pattern))
        if count > 0:
            groups_found[name] = count
    return groups_found

def check_functional_group_changes(mol1, mol2):
    """Check for functional group changes between two molecules."""
    groups_mol1 = get_functional_groups(mol1)
    groups_mol2 = get_functional_groups(mol2)
    return groups_mol1 != groups_mol2

def check_ring_changes(mol1, mol2):
    """Check for ring opening or closure changes between two molecules."""
    rings_mol1 = mol1.GetRingInfo().NumRings()
    rings_mol2 = mol2.GetRingInfo().NumRings()
    return rings_mol1 != rings_mol2



def characterize_modifications(args):
    row, chembl_smiles_dict = args
    mol1_smiles = fetch_smiles_from_chembl(row['Molecule1_ID'])
    mol2_smiles = fetch_smiles_from_chembl(row['Molecule2_ID'])


    # If SMILES not found, return a corresponding error message
    if mol1_smiles == "SMILES Not Found" or mol2_smiles == "SMILES Not Found":
        print("Error: One or both SMILES strings not found.")
        return "Error: One or both SMILES strings not found."
        

    mol1 = Chem.MolFromSmiles(mol1_smiles) if mol1_smiles else None
    mol2 = Chem.MolFromSmiles(mol2_smiles) if mol2_smiles else None
    
    if not mol1 or not mol2:
        return "Error: Invalid SMILES string."

    modifications = []

    # Finding Maximum Common Substructure (MCS)
    mcs_result = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)

    # Atom maps for comparison
    atom_map_mol1 = get_atom_map(mol1)
    atom_map_mol2 = get_atom_map(mol2)

    # Bond maps for comparison
    bond_map_mol1 = get_bond_map(mol1)
    bond_map_mol2 = get_bond_map(mol2)

    # MCS atom and bond maps
    mcs_mol_match1 = mol1.GetSubstructMatch(mcs_mol)
    mcs_mol_match2 = mol2.GetSubstructMatch(mcs_mol)
    mcs_bond_map = get_bond_map(mcs_mol)

    # Identifying specific atoms present in one molecule and not the other
    unique_atoms_mol1 = set(atom_map_mol1.keys()) - set(mcs_mol_match1)
    unique_atoms_mol2 = set(atom_map_mol2.keys()) - set(mcs_mol_match2)
    if unique_atoms_mol1 or unique_atoms_mol2:
        modifications.append(f"Unique atoms - Mol1: {[atom_map_mol1[idx] for idx in unique_atoms_mol1]}, Mol2: {[atom_map_mol2[idx] for idx in unique_atoms_mol2]}")

    # Identifying specific bonds present in one molecule and not the other
    unique_bonds_mol1 = bond_map_mol1 - mcs_bond_map
    unique_bonds_mol2 = bond_map_mol2 - mcs_bond_map
    if unique_bonds_mol1 or unique_bonds_mol2:
        modifications.append(f"Unique bonds - Mol1: {len(unique_bonds_mol1)}, Mol2: {len(unique_bonds_mol2)}")

    # Stereocenter differences
    if check_stereocenter_differences(mol1, mol2):
        modifications.append("Stereocenter differences detected")

    # Functional group changes
    if check_functional_group_changes(mol1, mol2):
        modifications.append("Functional group changes detected")
    
    # Ring changes
    if check_ring_changes(mol1, mol2):
        modifications.append("Ring opening/closure detected")

    return '; '.join(modifications) if modifications else "No significant differences identified"


        
# Preparing the arguments for multiprocessing
args = [(row, chembl_smiles_dict) for index, row in df_filtered.iterrows()]
    
# Using multiprocessing to process the DataFrame
with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(characterize_modifications, args), total=len(args)))

# Storing results back into the DataFrame
df_filtered['Modifications'] = results


# Saving the results to a CSV file
output_csv_path = '/Users/vishnu/characterized_modifications.csv'
df_filtered.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")


