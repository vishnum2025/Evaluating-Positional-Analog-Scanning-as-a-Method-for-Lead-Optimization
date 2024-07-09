from collections import Counter
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem
from rdkit import DataStructs
from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm


class ChemicalAnalysis:
    @staticmethod
    def censurize(smi):
        # Simplify SMILES by replacing heavy atoms with generic representations
        return ''.join(['A' if c.isalpha() and c.isupper() else 'a' if c.isalpha() else c for c in smi if c not in "=/@-"])

    @staticmethod
    def bond_diff(mol1, mol2):
        # Compute bond difference using Maximum Common Substructure (MCS)
        mcs = rdFMCS.FindMCS([mol1, mol2], bondCompare=rdFMCS.BondCompare.CompareAny)
        return sum(len(mol.GetBonds()) for mol in [mol1, mol2]) - 2 * mcs.numBonds

    @staticmethod
    def atom_diff(mol1, mol2):
        # Compute atom difference between two molecules
        formula_diff = np.zeros(100)  # Assuming a maximum atomic number of 99
        for mol in [mol1, mol2]:
            for atom in mol.GetAtoms():
                formula_diff[atom.GetAtomicNum()] += 1 if mol == mol1 else -1
        return int(np.abs(formula_diff).sum() / 2)


def bulk_tanimoto_similarity(fp_list):
    """Calculates the Tanimoto similarity matrix for a list of fingerprints."""
    n = len(fp_list)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            similarity_matrix[i, j] = DataStructs.FingerprintSimilarity(fp_list[i], fp_list[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]  # Symmetric matrix
    return similarity_matrix



def find_analogs(args):
    df, assay_chembl_ids, settings = args
    progress_bar = tqdm(assay_chembl_ids, desc="Processing Assays", leave=False)
    for assay_chembl_id in progress_bar:
        progress_bar.set_description(f"Processing {assay_chembl_id}")
        subset = df[df['assay_chembl_id'] == assay_chembl_id].copy()

        # Generate molecular data
        subset["mols"] = subset["canonical_smiles"].apply(Chem.MolFromSmiles)
        subset['frameworks'] = subset['canonical_smiles'].apply(lambda smi: Chem.MolFromSmarts(ChemicalAnalysis.censurize(smi)))
        subset["Fingerprint"] = subset["mols"].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

        # Calculate bulk Tanimoto similarities
        fp_list = subset["Fingerprint"].tolist()
        similarity_matrix = bulk_tanimoto_similarity(fp_list)

        # Process the similarity matrix and apply the cutoffs
        results = []
        n = len(subset)
        for i in range(n):
            for j in range(i+1, n):
                if similarity_matrix[i, j] >= settings['tc_cutoff']:
                    adist = ChemicalAnalysis.atom_diff(subset["mols"].iloc[i], subset["mols"].iloc[j])
                    if adist <= settings['a_cutoff']:
                        bdist = ChemicalAnalysis.bond_diff(subset["frameworks"].iloc[i], subset["frameworks"].iloc[j])
                        if adist + bdist <= settings['d_cutoff']:
                            results.append(f"{assay_chembl_id}, {subset['molecule_chembl_id'].iloc[i]}, {subset['molecule_chembl_id'].iloc[j]}, {adist+bdist}\n")

        output_path = f"output/{assay_chembl_id}_proposed_pairs.csv"
        with open(output_path, "w") as f:
            if results:
                f.writelines(results)
            else:
                f.write(f"{assay_chembl_id}, None, None, 100\n")

if __name__ == '__main__':
    directory = 'output/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                text = f.read()
            if len(text) == 0:
                os.remove(filepath)
                print(f"Deleted empty file: {filename}")

    completed = {file.split("_")[0] for file in os.listdir(directory) if file.endswith('.csv')}

    df = pd.read_csv("/Users/vishnu/filtered_output_Ro5.csv", sep="\t") 


    settings = {'tc_cutoff': 0.6, 'a_cutoff': 3, 'd_cutoff': 4}
    assays_to_process = df['assay_chembl_id'].unique()

    # Create chunks of assays to distribute the workload evenly across processes
    n_processes = cpu_count()
    assay_chunks = np.array_split(assays_to_process, n_processes)

    # Prepare inputs for each process
    inputs = [(df, chunk.tolist(), settings) for chunk in assay_chunks]


    with Pool(n_processes) as pool:
        list(tqdm(pool.imap_unordered(find_analogs, inputs), total=len(inputs), desc="Overall Progress"))

# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import rdFMCS, AllChem
# from rdkit import DataStructs
# from multiprocessing import Pool, cpu_count
# import os
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# class ChemicalAnalysis:
#     @staticmethod
#     def censurize(smi):
#         return ''.join(['A' if c.isalpha() and c.isupper() else 'a' if c.isalpha() else c for c in smi if c not in "=/@-"])

#     @staticmethod
#     def bond_diff(mol1, mol2):
#         mcs = rdFMCS.FindMCS([mol1, mol2], bondCompare=rdFMCS.BondCompare.CompareAny)
#         return sum(len(mol.GetBonds()) for mol in [mol1, mol2]) - 2 * mcs.numBonds

#     @staticmethod
#     def atom_diff(mol1, mol2):
#         formula_diff = np.zeros(100)
#         for mol in [mol1, mol2]:
#             for atom in mol.GetAtoms():
#                 formula_diff[atom.GetAtomicNum()] += 1 if mol == mol1 else -1
#         return int(np.abs(formula_diff).sum() / 2)

# def bulk_tanimoto_similarity(fp_list):
#     n = len(fp_list)
#     similarity_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i+1, n):
#             similarity_matrix[i, j] = DataStructs.FingerprintSimilarity(fp_list[i], fp_list[j])
#             similarity_matrix[j, i] = similarity_matrix[i, j]
#     return similarity_matrix

# def find_analogs(args):
#     df, assay_chembl_ids, settings = args
#     all_results = []
#     progress_bar = tqdm(assay_chembl_ids, desc="Processing Assays", leave=False)
#     for assay_chembl_id in progress_bar:
#         progress_bar.set_description(f"Processing {assay_chembl_id}")
#         subset = df[df['assay_chembl_id'] == assay_chembl_id].copy()

#         # Generate molecular data
#         subset["mols"] = subset["canonical_smiles"].apply(Chem.MolFromSmiles)
#         subset["Fingerprint"] = subset["mols"].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

#         # Calculate bulk Tanimoto similarities
#         fp_list = subset["Fingerprint"].tolist()
#         similarity_matrix = bulk_tanimoto_similarity(fp_list)

#         # Process the similarity matrix and apply the cutoffs
#         n = len(subset)
#         for i in range(n):
#             for j in range(i+1, n):
#                 tanimoto_sim = similarity_matrix[i, j]
#                 if tanimoto_sim >= settings['tc_cutoff']:
#                     adist = ChemicalAnalysis.atom_diff(subset["mols"].iloc[i], subset["mols"].iloc[j])
#                     if adist <= settings['a_cutoff']:
#                         results = {
#                             'AssayID': assay_chembl_id,
#                             'Molecule1_ID': subset['molecule_chembl_id'].iloc[i],
#                             'Molecule2_ID': subset['molecule_chembl_id'].iloc[j],
#                             'AtomDifference': adist,
#                             'TanimotoSimilarity': tanimoto_sim
#                         }
#                         all_results.append(results)
#     return all_results


# if __name__ == '__main__':
#     df = pd.read_csv("/Users/vishnu/filtered_output_Ro5.csv", sep="\t").head(10000)
#     thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
#     assay_to_process = df['assay_chembl_id'].unique()
    
#     n_processes = cpu_count()
#     atom_difference_bins = range(0, 5)  # Example bin range for atom differences
    
#     results_by_threshold = {}
#     for tc in thresholds:
#         settings = {'tc_cutoff': tc, 'a_cutoff': 3, 'd_cutoff': 4}
#         inputs = [(df, assay_to_process, settings)]
#         with Pool(n_processes) as pool:  
#             results = pool.map(find_analogs, inputs)
#         results_by_threshold[tc] = results[0]

#     # Create a DataFrame to store bin counts for each threshold
#     bin_counts_by_threshold = pd.DataFrame(index=atom_difference_bins, columns=thresholds)
    
#     for tc in thresholds:
#         # Assume 'data' is a list of dictionaries with 'AtomDifference' as one of the keys
#         atom_diffs = [item['AtomDifference'] for item in results_by_threshold[tc]]
#         bin_counts = pd.cut(atom_diffs, bins=atom_difference_bins, labels=atom_difference_bins[:-1], include_lowest=True).value_counts()
#         bin_counts_by_threshold[tc] = bin_counts

#     bin_counts_by_threshold.plot(kind='bar')
#     plt.xlabel('Atom Difference Bins')
#     plt.ylabel('Count')
#     plt.title('Number of Molecule Pairs by Atom Difference and Tanimoto Threshold')
#     plt.legend(title='Tanimoto Threshold')
#     plt.show()


