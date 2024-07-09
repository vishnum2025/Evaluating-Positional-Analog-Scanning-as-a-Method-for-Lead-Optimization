import pandas as pd
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.DataStructs import BulkTanimotoSimilarity
import numpy as np
from collections import Counter
from multiprocessing.pool import Pool
import os


def bond_diff(framework1, framework2):
    """
    Calculate the difference in the number of bonds between two molecular frameworks.

    Parameters:
    framework1 (Chem.Mol): The first molecular framework.
    framework2 (Chem.Mol): The second molecular framework.

    Returns:
    int: The difference in the number of bonds between the two frameworks.

    The bond_diff function calculates the difference in the number of bonds
    between two molecular frameworks. It first identifies the Maximum Common Substructure (MCS)
    between the two frameworks using the FindMCS function. Then, it computes the difference
    in the number of bonds by subtracting twice the number of bonds in the MCS from
    the total number of bonds in each framework.
    """
    frameworks = (framework1, framework2)
    mcs = FindMCS(frameworks)
    changes = -2 * mcs.numBonds
    for framework in frameworks:
        changes += len(framework.GetBonds())
    return changes


def censurize(smi):
    """
    Convert a SMILES string into a censored SMARTS string.

    Parameters:
    smi (str): A SMILES (Simplified Molecular Input Line Entry System) string.

    Returns:
    str: A censored SMARTS (SMILES Arbitrary Target Specification) string.

    The censurize function converts a SMILES string into a censored SMARTS string
    by removing bond specifications and hydrogens. It removes heavy atom info by
    using the smarts syntax, preserving aromatic information
    """
    smarts = ""
    for car in smi:
        if car.isalpha():
            if car.upper() not in {"R", "L", "H"}: # R and L are only used in Br and Cl
                if car.isupper():
                    smarts += "A"
                else:
                    smarts += "a"
        elif car not in {"=", "/", "@", "-"}: # Remove bond info for now
            smarts += car
    return smarts


def atom_diff(mol1, mol2):
    """
    Calculate the difference in the number of atoms between two molecular structures.

    Parameters:
    mol1 (Chem.Mol): The first molecular structure.
    mol2 (Chem.Mol): The second molecular structure.

    Returns:
    int: The difference in the number of atoms between the two structures.

    The atom_diff function calculates the difference in the count of each element
    for two different rdkit molecules. It divides the output by two and rounds down
    to give the least number of mutations required to go from one molecule to another.
    An uneven difference would mean that an atom has to be added to the framework.
    It will, therefor, show up when calculating bond_diff instead.
    """
    mols = (mol1, mol2)
    formulas = []
    for mol in mols:
        formulas.append(np.zeros(100)) # 100 is just an arbitrary length to ensure that all element numbers fit
        for atom in mol.GetAtoms():
            formulas[-1][atom.GetAtomicNum()] += 1
    return int(np.absolute(formulas[0] - formulas[1]).sum() / 2)


def find_analogs(assay_chembl_ids):
    # the function will recieve a list of assays. for each assay it will suggest analog pairs and write to a csv-file
    for assay_chembl_id in assay_chembl_ids:
        # grab subset to consider
        subset = df.loc[df["assay_chembl_id"] == assay_chembl_id].copy()

        # open output file
        output_path = f"output/{assay_chembl_id}_proposed_pairs_2.csv"
        with open(output_path, "w") as f:
            # transform SMILES into rdkit molecules
            subset["mols"] = [Chem.MolFromSmiles(smi) for smi in subset["canonical_smiles"]]
            # also, make "molecular frameworks", which are representations of the molecules with heavy atoms
            # exchanged for wildcard atoms (A: aliphatic, a: aromatic) and bond info removed.
            subset["frameworks"] = [Chem.MolFromSmarts(censurize(smi)) for smi in subset["canonical_smiles"]]
            # finally, morgan fingerprints are generated for fingprint similarity filtering
            subset["fp"] = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in subset["mols"]]

            # prepare a distance matrix and set default distance to an arbitrary large number (100)
            n = len(subset)
            matrix = np.ones((n, n)) * 100

            # now, fill in the distance matrix
            for i in range(n):
                # calculate fingerprint similarity between mol[i] and remaining mols
                sim = BulkTanimotoSimilarity(subset["fp"].iloc[i], list(subset["fp"].iloc[(i + 1):]))
                for j in range(n - i - 1):
                    # apply first filter (tanimoto similarity coefficient)
                    if sim[j] >= tc_cutoff:
                        j += i + 1
                        # apply second filter (mutation distance between chemical formulas)
                        adist = atom_diff(subset["mols"].iloc[i], subset["mols"].iloc[j])
                        if adist <= a_cutoff:
                            # finally, calculate the "framework distance"
                            bdist = bond_diff(subset["frameworks"].iloc[i], subset["frameworks"].iloc[j])
                            # final distance will be the sum of mutation distance and framework distance
                            dist = adist + bdist
                            matrix[i, j] = dist
                            matrix[j, i] = dist

            # determine smallest distance for each molecule
            min_values = np.min(matrix, axis=0).astype(int)

            # prepare empty set to remove duplicates of the same match
            strings = set()
            for i in range(n):
                # check if the shortest distance is shorter than the threshold
                if min_values[i] <= d_cutoff:
                    # get chembl name of i
                    name = subset.molecule_chembl_id.iloc[i]
                    # get chembl names of all molecules separated by exactly the shortest distance found
                    analogs = subset.molecule_chembl_id[matrix[i] == min_values[i]]
                    for analog in analogs:
                        # sort names of the pair to allow removal of duplicates
                        pair = sorted([name, analog])
                        strings.add(f"{assay_chembl_id}, {pair[0]}, {pair[1]}, {min_values[i]}\n")

            # write all the promising pairs to the csv-file
            for string in strings:
                f.write(string)
            if len(strings) == 0:
                f.write(f"{assay_chembl_id}, None, None, 100\n")


# Specify settings settings
n = 60 # number of threads to use
tc_cutoff = 0.6 # tanimoto coefficient tolerated
a_cutoff = 3 # maximum atom difference tolerated
d_cutoff = 4 # maximum sum of atom and bond difference tolerated

if __name__ == '__main__':
    # check to see if there are uncompleted files in the output directory and if so remove them
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
    
    # save the names that are completed succesfully
    completed = {file.split("_")[0] for file in os.listdir("output/") if file.endswith('.csv')}
    
    df = pd.read_csv("/Users/vishnu/filtered_output_Ro5.csv", sep="\t")
    
    # sort assays by the molecule count of each to distribute evenly across each thread
    counts = pd.DataFrame(Counter(df["assay_chembl_id"]).items(), columns=['assay_chembl_id', 'count'])
    counts = counts.sort_values("count")
    
    # remove completed assays
    counts = counts[[assay_id not in completed for assay_id in counts["assay_chembl_id"]]]
    
    # generate inputs for multiprocessing
    inputs = [[] for _ in range(n)]
    for i, assay_chembl_id in enumerate(counts["assay_chembl_id"]):
        inputs[i%n].append(assay_chembl_id)
    
    # run analog search
    with Pool() as pool:
        results = pool.map(find_analogs, inputs)
