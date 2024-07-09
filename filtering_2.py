from multiprocessing import Pool
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from tqdm import tqdm  

def is_Ro5_compliant(smiles):
    if not isinstance(smiles, str):
        return False

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False

    mwt = Descriptors.MolWt(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    logp = Descriptors.MolLogP(mol)

    return mwt <= 500 and hbd <= 5 and hba <= 10 and logp <= 5

def parallel_process(array, function, n_cores=4):
    with Pool(n_cores) as pool:
        result = pool.map(function, array)
    return result

def main():

    data = pd.read_csv("/Users/vishnu/output.csv", sep="\t")
    print("1")

    data = data[data['standard_units'] == 'nM']
    print("2")

    data['Ro5_compliant'] = parallel_process(data['canonical_smiles'].to_numpy(), is_Ro5_compliant, n_cores=4)
    data = data[data['Ro5_compliant']]
    print("3")

    molecule_counts = data.groupby('assay_chembl_id')['molecule_chembl_id'].nunique()
    valid_assays = molecule_counts[molecule_counts >= 2].index
    data = data[data['assay_chembl_id'].isin(valid_assays)]


    print(f"Processed {len(data)} records out of {len(data)}")

    data.to_csv("filtered_output_Ro5.csv", sep="\t", index=False)

if __name__ == '__main__':
    main()