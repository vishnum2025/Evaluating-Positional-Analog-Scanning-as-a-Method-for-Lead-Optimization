import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

def generate_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) if mol else None
    except Exception as e:
        print(f"Error in fingerprint generation for {smiles}: {e}")
        return None

def compute_similarity_and_mcs(args):
    idx1, idx2, smiles1, smiles2, fp1, fp2 = args
    if fp1 and fp2:
        sim = BulkTanimotoSimilarity(fp1, [fp2])[0]
        if sim >= 0.9:
            try:
                mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
                mcs_result = rdFMCS.FindMCS([mol1, mol2], timeout=2)
                return idx1, idx2, sim, mcs_result.smartsString if mcs_result.smartsString else ""
            except Exception as e:
                print(f"Error in MCS calculation for pair ({idx1}, {idx2}): {e}")
    return idx1, idx2, 0, ""

def batch_process(data, batch_size):
    results = []
    total = len(data)
    with tqdm(total=total, desc="Overall Progress") as pbar:
        with ProcessPoolExecutor() as executor:
            for start in range(0, total, batch_size):
                end = start + batch_size
                batch_data = data.iloc[start:end]
                fps = list(tqdm(executor.map(generate_fingerprint, batch_data['canonical_smiles']), total=len(batch_data), desc="Fingerprints"))
                pairs = [(batch_data.iloc[i].name, batch_data.iloc[j].name, batch_data.iloc[i]['canonical_smiles'], batch_data.iloc[j]['canonical_smiles'], fps[i], fps[j])
                         for i in range(len(batch_data)) for j in range(i + 1, len(batch_data))]
                batch_results = list(tqdm(executor.map(compute_similarity_and_mcs, pairs), total=len(pairs), desc="Similarity & MCS"))
                results.extend(batch_results)
                pbar.update(len(batch_data))
    return results

def main():
    data = pd.read_csv("/Users/vishnu/filtered_output_Ro5.csv", sep="\t")
    batch_size = 100  # Adjust based on your system's performance and memory capacity

    results = batch_process(data, batch_size)
    
    results_df = pd.DataFrame(results, columns=['Molecule_1', 'Molecule_2', 'Tanimoto_Similarity', 'MCS_Smarts'])
    results_df.to_csv("molecular_similarity_mcs_results_batched.csv", index=False)
    print(f"Processed {len(results_df)} molecule pairs.")

if __name__ == '__main__':
    main()
