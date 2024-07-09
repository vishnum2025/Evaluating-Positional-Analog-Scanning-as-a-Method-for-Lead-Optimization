# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import rdFMCS, Descriptors
# from multiprocessing import Pool, cpu_count
# import logging
# import signal
# from tqdm import tqdm

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class TimeoutException(Exception):
#     pass

# def timeout_handler(signum, frame):
#     raise TimeoutException

# def compute_mcs_with_timeout(mol1, mol2, timeout=3):
#     # Set the signal handler and specify a timeout
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(timeout)
#     try:
#         result = rdFMCS.FindMCS([mol1, mol2])
#         return result
#     except TimeoutException:
#         logging.warning("Timeout occurred during MCS computation")
#         return None
#     finally:
#         signal.alarm(0)  # Disable the alarm

# def analyze_row(args):
#     index, row = args
#     try:
#         mol1 = Chem.MolFromSmiles(row['canonical_smiles_1'])
#         mol2 = Chem.MolFromSmiles(row['canonical_smiles_2'])
#         if mol1 is None or mol2 is None:
#             return [index, False, None]
#         count1 = Descriptors.HeavyAtomCount(mol1)
#         count2 = Descriptors.HeavyAtomCount(mol2)
#         if abs(count1 - count2) == 1:
#             mcs = compute_mcs_with_timeout(mol1, mol2, timeout=3)  # 3-second timeout
#             if mcs is None or mcs.numAtoms == 0:
#                 return [index, False, None]
#             common_scaffold = Chem.MolFromSmarts(mcs.smartsString)
#             return [index, True, Chem.MolToSmiles(common_scaffold)]
#         return [index, False, None]
#     except Exception as e:
#         logging.error(f"Error processing row {index}: {e}")
#         return [index, False, None]

# def parallel_process(df, func):
#     with Pool(cpu_count()) as pool:
#         results = list(tqdm(pool.imap(func, df.iterrows()), total=len(df)))
#     return pd.DataFrame(results, columns=['Index', 'Single_Addition', 'Common_Scaffold'])

# def main():
#     csv_file_path = '/Users/vishnu/characterized_modifications.csv'
#     activity_file_path = '/Users/vishnu/filtered_output_Ro5.csv'
#     df_filtered = pd.read_csv(csv_file_path)
#     df_filtered['Molecule1_ID'] = df_filtered['Molecule1_ID'].str.strip()
#     df_filtered['Molecule2_ID'] = df_filtered['Molecule2_ID'].str.strip()
#     activity_df = pd.read_csv(activity_file_path, sep="\t")
#     activity_df['molecule_chembl_id'] = activity_df['molecule_chembl_id'].str.strip()

#     df_inversion = df_filtered[df_filtered['Modifications'].str.contains('Inversion of stereocenters detected', na=False)]
#     df_merged = df_inversion.merge(
#         activity_df[['molecule_chembl_id', 'canonical_smiles']],
#         left_on='Molecule1_ID', right_on='molecule_chembl_id'
#     ).merge(
#         activity_df[['molecule_chembl_id', 'canonical_smiles']],
#         left_on='Molecule2_ID', right_on='molecule_chembl_id',
#         suffixes=('_1', '_2')
#     )

#     results = parallel_process(df_merged, analyze_row)
#     df_results = pd.DataFrame(results, columns=['Index', 'Single_Addition', 'Common_Scaffold'])
#     df_merged = pd.concat([df_merged, df_results.set_index('Index')], axis=1)

#     output_file_path = '/Users/vishnu/single_atom_additions_2.csv'
#     df_merged.to_csv(output_file_path, index=False)
#     logging.info(f"Results saved to {output_file_path}")

# if __name__ == '__main__':
#     main()



import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS, Descriptors
from joblib import Parallel, delayed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import signal
from contextlib import contextmanager

# Create a timeout handler
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def analyze_row(index, row):
    try:
        with time_limit(3): 
            mol1 = Chem.MolFromSmiles(row['canonical_smiles_1'])
            mol2 = Chem.MolFromSmiles(row['canonical_smiles_2'])
            if not mol1 or not mol2:
                return index, False, None
            count1 = Descriptors.HeavyAtomCount(mol1)
            count2 = Descriptors.HeavyAtomCount(mol2)
            if abs(count1 - count2) == 1:
                mcs = rdFMCS.FindMCS([mol1, mol2])
                if mcs.numAtoms == 0:
                    return index, False, None
                common_scaffold = Chem.MolFromSmarts(mcs.smartsString)
                return index, True, Chem.MolToSmiles(common_scaffold)
            return index, False, None
    except TimeoutException:
        logging.error(f"Timeout processing row {index}")
        return index, False, None
    except Exception as e:
        logging.error(f"Error processing row {index}: {e}")
        return index, False, None



def main():
    csv_file_path = '/Users/vishnu/characterized_modifications.csv'
    activity_file_path = '/Users/vishnu/filtered_output_Ro5.csv'

    df_filtered = pd.read_csv(csv_file_path)
    df_filtered['Molecule1_ID'] = df_filtered['Molecule1_ID'].str.strip()
    df_filtered['Molecule2_ID'] = df_filtered['Molecule2_ID'].str.strip()
    activity_df = pd.read_csv(activity_file_path, sep="\t")
    activity_df['molecule_chembl_id'] = activity_df['molecule_chembl_id'].str.strip()

    df_inversion = df_filtered[df_filtered['Modifications'].str.contains('Inversion of stereocenters detected', na=False)]
    df_merged = df_inversion.merge(
        activity_df[['molecule_chembl_id', 'canonical_smiles']],
        left_on='Molecule1_ID', right_on='molecule_chembl_id'
    ).merge(
        activity_df[['molecule_chembl_id', 'canonical_smiles']],
        left_on='Molecule2_ID', right_on='molecule_chembl_id',
        suffixes=('_1', '_2')
    )

    results = Parallel(n_jobs=-1, verbose=10)(delayed(analyze_row)(i, row) for i, row in df_merged.iterrows())
    results_df = pd.DataFrame(results, columns=['Index', 'Single_Addition', 'Common_Scaffold'])
    df_final = df_merged.join(results_df.set_index('Index'))

    output_file_path = '/Users/vishnu/single_atom_additions_2.csv'
    df_final.to_csv(output_file_path, index=False)
    logging.info(f"Results saved to {output_file_path}")

if __name__ == '__main__':
    main()
