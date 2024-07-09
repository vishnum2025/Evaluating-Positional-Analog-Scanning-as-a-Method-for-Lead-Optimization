from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from time import time

# checking if compound is drug-like
def is_drug_like(smiles):
    if smiles is None:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False

    mwt = Descriptors.MolWt(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    logp = Descriptors.MolLogP(mol)

    return mwt <= 500 and hbd <= 5 and hba <= 10 and logp <= 5


columns = ['activity_id', 
           'assay_chembl_id', 
           'assay_description',
           'assay_type',
           'canonical_smiles',
           'document_chembl_id',
           'document_journal',
           'document_year',
           'molecule_chembl_id',
           'parent_molecule_chembl_id',
           'standard_type',
           'standard_units',
           'standard_value']


active_data = new_client.activity
active_data = active_data.filter(standard_type__in=["IC50", "EC50", "Ki"])
active_data = active_data.filter(standard_relation__exact=["="])
active_data = active_data.filter(standard_units="nM")



batch_size = 10000 


def process_batch(batch):
    processed_data = []
    for data in batch:
        if is_drug_like(data['canonical_smiles']):
            processed_data.append({col: data[col] for col in columns})
    return processed_data

record_count = 0
with open("filtered_output.csv", "w") as f:
    f.write("\t".join(columns) + "\n")

    with ThreadPoolExecutor() as executor:
        while True:
            batch = list(active_data.only(columns)[record_count:record_count+batch_size])
            if not batch:
                break  

            future = executor.submit(process_batch, batch)
            processed_batch = future.result()


            for record in processed_batch:
                f.write("\t".join(str(record[col]) for col in columns) + "\n")

            record_count += batch_size
            

print("Filtering and data export complete.")