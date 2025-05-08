###############################################  Subset of NaN values    ###############################################


# # Load data quickly without type inference
# data = pd.read_csv(
#     "/Users/vishnu/Desktop/viji_files/sem8/BTP_3/mol_3d_descriptors.csv",
#     dtype='object'  # Treat all columns as strings for fastest loading
# )

# # Identify rows with ANY NaN values
# nan_mask = data.isna().any(axis=1)
# nan_subset = data.loc[nan_mask]

# # Save subset to new CSV (only rows with NaN values)
# nan_subset.to_csv(
#     "/Users/vishnu/Desktop/viji_files/sem8/BTP_3/nan_rows_subset.csv", 
#     index=False
# )

# print(f"Saved {len(nan_subset)} rows with NaN values to CSV")


############################################ Checking for Invalid SMILES  ###############################################

# import pandas as pd
# from rdkit import Chem
# from joblib import Parallel, delayed

# # Load the NaN subset
# nan_subset = pd.read_csv("/Users/vishnu/Desktop/viji_files/sem8/BTP_3/nan_rows_subset.csv")

# # Function to validate SMILES
# def is_valid_smiles(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         return mol is not None
#     except:
#         return False

# # Validate SMILES in parallel (fast)
# def validate_column(col_name):
#     return Parallel(n_jobs=-1)(
#         delayed(is_valid_smiles)(s) for s in nan_subset[col_name]
#     )

# # Check validity of SMILES_1 and SMILES_2
# nan_subset["valid_smiles_1"] = validate_column("canonical_smiles_1")
# nan_subset["valid_smiles_2"] = validate_column("canonical_smiles_2")

# # Filter rows with invalid SMILES
# invalid_smiles_rows = nan_subset[
#     (~nan_subset["valid_smiles_1"]) | (~nan_subset["valid_smiles_2"])
# ]
# print(f"Rows with invalid SMILES: {len(invalid_smiles_rows)}")


######################################################   Debug Mordred Descriptor Calculation  ###############################################

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(filename="descriptor_errors.log", level=logging.ERROR)

def regenerate_descriptors(row):
    """Recompute descriptors for a single row (molecule pair)."""
    smiles_1 = row["canonical_smiles_1"]
    smiles_2 = row["canonical_smiles_2"]
    
    try:
        # Process molecule 1
        mol1 = Chem.MolFromSmiles(smiles_1)
        mol1 = Chem.AddHs(mol1)
        AllChem.EmbedMolecule(mol1, maxAttempts=100)  # Generate 3D coords
        desc1 = Calculator(descriptors, ignore_3D=False)(mol1)
        desc1_dict = {f"mol_1_{k}": v for k, v in desc1.asdict().items()}  # Prefix for mol1
        
        # Process molecule 2
        mol2 = Chem.MolFromSmiles(smiles_2)
        mol2 = Chem.AddHs(mol2)
        AllChem.EmbedMolecule(mol2, maxAttempts=100)
        desc2 = Calculator(descriptors, ignore_3D=False)(mol2)
        desc2_dict = {f"mol_2_{k}": v for k, v in desc2.asdict().items()}  # Prefix for mol2
        
        # Update the row with regenerated descriptors
        for col in row.index:
            if col in desc1_dict:
                row[col] = desc1_dict[col]
            if col in desc2_dict:
                row[col] = desc2_dict[col]
        return row
    
    except Exception as e:
        logging.error(f"Failed for {smiles_1}.{smiles_2}: {str(e)}")
        return row  # Return the original row (unchanged)

def main():
    # Load the original dataset
    data = pd.read_csv(
        "/Users/vishnu/Desktop/viji_files/sem8/BTP_3/nan_rows_subset.csv",
        dtype='object'  # Fast loading, no type inference
    )
    
    # Identify rows with NaN values
    nan_mask = data.isna().any(axis=1)
    nan_subset = data[nan_mask].copy()
    
    # Regenerate descriptors for NaN subset
    tqdm.pandas(desc="Regenerating descriptors")
    nan_subset_cleaned = nan_subset.progress_apply(regenerate_descriptors, axis=1)
    
    # Fill remaining NaN values (only for descriptor columns)
    descriptor_cols = [col for col in nan_subset_cleaned.columns 
                       if col.startswith("mol_1_") or col.startswith("mol_2_")]
    nan_subset_cleaned[descriptor_cols] = nan_subset_cleaned[descriptor_cols].fillna(0)
    
    # Merge cleaned subset back into original data
    final_data = pd.concat([data[~nan_mask], nan_subset_cleaned], ignore_index=True)
    
    # Save the final dataset
    final_data.to_csv(
        "/Users/vishnu/Desktop/viji_files/sem8/BTP_3/nan_subset_regenerated.csv",
        index=False
    )
    print("Regenerated descriptors saved!")

if __name__ == "__main__":
    main()




######################      Merging regenerated subset and original data     ########################


# Load the original dataset and regenerated subset
original_data = pd.read_csv("/Users/vishnu/Desktop/viji_files/sem8/BTP_3/mol_3d_descriptors.csv")
cleaned_subset = pd.read_csv("/Users/vishnu/Desktop/viji_files/sem8/BTP_3/nan_subset_regenerated.csv")

# Remove rows from the original dataset that were regenerated
original_cleaned = original_data.drop(cleaned_subset.index)

# Merge the cleaned subset back into the original data
final_data = pd.concat([original_cleaned, cleaned_subset], ignore_index=True)

# Save the final dataset
final_data.to_csv("/Users/vishnu/Desktop/viji_files/sem8/BTP_3/mol_3d_descriptors_final.csv", index=False)


