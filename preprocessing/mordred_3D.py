from rdkit import Chem
from mordred import Calculator, descriptors
import pandas as pd
import os
from tqdm import tqdm
import gc  # For garbage collection

# Set paths to the directories containing MOL files
mol_1_dir = "/Users/vishnu/Desktop/viji_files/sem8/mol_1"
mol_2_dir = "/Users/vishnu/Desktop/viji_files/sem8/mol_2"
output_dir = "/Users/vishnu/Desktop/viji_files/sem8/"
mol_1_output = os.path.join(output_dir, "mol_1_descriptors.csv")
mol_2_output = os.path.join(output_dir, "mol_2_descriptors.csv")
final_output = os.path.join(output_dir, "mol_3d_descriptors.csv")

#Load the original dataset to ensure ordering
original_data = "/Users/vishnu/Desktop/viji_files/sem8/processed_3D_structures__full.csv"

# Initialize Mordred descriptor calculator
calc = Calculator(descriptors, ignore_3D=False)  # Include 3D descriptors

# Batch size for processing
batch_size = 5000  # Adjust based on system's memory

def process_batch(batch_files, prefix, mol_dir):
    """
    Process a batch of MOL files and calculate Mordred descriptors.
    """
    descriptors_list = []
    for file_name in tqdm(batch_files, desc=f"Processing {prefix}"):
        try:
            mol_path = os.path.join(mol_dir, file_name)
            mol = Chem.MolFromMolFile(mol_path)
            if mol is not None:
                descriptors = calc(mol).asdict()
                descriptors['file_name'] = file_name  # Include file name for reference
                descriptors_list.append(descriptors)
            else:
                print(f"Could not parse {file_name}.")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    return pd.DataFrame(descriptors_list)

def process_descriptors_in_batches(file_order, prefix, mol_dir, output_file):
    """
    Process MOL files in batches and save intermediate results to avoid data loss.
    """
    for start_idx in range(0, len(file_order), batch_size):
        batch_files = file_order[start_idx:start_idx + batch_size]
        batch_descriptors = process_batch(batch_files, prefix, mol_dir)
        
        # Save batch to avoid losing progress
        batch_descriptors.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        print(f"Saved batch {start_idx // batch_size + 1} to {output_file}.")
        
        # Collect garbage to free memory
        del batch_descriptors
        gc.collect()

# Ensure the file order matches the dataset row order
mol_1_files = [f"mol_1_{i}.mol" for i in range(len(original_data))]
mol_2_files = [f"mol_2_{i}.mol" for i in range(len(original_data))]

# Calculate descriptors for mol_1
print("Calculating descriptors for mol_1...")
process_descriptors_in_batches(mol_1_files, "mol_1", mol_1_dir, mol_1_output)

# Calculate descriptors for mol_2
print("Calculating descriptors for mol_2...")
process_descriptors_in_batches(mol_2_files, "mol_2", mol_2_dir, mol_2_output)

# Load the processed descriptors
mol_1_descriptors = pd.read_csv(mol_1_output)
mol_2_descriptors = pd.read_csv(mol_2_output)

# Prefix column names to distinguish between the two sets of descriptors
mol_1_descriptors.columns = [f"mol_1_{col}" if col != "file_name" else col for col in mol_1_descriptors.columns]
mol_2_descriptors.columns = [f"mol_2_{col}" if col != "file_name" else col for col in mol_2_descriptors.columns]

# Merge descriptors with file names for mol_1 and mol_2
mol_1_descriptors['mol_1_file'] = mol_1_descriptors['file_name']
mol_2_descriptors['mol_2_file'] = mol_2_descriptors['file_name']
mol_1_descriptors.drop(columns=['file_name'], inplace=True)
mol_2_descriptors.drop(columns=['file_name'], inplace=True)

# Combine descriptors into a single DataFrame
final_descriptors = pd.concat([mol_1_descriptors, mol_2_descriptors], axis=1)

# Save the final dataset
final_descriptors.to_csv(final_output, index=False)
print(f"Descriptors calculated and saved to {final_output}")







#################################################### viewing the data #####################################################################

# data = pd.read_csv("/Users/vishnu/Desktop/viji_files/sem8/mol_3d_descriptors.csv", low_memory=False)
# print(len(data))
# print(data.columns)
# print(data.head())



# #################################################### SUBSET OF DATASET (to view) #####################################################################

# data = pd.read_csv("/Users/vishnu/Desktop/viji_files/sem8/final_dataset_with_3d_descriptors.csv", low_memory=False)

# # Extract the first 100 rows
# subset_data = data.iloc[:100, :]

# # Save the subset to a new CSV file
# subset_file_path = "/Users/vishnu/Desktop/viji_files/sem8/3d_data_subset.csv"
# subset_data.to_csv(subset_file_path, index=False)

# print(f"Subset of data saved at: {subset_file_path}")




