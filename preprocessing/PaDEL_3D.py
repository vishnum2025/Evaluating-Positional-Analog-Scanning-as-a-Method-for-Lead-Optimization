import os
import pandas as pd
import padelpy
from padelpy import padeldescriptor
from tqdm import tqdm


# Load the dataset
data = pd.read_csv("/Users/vishnu/Desktop/viji_files/sem8/processed_3D_structures__full.csv")


############################################# Extract MOL Data and Save as Individual Files ####################################


# Create directories to save MOL files
mol_1_dir = "/Users/vishnu/Desktop/viji_files/sem8/mol_1"
mol_2_dir = "/Users/vishnu/Desktop/viji_files/sem8/mol_2"
os.makedirs(mol_1_dir, exist_ok=True)
os.makedirs(mol_2_dir, exist_ok=True)

# Save MOL data to files with progress tracking
print("Saving MOL files...")
for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing molecules"):
    mol_1_content = row["mol_1"]
    mol_2_content = row["mol_2"]

    try:
        # Save mol_1
        mol_1_path = os.path.join(mol_1_dir, f"mol_1_{index}.mol")
        with open(mol_1_path, "w") as mol_file:
            mol_file.write(mol_1_content)

        # Save mol_2
        mol_2_path = os.path.join(mol_2_dir, f"mol_2_{index}.mol")
        with open(mol_2_path, "w") as mol_file:
            mol_file.write(mol_2_content)
    except Exception as e:
        print(f"Error processing row {index}: {e}")

print("MOL files saved successfully.")




################################################## PaDEL for 3D Descriptor Calculation #####################################




# import os
# import signal
# import pandas as pd
# from padelpy import padeldescriptor
# from tqdm import tqdm


# # Timeout handler
# class TimeoutException(Exception):
#     pass


# def timeout_handler(signum, frame):
#     raise TimeoutException("File processing exceeded timeout")


# def process_single_file(mol_file, output_dir):
#     try:
#         temp_output_file = os.path.join(output_dir, os.path.basename(mol_file).replace(".mol", "_desc.csv"))

#         # Calculate descriptors using PaDEL
#         padeldescriptor(
#             mol_dir=mol_file,
#             d_file=temp_output_file,
#             d_2d=False,
#             d_3d=True,
#             fingerprints=False
#         )

#         # Return the result
#         return pd.read_csv(temp_output_file)

#     except TimeoutException:
#         # Log timeout errors
#         with open("error_log.txt", "a") as log_file:
#             log_file.write(f"Timeout error for file {mol_file}\n")
#         return None
#     except Exception as e:
#         # Log any other errors
#         with open("error_log.txt", "a") as log_file:
#             log_file.write(f"Error processing file {mol_file}: {e}\n")
#         return None


# def batch_process_files(mol_dir, output_file, timeout=10, batch_size=100):
#     # Get a list of all .mol files in the directory
#     mol_files = [os.path.join(mol_dir, f) for f in os.listdir(mol_dir) if f.endswith(".mol")]
#     num_files = len(mol_files)

#     print(f"Found {num_files} MOL files in {mol_dir}.")

#     # Create a temporary directory for storing intermediate outputs
#     temp_output_dir = "temp_descriptors"
#     os.makedirs(temp_output_dir, exist_ok=True)

#     # Process files in batches
#     results = []
#     for batch_start in tqdm(range(0, num_files, batch_size), desc="Processing batches", unit="batch"):
#         batch_files = mol_files[batch_start:batch_start + batch_size]

#         # Process each file in the batch
#         for mol_file in tqdm(batch_files, desc=f"Processing batch {batch_start // batch_size + 1}", unit="file"):
#             signal.signal(signal.SIGALRM, timeout_handler)
#             signal.alarm(timeout)  # Set timeout
#             try:
#                 result = process_single_file(mol_file, temp_output_dir)
#                 if result is not None:
#                     results.append(result)
#             except TimeoutException:
#                 print(f"Timeout while processing {mol_file}")
#             finally:
#                 signal.alarm(0)  # Disable the alarm

#     # Combine all descriptors into a single DataFrame
#     if results:
#         final_descriptors = pd.concat(results, ignore_index=True)
#         final_descriptors.to_csv(output_file, index=False)
#         print(f"3D descriptors saved to {output_file}")
#     else:
#         print("No descriptors were successfully processed.")


# if __name__ == "__main__":
#     # Example usage for mol_1
#     mol_1_dir = "/Users/vishnu/Desktop/viji_files/sem8/mol_1"
#     output_file_1 = "/Users/vishnu/Desktop/viji_files/sem8/mol_1_descriptors.csv"
#     batch_process_files(mol_dir=mol_1_dir, output_file=output_file_1, timeout=10, batch_size=100)

#     # Example usage for mol_2
#     mol_2_dir = "/Users/vishnu/Desktop/viji_files/sem8/mol_2"
#     output_file_2 = "/Users/vishnu/Desktop/viji_files/sem8/mol_2_descriptors.csv"
#     batch_process_files(mol_dir=mol_2_dir, output_file=output_file_2, timeout=10, batch_size=100)

