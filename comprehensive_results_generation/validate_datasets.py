import pandas as pd
import sys

# Define the directory containing the TSV files
tsv_dir = "/home/ubuntu/zip_contents/"

# List of files to process
files_to_process = [
    "Arabic LLMs Hallucination-OSACT2024-Train.txt",
    "Arabic LLMs Hallucination-OSACT2024-Dev.txt",
    "Arabic LLMs Hallucination-OSACT2024-Test.txt"
]

all_validations_passed = True

for file_name in files_to_process:
    file_path = tsv_dir + file_name
    try:
        # Load the TSV file using pandas
        # Corrected sep and quoting parameters
        df = pd.read_csv(file_path, sep='\t', quoting=3) # csv.QUOTE_NONE

        # Perform any necessary validation checks here
        # For example, check if the DataFrame is empty
        if df.empty:
            print(f"Warning: {file_name} is empty or could not be parsed correctly.")
            all_validations_passed = False
            continue

        print(f"Successfully processed: {file_name}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        all_validations_passed = False
        continue

if all_validations_passed:
    print("All files validated successfully.")
else:
    print("One or more files failed validation.")

