import pandas as pd
import os

# Define the directory containing the TSV files
tsv_dir = "/home/ubuntu/zip_contents/"

# List of files to process
files_to_process = [
    "Arabic LLMs Hallucination-OSACT2024-Train.txt",
    "Arabic LLMs Hallucination-OSACT2024-Dev.txt",
    "Arabic LLMs Hallucination-OSACT2024-Test.txt"
]

for file_name in files_to_process:
    file_path = os.path.join(tsv_dir, file_name)
    print(f"\n--- Processing file: {file_name} ---")
    try:
        # Load the TSV file into a pandas DataFrame
        # The instructions state TSV, and previous successful individual load used sep='\t'
        df = pd.read_csv(file_path, sep='\t', engine='python')

        print(f"First 5 rows of {file_name}:")
        print(df.head())

        print(f"\nShape of {file_name} (rows, columns):")
        print(df.shape)

        print(f"\nColumn names for {file_name}:")
        print(df.columns.tolist())

        print(f"\nBasic info for {file_name}:")
        df.info()

        # Verify expected columns
        expected_columns = ['claim_id', 'word_pos', 'readability', 'model', 'claim', 'label']
        # The first file loaded successfully had these columns. Let's assume they are consistent.
        # If not, the output will show the actual columns.
        if all(col in df.columns for col in expected_columns):
            print(f"\nAll expected columns found in {file_name}.")
        else:
            print(f"\nMissing or unexpected columns in {file_name}. Expected: {expected_columns}, Found: {df.columns.tolist()}")

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

