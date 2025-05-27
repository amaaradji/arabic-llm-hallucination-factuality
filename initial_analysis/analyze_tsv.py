#!/usr/bin/env python
import pandas as pd
import os

# Define the directory containing the TSV files
tsv_dir = "/home/ubuntu/zip_contents/"

# List of files to process
files_to_process = [
    os.path.join(tsv_dir, "Arabic_Benchmark_for_LLMs_-_Train.tsv"),
    os.path.join(tsv_dir, "Arabic_Benchmark_for_LLMs_-_Dev.tsv"),
    os.path.join(tsv_dir, "Arabic_Benchmark_for_LLMs_-_Test.tsv")
]

# Initialize a dictionary to store the results
results = {}

# Process each file
for file_path in files_to_process:
    try:
        # Load the TSV file into a pandas DataFrame
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip') # Added on_bad_lines='skip' to handle potential errors

        # Get basic information about the DataFrame
        file_name = os.path.basename(file_path)
        num_rows, num_cols = df.shape
        column_names = df.columns.tolist()

        # Store the information in the results dictionary
        results[file_name] = {
            "rows": num_rows,
            "columns": num_cols,
            "column_names": column_names
        }

        # Print some basic information for verification
        print(f"Successfully processed: {file_name}")
        print(f"  Rows: {num_rows}, Columns: {num_cols}")
        print(f"  Column Names: {column_names[:5]}...") # Print first 5 column names

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

# Print the results
print("\nSummary of TSV files:")
for file_name, data in results.items():
    print(f"  {file_name}: Rows={data['rows']}, Columns={data['columns']}")


