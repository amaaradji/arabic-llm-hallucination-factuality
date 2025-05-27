import pandas as pd
import os

# Define the directory containing the TSV files
tsv_dir = "/home/ubuntu/zip_contents/"

# File to process
file_name = "Arabic LLMs Hallucination-OSACT2024-Train.txt"
file_path = os.path.join(tsv_dir, file_name)

print(f"--- Inspecting file: {file_name} ---")

# First, let's try to print the first few lines to see the structure
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        print("First 5 lines of the raw file:")
        for i in range(5):
            line = f.readline()
            if not line:
                break
            print(line.strip())
except Exception as e:
    print(f"Error reading raw file: {e}")

print(f"\n--- Attempting to parse file: {file_name} with pandas ---")
try:
    # Try different parsing strategies
    # Strategy 1: Default tab separation (as per previous successful Dev/Test)
    try:
        print("\nAttempting with sep=\t")
        df = pd.read_csv(file_path, sep='\t', engine='python')
        print(f"Successfully parsed with sep=\t")
        print(df.head())
        print(df.shape)
        print(df.columns.tolist())
        df.info()
    except Exception as e1:
        print(f"Failed with sep=\t: {e1}")
        # Strategy 2: Try with quoting=3 (csv.QUOTE_NONE) if tab separation fails due to quotes
        try:
            print("\nAttempting with sep=\t and quoting=3 (QUOTE_NONE)")
            df = pd.read_csv(file_path, sep='\t', engine='python', quoting=3) # quoting=3 is csv.QUOTE_NONE
            print(f"Successfully parsed with sep=\t and quoting=3")
            print(df.head())
            print(df.shape)
            print(df.columns.tolist())
            df.info()
        except Exception as e2:
            print(f"Failed with sep=\t and quoting=3: {e2}")
            # Strategy 3: Try with error_bad_lines=False or on_bad_lines='skip' (already tried, but let's be explicit)
            try:
                print("\nAttempting with sep=\t and on_bad_lines='skip'")
                df = pd.read_csv(file_path, sep='\t', engine='python', on_bad_lines='skip')
                print(f"Successfully parsed with sep=\t and on_bad_lines='skip'")
                print(df.head())
                print(df.shape)
                print(df.columns.tolist())
                df.info()
            except Exception as e3:
                print(f"Failed with sep=\t and on_bad_lines='skip': {e3}")
                print("All parsing strategies failed.")

except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


