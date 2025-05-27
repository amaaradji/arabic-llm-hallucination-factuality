import pandas as pd
import os

# Define the directory containing the TSV files
tsv_dir = "/home/ubuntu/zip_contents/"

# List of files to process
files_to_process = {
    "Train": "Arabic LLMs Hallucination-OSACT2024-Train.txt",
    "Dev": "Arabic LLMs Hallucination-OSACT2024-Dev.txt",
    "Test": "Arabic LLMs Hallucination-OSACT2024-Test.txt"
}

all_validations_passed = True
validation_summary = ""

print("--- Starting Data Validation ---")
validation_summary += "--- Starting Data Validation ---\n"

for dataset_name, file_name in files_to_process.items():
    file_path = os.path.join(tsv_dir, file_name)
    validation_summary += f"\n--- Validating file: {file_name} ({dataset_name}) ---\n"
    print(f"\n--- Validating file: {file_name} ({dataset_name}) ---")
    try:
        # Corrected robust parsing strategy: sep=\t and quoting=3 (csv.QUOTE_NONE)
        df = pd.read_csv(file_path, sep=\'\t\', engine=\'python\', quoting=3)

        validation_summary += f"Shape: {df.shape}\n"
        print(f"Shape: {df.shape}")

        # Validate column names exist
        expected_columns = ["claim_id", "word_pos", "readability", "model", "claim", "label"]
        if not all(col in df.columns for col in expected_columns):
            validation_summary += f"ERROR: Missing one or more expected columns. Found: {df.columns.tolist()}\n"
            print(f"ERROR: Missing one or more expected columns. Found: {df.columns.tolist()}")
            all_validations_passed = False
        else:
            validation_summary += f"All expected columns found: {df.columns.tolist()}\n"
            print(f"All expected columns found: {df.columns.tolist()}")

        # Validate \'word_pos\'
        if "word_pos" in df.columns:
            unique_word_pos = df["word_pos"].unique()
            validation_summary += f"Unique \'word_pos\' examples (first 5): {list(unique_word_pos[:5])}\n"
            print(f"Unique \'word_pos\' examples (first 5): {list(unique_word_pos[:5])}")
        else:
            validation_summary += "ERROR: \'word_pos\' column not found.\n"
            print("ERROR: \'word_pos\' column not found.")
            all_validations_passed = False

        # Validate \'readability\'
        if "readability" in df.columns:
            df_readability_numeric = pd.to_numeric(df["readability"], errors='coerce')
            unique_readability = sorted(df_readability_numeric.dropna().unique())
            validation_summary += f"Unique \'readability\' values (numeric, non-null): {unique_readability}\n"
            print(f"Unique \'readability\' values (numeric, non-null): {unique_readability}")
            if not all(1 <= int(level) <= 4 for level in unique_readability):
                validation_summary += "ERROR: Readability levels are not within the expected range of 1-4.\n"
                print("ERROR: Readability levels are not within the expected range of 1-4.")
                all_validations_passed = False
            elif df_readability_numeric.isnull().any():
                validation_summary += "WARNING: Some readability values are non-numeric or null.\n"
                print("WARNING: Some readability values are non-numeric or null.")
            else:
                validation_summary += "Readability levels are numeric and within the expected range (1-4).\n"
                print("Readability levels are numeric and within the expected range (1-4).")
        else:
            validation_summary += "ERROR: \'readability\' column not found.\n"
            print("ERROR: \'readability\' column not found.")
            all_validations_passed = False

        # Validate \'model\'
        if "model" in df.columns:
            unique_models = df["model"].unique()
            validation_summary += f"Unique \'model\' values: {list(unique_models)}\n"
            print(f"Unique \'model\' values: {list(unique_models)}")
            expected_models_variants = ["gpt-3.5", "gpt-4", "chatgpt", "gpt4"]
            found_models_lower = [str(model).lower() for model in unique_models]
            if not all(model_lower in expected_models_variants for model_lower in found_models_lower):
                validation_summary += f"WARNING: Model names might not strictly match expected variants. Found: {list(unique_models)}. Expected variants (case-insensitive): {expected_models_variants}. Please verify.\n"
                print(f"WARNING: Model names might not strictly match expected variants. Found: {list(unique_models)}. Expected variants (case-insensitive): {expected_models_variants}. Please verify.")
            else:
                validation_summary += "Model names appear to be consistent with expectations (case-insensitive).\n"
                print("Model names appear to be consistent with expectations (case-insensitive).")
        else:
            validation_summary += "ERROR: \'model\' column not found.\n"
            print("ERROR: \'model\' column not found.")
            all_validations_passed = False

        # Validate \'label\'
        if "label" in df.columns:
            unique_labels = sorted(df["label"].astype(str).unique())
            validation_summary += f"Unique \'label\' values: {unique_labels}\n"
            print(f"Unique \'label\' values: {unique_labels}")
            expected_labels = sorted(["FC", "FI", "NF"])
            if unique_labels != expected_labels:
                validation_summary += f"ERROR: Label values do not match expected {{FC, FI, NF}}. Found: {unique_labels}, Expected: {expected_labels}\n"
                print(f"ERROR: Label values do not match expected {{FC, FI, NF}}. Found: {unique_labels}, Expected: {expected_labels}")
                all_validations_passed = False
            else:
                validation_summary += "Label values match expected {FC, FI, NF}.\n"
                print("Label values match expected {FC, FI, NF}.")
        else:
            validation_summary += "ERROR: \'label\' column not found.\n"
            print("ERROR: \'label\' column not found.")
            all_validations_passed = False

    except FileNotFoundError:
        validation_summary += f"ERROR: The file {file_path} was not found.\n"
        print(f"ERROR: The file {file_path} was not found.")
        all_validations_passed = False
    except Exception as e:
        validation_summary += f"ERROR: An error occurred while processing {file_path}: {e}\n"
        print(f"ERROR: An error occurred while processing {file_path}: {e}")
        all_validations_passed = False

print("\n--- Overall Validation Status ---")
validation_summary += "\n--- Overall Validation Status ---\n"
if all_validations_passed:
    print("All datasets successfully validated against specifications.")
    validation_summary += "All datasets successfully validated against specifications.\n"
else:
    print("One or more datasets failed validation. Please review the logs.")
    validation_summary += "One or more datasets failed validation. Please review the logs.\n"

# Save summary to a file
with open("/home/ubuntu/validation_summary.txt", "w", encoding="utf-8") as f:
    f.write(validation_summary)
print("Validation summary saved to /home/ubuntu/validation_summary.txt")

