import pandas as pd
import os
import re

# Define the directory containing the TSV files
tsv_dir = "/home/ubuntu/comprehensive_results_workdir/input_data/"
output_dir = "/home/ubuntu/comprehensive_results_workdir/output_data/descriptive_stats_results/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Files to process
files_to_process = {
    "Train": "Arabic LLMs Hallucination-OSACT2024-Train.txt",
    "Dev": "Arabic LLMs Hallucination-OSACT2024-Dev.txt",
    "Test": "Arabic LLMs Hallucination-OSACT2024-Test.txt"
}

results_summary_text = "Descriptive Statistics - Hallucination Rate per POS Tag\n\n"

print("--- Calculating Hallucination Rate per POS Tag ---")

# Expected POS tags from instructions: noun, verb, adj
# The word_pos column format is like 'word_POSTAG' or 'word_POSTAG_subtag'
# We need to extract the main POS tag.
def extract_pos_tag(word_pos_string):
    if pd.isna(word_pos_string):
        return None
    # Convert to string to be safe
    word_pos_string = str(word_pos_string)
    # Split by underscore
    parts = word_pos_string.split("_")
    # The POS tag is usually the second part (e.g., word_noun, word_verb_something)
    # We are looking for 'noun', 'verb', or 'adj'
    for part in parts:
        if part in ["noun", "verb", "adj"]:
            return part
    return "other" # Categorize if not one of the main three

for dataset_name, file_name in files_to_process.items():
    file_path = os.path.join(tsv_dir, file_name)
    results_summary_text += f"\nDataset: {dataset_name} ({file_name})\n"
    print(f"\nProcessing dataset: {dataset_name} ({file_name})")
    try:
        # Corrected pd.read_csv call
        df = pd.read_csv(file_path, sep='\t', engine='python', quoting=3) # csv.QUOTE_NONE

        df["pos_extracted"] = df["word_pos"].apply(extract_pos_tag)
        
        # Define hallucinated labels
        hallucinated_labels = ["FI", "NF"]
        df["is_hallucination"] = df["label"].isin(hallucinated_labels)

        # Calculate hallucination rate per extracted POS tag
        hallucination_rates_per_pos = {}
        # Focus on the instructed POS tags: noun, verb, adj, and include 'other' for completeness
        relevant_pos_tags = ["noun", "verb", "adj", "other"]

        for pos_tag_val in relevant_pos_tags:
            subset = df[df["pos_extracted"] == pos_tag_val]
            if subset.empty:
                hallucination_rates_per_pos[pos_tag_val] = {"rate": 0, "count": 0, "hallucinated_count": 0}
                results_summary_text += f"  POS Tag \"{pos_tag_val}\": Rate = 0.00% (0/0)\n"
                print(f"  POS Tag \"{pos_tag_val}\": Rate = 0.00% (0/0)")
            else:
                hallucinated_count = subset["is_hallucination"].sum()
                total_count = len(subset)
                rate = (hallucinated_count / total_count) * 100 if total_count > 0 else 0
                hallucination_rates_per_pos[pos_tag_val] = {"rate": rate, "count": total_count, "hallucinated_count": hallucinated_count}
                results_summary_text += f"  POS Tag \"{pos_tag_val}\": Rate = {rate:.2f}% ({hallucinated_count}/{total_count})\n"
                print(f"  POS Tag \"{pos_tag_val}\": Rate = {rate:.2f}% ({hallucinated_count}/{total_count})")
        
        df_pos_rates = pd.DataFrame.from_dict(hallucination_rates_per_pos, orient="index")
        df_pos_rates.index.name = "POS_Tag"
        csv_path = os.path.join(output_dir, f"{dataset_name}_hallucination_rate_by_pos.csv")
        df_pos_rates.to_csv(csv_path)
        results_summary_text += f"  Results saved to: {csv_path}\n"

    except FileNotFoundError:
        error_message = f"ERROR: The file {file_path} was not found."
        results_summary_text += error_message + "\n"
        print(error_message)
    except Exception as e:
        error_message = f"ERROR: An error occurred while processing {file_path}: {e}"
        results_summary_text += error_message + "\n"
        print(error_message)

print("\n--- Calculation Complete ---")
results_summary_text += "\n--- Calculation Complete ---\n"

summary_file_path = os.path.join(output_dir, "hallucination_rate_by_pos_summary.txt")
with open(summary_file_path, "w", encoding="utf-8") as f:
    f.write(results_summary_text)
print(f"Summary of hallucination rates by POS saved to {summary_file_path}")

