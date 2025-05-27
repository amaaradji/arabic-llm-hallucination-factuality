import pandas as pd
import os

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

results_summary_text = "Descriptive Statistics - Cross-Tabulation: Readability x POS x Model Hallucination Rates\n\n"

print("--- Calculating Cross-Tabulation: Readability x POS x Model Hallucination Rates ---")

# POS extraction function (reused from previous script)
def extract_pos_tag(word_pos_string):
    if pd.isna(word_pos_string):
        return None
    word_pos_string = str(word_pos_string)
    parts = word_pos_string.split("_")
    for part in parts:
        if part in ["noun", "verb", "adj"]:
            return part
    return "other"

for dataset_name, file_name in files_to_process.items():
    file_path = os.path.join(tsv_dir, file_name)
    results_summary_text += f"\nDataset: {dataset_name} ({file_name})\n"
    print(f"\nProcessing dataset: {dataset_name} ({file_name})")
    try:
        # Corrected pd.read_csv call
        df = pd.read_csv(file_path, sep=		'\t', engine='python', quoting=3) # csv.QUOTE_NONE

        # Prepare columns
        df["pos_extracted"] = df["word_pos"].apply(extract_pos_tag)
        df["readability"] = pd.to_numeric(df["readability"], errors='coerce').dropna().astype(int)
        # Ensure model names are consistent (e.g., lowercase, handle variants)
        df["model"] = df["model"].astype(str).str.lower().replace({"gpt4": "gpt-4", "chatgpt": "gpt-3.5"}) # Example normalization
        
        hallucinated_labels = ["FI", "NF"]
        df["is_hallucination"] = df["label"].isin(hallucinated_labels)

        # Filter for relevant POS tags and readability levels
        relevant_pos_tags = ["noun", "verb", "adj", "other"]
        df = df[df["pos_extracted"].isin(relevant_pos_tags)]
        df = df[df["readability"].isin([1, 2, 3, 4])]
        # Filter for relevant models if necessary, e.g., gpt-3.5, gpt-4
        relevant_models = ["gpt-3.5", "gpt-4"]
        df = df[df["model"].isin(relevant_models)]

        if df.empty:
            results_summary_text += "  No data after filtering for relevant POS, readability, and model.\n"
            print("  No data after filtering for relevant POS, readability, and model.")
            continue

        # Group by readability, pos_extracted, and model
        grouped = df.groupby(["readability", "pos_extracted", "model"])

        cross_tab_results = []
        for name, group_df in grouped:
            readability_level, pos_tag_val, model_val = name
            total_count = len(group_df)
            hallucinated_count = group_df["is_hallucination"].sum()
            rate = (hallucinated_count / total_count) * 100 if total_count > 0 else 0
            
            cross_tab_results.append({
                "Readability": readability_level,
                "POS_Tag": pos_tag_val,
                "Model": model_val,
                "Hallucination_Rate": rate,
                "Hallucinated_Count": hallucinated_count,
                "Total_Count": total_count
            })
            results_summary_text += f"  R:{readability_level}, POS:{pos_tag_val}, M:{model_val} -> Rate: {rate:.2f}% ({hallucinated_count}/{total_count})\n"
            print(f"  R:{readability_level}, POS:{pos_tag_val}, M:{model_val} -> Rate: {rate:.2f}% ({hallucinated_count}/{total_count})")

        if not cross_tab_results:
            results_summary_text += "  No results generated for cross-tabulation (possibly empty groups).\n"
            print("  No results generated for cross-tabulation (possibly empty groups).")
        else:
            df_cross_tab = pd.DataFrame(cross_tab_results)
            csv_path = os.path.join(output_dir, f"{dataset_name}_cross_tab_readability_pos_model.csv")
            df_cross_tab.to_csv(csv_path, index=False)
            results_summary_text += f"  Cross-tabulation results saved to: {csv_path}\n"

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

summary_file_path = os.path.join(output_dir, "cross_tab_readability_pos_model_summary.txt")
with open(summary_file_path, "w", encoding="utf-8") as f:
    f.write(results_summary_text)
print(f"Summary of cross-tabulation saved to {summary_file_path}")

