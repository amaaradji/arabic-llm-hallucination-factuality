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

results_summary = "Descriptive Statistics - Hallucination Rate per Readability Level\n\n"

print("--- Calculating Hallucination Rate per Readability Level ---")

for dataset_name, file_name in files_to_process.items():
    file_path = os.path.join(tsv_dir, file_name)
    results_summary += f"\nDataset: {dataset_name} ({file_name})\n"
    print(f"\nProcessing dataset: {dataset_name} ({file_name})")
    try:
        # Load the TSV file using the robust parsing strategy
        df = pd.read_csv(file_path, sep='\t', engine='python', quoting=3) # csv.QUOTE_NONE, corrected separator

        # Ensure readability is treated as integer
        df["readability"] = pd.to_numeric(df["readability"], errors='coerce').dropna().astype(int)

        # Define hallucinated labels
        hallucinated_labels = ["FI", "NF"]
        df["is_hallucination"] = df["label"].isin(hallucinated_labels)

        # Calculate hallucination rate per readability level
        hallucination_rates_per_readability = {}
        for level in sorted(df["readability"].unique()):
            if not (1 <= level <= 4):
                print(f"  Skipping unexpected readability level: {level}")
                continue
            
            subset = df[df["readability"] == level]
            if subset.empty:
                hallucination_rates_per_readability[level] = {"rate": 0, "count": 0, "hallucinated_count": 0}
                results_summary += f"  Readability Level {level}: Rate = 0.00% (0/0)\n"
                print(f"  Readability Level {level}: Rate = 0.00% (0/0)")
            else:
                hallucinated_count = subset["is_hallucination"].sum()
                total_count = len(subset)
                rate = (hallucinated_count / total_count) * 100 if total_count > 0 else 0
                hallucination_rates_per_readability[level] = {"rate": rate, "count": total_count, "hallucinated_count": hallucinated_count}
                results_summary += f"  Readability Level {level}: Rate = {rate:.2f}% ({hallucinated_count}/{total_count})\n"
                print(f"  Readability Level {level}: Rate = {rate:.2f}% ({hallucinated_count}/{total_count})")
        
        df_readability_rates = pd.DataFrame.from_dict(hallucination_rates_per_readability, orient="index")
        df_readability_rates.index.name = "Readability_Level"
        csv_path = os.path.join(output_dir, f"{dataset_name}_hallucination_rate_by_readability.csv")
        df_readability_rates.to_csv(csv_path)
        results_summary += f"  Results saved to: {csv_path}\n"

    except FileNotFoundError:
        error_message = f"ERROR: The file {file_path} was not found."
        results_summary += error_message + "\n"
        print(error_message)
    except Exception as e:
        error_message = f"ERROR: An error occurred while processing {file_path}: {e}"
        results_summary += error_message + "\n"
        print(error_message)

print("\n--- Calculation Complete ---")
results_summary += "\n--- Calculation Complete ---\n"

# Save summary to a file
summary_file_path = os.path.join(output_dir, "hallucination_rate_by_readability_summary.txt")
with open(summary_file_path, "w", encoding="utf-8") as f:
    f.write(results_summary)
print(f"Summary of hallucination rates by readability saved to {summary_file_path}")

