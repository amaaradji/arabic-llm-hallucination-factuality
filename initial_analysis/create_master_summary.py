import os

output_dir = "/home/ubuntu/descriptive_stats_results/"
master_summary_file_path = os.path.join(output_dir, "master_descriptive_stats_summary.txt")

summary_files_to_concatenate = [
    "hallucination_rate_by_readability_summary.txt",
    "hallucination_rate_by_pos_summary.txt",
    "cross_tab_readability_pos_model_summary.txt"
]

print(f"--- Creating Master Summary File for Descriptive Statistics ---")

master_summary_content = "Master Summary of Descriptive Statistics\n"
master_summary_content += "=========================================\n\n"

for summary_file_name in summary_files_to_concatenate:
    file_path = os.path.join(output_dir, summary_file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            master_summary_content += f"\n--- Content from {summary_file_name} ---\n"
            master_summary_content += f.read()
            master_summary_content += "\n--- End of {summary_file_name} ---\n\n"
        print(f"Successfully read and added: {summary_file_name}")
    except FileNotFoundError:
        error_message = f"ERROR: Summary file {file_path} not found."
        master_summary_content += error_message + "\n"
        print(error_message)
    except Exception as e:
        error_message = f"ERROR: Could not read {file_path}: {e}"
        master_summary_content += error_message + "\n"
        print(error_message)

with open(master_summary_file_path, "w", encoding="utf-8") as f:
    f.write(master_summary_content)

print(f"\nMaster summary file created at: {master_summary_file_path}")

print("\n--- Listing all generated CSV and TXT files in descriptive_stats_results --- ")
all_files_in_output_dir = []
for item in os.listdir(output_dir):
    if os.path.isfile(os.path.join(output_dir, item)) and (item.endswith(".csv") or item.endswith(".txt")):
        all_files_in_output_dir.append(item)
        print(item)

print("\n--- Summary File Creation Complete ---")

