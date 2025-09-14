import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Define input and output directories
input_dir = "/home/ubuntu/comprehensive_results_workdir/output_data/descriptive_stats_results/"
plots_output_dir = os.path.join(input_dir, "plots/")
tables_output_dir = os.path.join(input_dir, "tables/")

# Create output directories if they don_t exist
os.makedirs(plots_output_dir, exist_ok=True)
os.makedirs(tables_output_dir, exist_ok=True)

# --- 1. Generate Bar Plots for Hallucination Rate by Readability --- 
print("--- Generating Bar Plots for Hallucination Rate by Readability ---")
readability_files = [
    "Train_hallucination_rate_by_readability.csv",
    "Dev_hallucination_rate_by_readability.csv",
    "Test_hallucination_rate_by_readability.csv"
]

for file_name in readability_files:
    dataset_name = file_name.split("_")[0]
    file_path = os.path.join(input_dir, file_name)
    try:
        df_readability = pd.read_csv(file_path)
        if df_readability.empty or "Readability_Level" not in df_readability.columns or "rate" not in df_readability.columns:
            print(f"Skipping plot for {file_name} due to missing columns or empty data.")
            continue

        plt.figure(figsize=(8, 6))
        sns.barplot(x="Readability_Level", y="rate", data=df_readability, palette="viridis")
        plt.title(f"Hallucination Rate by Readability Level - {dataset_name} Set")
        plt.xlabel("Readability Level")
        plt.ylabel("Hallucination Rate (%)")
        plt.ylim(0, 100)
        plot_path = os.path.join(plots_output_dir, f"{dataset_name}_hallucination_rate_by_readability.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved: {plot_path}")
    except Exception as e:
        print(f"Error generating plot for {file_name}: {e}")

# --- 2. Generate Bar Plots for Hallucination Rate by POS --- 
print("\n--- Generating Bar Plots for Hallucination Rate by POS ---")
pos_files = [
    "Train_hallucination_rate_by_pos.csv",
    "Dev_hallucination_rate_by_pos.csv",
    "Test_hallucination_rate_by_pos.csv"
]

for file_name in pos_files:
    dataset_name = file_name.split("_")[0]
    file_path = os.path.join(input_dir, file_name)
    try:
        df_pos = pd.read_csv(file_path)
        if df_pos.empty or "POS_Tag" not in df_pos.columns or "rate" not in df_pos.columns:
            print(f"Skipping plot for {file_name} due to missing columns or empty data.")
            continue
        
        # Filter for specific POS tags if needed, e.g., noun, verb, adj
        # df_pos = df_pos[df_pos["POS_Tag"].isin(["noun", "verb", "adj"])]

        plt.figure(figsize=(8, 6))
        sns.barplot(x="POS_Tag", y="rate", data=df_pos, palette="mako")
        plt.title(f"Hallucination Rate by POS Tag - {dataset_name} Set")
        plt.xlabel("POS Tag")
        plt.ylabel("Hallucination Rate (%)")
        plt.ylim(0, 100)
        plot_path = os.path.join(plots_output_dir, f"{dataset_name}_hallucination_rate_by_pos.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved: {plot_path}")
    except Exception as e:
        print(f"Error generating plot for {file_name}: {e}")

# --- 3. Produce Tables for Descriptive Statistics --- 
# The CSV files already serve as tables. We can create formatted versions if needed.
# For now, we will just note that the CSVs are the primary tables.
# Let_s create a summary table from the cross-tabulation data for one dataset as an example.
print("\n--- Producing Tables for Descriptive Statistics (Example from Cross-Tabulation) ---")
cross_tab_file = "Train_cross_tab_readability_pos_model.csv" # Example with Train set
file_path = os.path.join(input_dir, cross_tab_file)

try:
    df_cross_tab = pd.read_csv(file_path)
    if not df_cross_tab.empty:
        # Example: Pivot table for better readability
        # Hallucination Rate
        pivot_rate = df_cross_tab.pivot_table(index=["Readability", "POS_Tag"], columns="Model", values="Hallucination_Rate")
        pivot_rate_path = os.path.join(tables_output_dir, "Train_cross_tab_hallucination_rate_pivot.csv")
        pivot_rate.to_csv(pivot_rate_path)
        print(f"Pivot table (Rate) saved: {pivot_rate_path}")
        
        # Hallucinated Count
        pivot_count = df_cross_tab.pivot_table(index=["Readability", "POS_Tag"], columns="Model", values="Hallucinated_Count", aggfunc="sum") # Summing counts
        pivot_count_path = os.path.join(tables_output_dir, "Train_cross_tab_hallucinated_count_pivot.csv")
        pivot_count.to_csv(pivot_count_path)
        print(f"Pivot table (Count) saved: {pivot_count_path}")

        # Save a markdown version for easy viewing
        md_table_path = os.path.join(tables_output_dir, "Train_cross_tab_hallucination_rate_pivot.md")
        with open(md_table_path, "w") as f:
            f.write(pivot_rate.to_markdown())
        print(f"Markdown table (Rate) saved: {md_table_path}")

    else:
        print(f"Skipping table generation for {cross_tab_file} as it is empty.")
except FileNotFoundError:
    print(f"File not found: {file_path}. Skipping table generation.")
except Exception as e:
    print(f"Error producing table for {cross_tab_file}: {e}")

print("\n--- Visualization and Table Generation Complete ---")

