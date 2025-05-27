import pandas as pd
import os
import re

# Define input and output directories
tsv_dir = "/home/ubuntu/zip_contents/"
output_dir = "/home/ubuntu/self_consistency_results/"
os.makedirs(output_dir, exist_ok=True)

# Test data file
test_file_name = "Arabic LLMs Hallucination-OSACT2024-Test.txt"
test_file_path = os.path.join(tsv_dir, test_file_name)

results_log_content = "Self-Consistency Detector Evaluation\n\n"

# List of Arabic hedging/uncertainty expressions
# Based on common phrases and terms identified from research (e.g., ربما, أعتقد, يبدو أن, ظن, حسب, خال, زعم, إن الشرطية)
# For simplicity, we will use a list of common phrases. More sophisticated detection might involve morphological analysis.
hedging_phrases_arabic = [
    "ربما",        # perhaps
    "أعتقد",       # I believe/think
    "يبدو أن",     # it seems that
    "لعل",         # perhaps, maybe
    "من المحتمل",  # it is probable/likely
    "يمكن",        # it is possible / can
    "يبدو",        # it seems
    "كأن",         # as if / it seems
    "أظن",         # I think (form of ظنّ)
    "أحسب",        # I reckon (form of حسب)
    "أخال",        # I imagine (form of خال)
    "أزعم",        # I claim (form of زعم)
    "إن كان",      # if it is (conditional إنْ)
    "قد يكون",     # it may be (قد + verb)
    "من الممكن",   # it is possible
    "على ما يبدو", # apparently
    "في رأيي",     # in my opinion
    "حسب اعتقادي" # according to my belief
]

print(f"--- Loading Test Data: {test_file_name} ---")
results_log_content += f"--- Loading Test Data: {test_file_name} ---\n"

try:
    df_test = pd.read_csv(test_file_path, sep=	'\t', engine='python', quoting=3)
    flagged_texts_info = []
    num_flagged_texts = 0

    print("--- Applying Self-Consistency Detector (Hedging Phrase Detection) ---")
    results_log_content += "--- Applying Self-Consistency Detector (Hedging Phrase Detection) ---\n"
    results_log_content += f"Using hedging phrases: {', '.join(hedging_phrases_arabic)}\n"

    for index, row in df_test.iterrows():
        text_content = str(row["claim"]) # Ensure text is string
        found_hedges = []
        is_flagged = False
        for phrase in hedging_phrases_arabic:
            # Simple substring check. More advanced would use regex with word boundaries for some phrases.
            if phrase in text_content:
                found_hedges.append(phrase)
                is_flagged = True
        
        if is_flagged:
            num_flagged_texts += 1
            flagged_texts_info.append({
                "id": row["claim_id"],
                "text": text_content,
                "detected_hedges": ", ".join(found_hedges)
            })

    total_texts = len(df_test)
    percentage_flagged = (num_flagged_texts / total_texts) * 100 if total_texts > 0 else 0

    results_log_content += f"Total texts in test set: {total_texts}\n"
    results_log_content += f"Number of texts flagged by the detector: {num_flagged_texts}\n"
    results_log_content += f"Percentage of texts flagged: {percentage_flagged:.2f}%\n\n"

    print(f"Total texts in test set: {total_texts}")
    print(f"Number of texts flagged by the detector: {num_flagged_texts}")
    print(f"Percentage of texts flagged: {percentage_flagged:.2f}%")

    # Save flagged texts examples to a CSV file
    df_flagged_texts = pd.DataFrame(flagged_texts_info)
    flagged_texts_csv_path = os.path.join(output_dir, "self_consistency_flagged_texts.csv")
    df_flagged_texts.to_csv(flagged_texts_csv_path, index=False, encoding=		'utf-8-sig')
    results_log_content += f"Examples of flagged texts saved to: {flagged_texts_csv_path}\n"
    print(f"Examples of flagged texts saved to: {flagged_texts_csv_path}")

    # Add some examples to the log file
    results_log_content += "\n--- Examples of Flagged Texts (First 5) ---\n"
    for i, item in enumerate(flagged_texts_info[:5]):
        results_log_content += f"ID: {item['id']}\nText: {item['text']}\nDetected Hedges: {item['detected_hedges']}\n---\n"

except FileNotFoundError:
    error_msg = f"ERROR: Test data file {test_file_path} not found."
    print(error_msg)
    results_log_content += error_msg + "\n"
except Exception as e:
    error_msg = f"ERROR: An error occurred during self-consistency detection: {e}"
    print(error_msg)
    results_log_content += error_msg + "\n"

# Save overall log
summary_file_path = os.path.join(output_dir, "self_consistency_detector_summary.txt")
with open(summary_file_path, "w", encoding="utf-8") as f:
    f.write(results_log_content)

print(f"\n--- Self-Consistency Detector Evaluation Complete. Summary saved to {summary_file_path} ---")


