import os
import csv
from openai import AzureOpenAI
import time

# --- Configuration ---
AZURE_OPENAI_API_KEY = "XXX"
AZURE_OPENAI_ENDPOINT = "https://o3miniapi.openai.azure.com/"
API_VERSION = "2024-12-01-preview"
MODEL_NAME = "o3-mini"
DEPLOYMENT_NAME = "o3-mini"

# --- Initialize Azure OpenAI Client ---
client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

def judge_claim(claim_text: str, temperature_setting: float) -> str:
    """Judges a single claim using the Azure OpenAI API with a specific temperature."""
    try:
        system_prompt = "You are an AI assistant. Your task is to determine if the following Arabic statement is factually correct or incorrect. Respond with only 'Factually Correct' or 'Factually Incorrect'."
        user_prompt = f"Statement: {claim_text}"

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature_setting,
            max_completion_tokens=20,  # Increased slightly just in case of longer standard responses
            n=1
        )
        judgment = response.choices[0].message.content.strip()
        if "Factually Correct" in judgment:
            return "Factually Correct"
        elif "Factually Incorrect" in judgment:
            return "Factually Incorrect"
        else:
            print(f"Warning: Unexpected judgment for claim '{claim_text}' with temp {temperature_setting}: {judgment}")
            return "Unknown"
    except Exception as e:
        print(f"Error judging claim '{claim_text}' with temp {temperature_setting}: {e}")
        # Check for specific API errors if needed, e.g., rate limits, auth
        if "429" in str(e) or "Too Many Requests" in str(e):
            print("Rate limit likely hit, sleeping for 60 seconds...")
            time.sleep(60)
            return "Retry"
        return "Error"

def check_self_consistency(claim_text: str, temp1: float = 0.2, temp2: float = 0.8) -> tuple[str, str, str]:
    """
    Checks a claim for self-consistency by judging it twice with different temperatures.
    Returns the judgments from both temperatures and the self-consistency result.
    """
    print(f"Processing claim: {claim_text}")
    judgment1 = judge_claim(claim_text, temp1)
    if judgment1 == "Retry": # Handle retry for first judgment
        judgment1 = judge_claim(claim_text, temp1)

    time.sleep(1) # Small delay between API calls to help with potential rate limits
    
    judgment2 = judge_claim(claim_text, temp2)
    if judgment2 == "Retry": # Handle retry for second judgment
        judgment2 = judge_claim(claim_text, temp2)

    consistency_result = "Error/Unknown"
    if judgment1 in ["Error", "Unknown"] or judgment2 in ["Error", "Unknown"]:
        consistency_result = "Error/Unknown"
    elif judgment1 == judgment2:
        consistency_result = "Consistent"
    else:
        consistency_result = "Potentially Hallucinated (Inconsistent)"
    
    print(f"  Temp {temp1}: {judgment1}")
    print(f"  Temp {temp2}: {judgment2}")
    print(f"  Result: {consistency_result}")
    return judgment1, judgment2, consistency_result

def load_claims_from_file(file_path: str, limit: int = None):
    """Loads claims from a tab-separated file."""
    claims_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            claims_data.append({'claim_id': row['claim_id'], 'claim_text': row['claim'], 'gold_label': row['label']})
    return claims_data

# --- Main execution for benchmarking ---
if __name__ == "__main__":
    # Using a small subset of the dev set for quick testing
    # dev_file_path = "/home/ubuntu/data/Arabic LLMs Hallucination-OSACT2024-Dev.txt"
    # For a more robust test, use the full dev or train set, but be mindful of API costs/limits
    # For this run, let's use a very small number of claims to test the pipeline
    dataset_file_path = "/home/ubuntu/data/Arabic LLMs Hallucination-OSACT2024-Dev.txt"
    claims_to_process = load_claims_from_file(dataset_file_path, limit=10) # Process first 10 claims for this test

    benchmark_results = []
    true_positives = 0 # Detected as Inconsistent, Gold is FI
    false_positives = 0 # Detected as Inconsistent, Gold is FC (or NF treated as not hallucinated)
    true_negatives = 0 # Detected as Consistent, Gold is FC (or NF)
    false_negatives = 0 # Detected as Consistent, Gold is FI
    error_count = 0

    for item in claims_to_process:
        claim_text = item['claim_text']
        gold_label = item['gold_label']
        
        j1, j2, cons_res = check_self_consistency(claim_text)
        
        benchmark_results.append({
            "claim_id": item['claim_id'],
            "claim": claim_text,
            "gold_label": gold_label,
            "judgment_temp1": j1,
            "judgment_temp2": j2,
            "consistency_result": cons_res
        })
        print("---")

        # Basic benchmarking logic:
        # We consider 'Potentially Hallucinated (Inconsistent)' as a positive detection for hallucination.
        # We consider 'FI' (Factually Incorrect) as the ground truth for hallucination.
        # 'FC' (Factually Correct) and 'NF' (Not Factual) are considered non-hallucinated for this simple benchmark.
        
        is_hallucinated_by_detector = (cons_res == "Potentially Hallucinated (Inconsistent)")
        is_truly_hallucinated = (gold_label == "FI")

        if cons_res == "Error/Unknown":
            error_count += 1
            continue # Skip metric calculation for errors

        if is_hallucinated_by_detector and is_truly_hallucinated:
            true_positives += 1
        elif is_hallucinated_by_detector and not is_truly_hallucinated:
            false_positives += 1
        elif not is_hallucinated_by_detector and not is_truly_hallucinated:
            true_negatives += 1
        elif not is_hallucinated_by_detector and is_truly_hallucinated:
            false_negatives += 1

    print("\n--- Benchmark Results Summary ---")
    for res in benchmark_results:
        print(res)
    
    print("\n--- Performance Metrics (Simple) ---")
    print(f"Claims processed (excluding errors): {len(claims_to_process) - error_count}")
    print(f"Errors/Unknowns from detector: {error_count}")
    print(f"True Positives (Detected FI as Inconsistent): {true_positives}")
    print(f"False Positives (Detected FC/NF as Inconsistent): {false_positives}")
    print(f"True Negatives (Detected FC/NF as Consistent): {true_negatives}")
    print(f"False Negatives (Detected FI as Consistent): {false_negatives}")

    # Calculate precision, recall, F1 (handle division by zero)
    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"Precision: {precision:.4f}")
    else:
        print("Precision: N/A (no positive detections)")

    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"Recall: {recall:.4f}")
    else:
        print("Recall: N/A (no actual positives in sample or all missed)")

    if precision > 0 and recall > 0: # Ensure precision and recall are valid before F1
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"F1-Score: {f1_score:.4f}")
    else:
        print("F1-Score: N/A")

    output_file_path = "/home/ubuntu/data/benchmark_self_consistency_results.txt"
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        outfile.write("--- Benchmark Results Summary ---\n")
        for res in benchmark_results:
            outfile.write(str(res) + "\n")
        outfile.write("\n--- Performance Metrics (Simple) ---\n")
        outfile.write(f"Claims processed (excluding errors): {len(claims_to_process) - error_count}\n")
        outfile.write(f"Errors/Unknowns from detector: {error_count}\n")
        outfile.write(f"True Positives (Detected FI as Inconsistent): {true_positives}\n")
        outfile.write(f"False Positives (Detected FC/NF as Inconsistent): {false_positives}\n")
        outfile.write(f"True Negatives (Detected FC/NF as Consistent): {true_negatives}\n")
        outfile.write(f"False Negatives (Detected FI as Consistent): {false_negatives}\n")
        if (true_positives + false_positives) > 0:
            outfile.write(f"Precision: {precision:.4f}\n")
        else:
            outfile.write("Precision: N/A (no positive detections)\n")
        if (true_positives + false_negatives) > 0:
            outfile.write(f"Recall: {recall:.4f}\n")
        else:
            outfile.write("Recall: N/A (no actual positives in sample or all missed)\n")
        if precision > 0 and recall > 0:
            outfile.write(f"F1-Score: {f1_score:.4f}\n")
        else:
            outfile.write("F1-Score: N/A\n")
    print(f"\nBenchmark results saved to {output_file_path}")


