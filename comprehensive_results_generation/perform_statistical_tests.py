import pandas as pd
import os
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import ttest_rel # For paired t-test
import numpy as np

# Define input and output directories
tsv_dir = "/home/ubuntu/zip_contents/"
output_dir = "/home/ubuntu/statistical_significance_results/"
os.makedirs(output_dir, exist_ok=True)

# Training data file
train_file_name = "Arabic LLMs Hallucination-OSACT2024-Train.txt"
train_file_path = os.path.join(tsv_dir, train_file_name)

results_log = "Statistical Significance Test Results\n\n"

print(f"--- Loading and Preprocessing Training Data for Significance Testing: {train_file_name} ---")
results_log += f"--- Loading and Preprocessing Training Data for Significance Testing: {train_file_name} ---\n"

def extract_pos_tag(word_pos_string):
    if pd.isna(word_pos_string):
        return "other"
    word_pos_string = str(word_pos_string)
    parts = word_pos_string.split("_")
    for part in parts:
        if part in ["noun", "verb", "adj"]:
            return part
    return "other"

try:
    df_train = pd.read_csv(train_file_path, sep=	'\t', engine='python', quoting=3)
    df_train["pos_extracted"] = df_train["word_pos"].apply(extract_pos_tag)
    df_train["readability"] = pd.to_numeric(df_train["readability"], errors='coerce')
    if df_train["readability"].isnull().any():
        median_readability = df_train["readability"].median()
        df_train["readability"].fillna(median_readability, inplace=True)
    df_train["readability"] = df_train["readability"].astype(int)
    df_train["model_normalized"] = df_train["model"].astype(str).str.lower().replace({"gpt4": "gpt-4", "chatgpt": "gpt-3.5"})
    
    feature_cols = ["readability", "pos_extracted", "model_normalized"]
    X = df_train[feature_cols].copy()
    y = df_train["label"].copy()

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    categorical_features = ["pos_extracted", "model_normalized"]
    preprocessor = ColumnTransformer(
        transformers=[
            (	'cat	', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    models_config = {
        "LogisticRegression": LogisticRegression(solver='liblinear', multi_class='auto', max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['accuracy', 'f1_macro']
    
    model_scores = {}

    for model_name, model_instance in models_config.items():
        print(f"--- Collecting scores for Model: {model_name} ---")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model_instance)])
        cv_results = cross_validate(pipeline, X, y_encoded, cv=cv, scoring=scoring_metrics, return_train_score=False)
        model_scores[model_name] = {
            'accuracy': cv_results['test_accuracy'],
            'f1_macro': cv_results['test_f1_macro']
        }
        results_log += f"Collected per-fold scores for {model_name}\n"
        results_log += f"  Accuracy scores: {cv_results['test_accuracy']}\n"
        results_log += f"  F1 Macro scores: {cv_results['test_f1_macro']}\n"

    # Perform paired t-tests
    lr_scores_acc = model_scores["LogisticRegression"]['accuracy']
    dt_scores_acc = model_scores["DecisionTree"]['accuracy']
    lr_scores_f1 = model_scores["LogisticRegression"]['f1_macro']
    dt_scores_f1 = model_scores["DecisionTree"]['f1_macro']

    results_log += "\n--- Paired T-test Results (Logistic Regression vs. Decision Tree) ---\n"
    print("\n--- Paired T-test Results (Logistic Regression vs. Decision Tree) ---")

    # Accuracy comparison
    t_stat_acc, p_value_acc = ttest_rel(lr_scores_acc, dt_scores_acc)
    results_log += f"Accuracy Comparison:\n"
    results_log += f"  Logistic Regression Mean Accuracy: {np.mean(lr_scores_acc):.4f}\n"
    results_log += f"  Decision Tree Mean Accuracy: {np.mean(dt_scores_acc):.4f}\n"
    results_log += f"  T-statistic: {t_stat_acc:.4f}, P-value: {p_value_acc:.4f}\n"
    print(f"Accuracy Comparison: T-statistic: {t_stat_acc:.4f}, P-value: {p_value_acc:.4f}")
    if p_value_acc < 0.05:
        results_log += "  Conclusion: There is a statistically significant difference in accuracy.\n"
    else:
        results_log += "  Conclusion: There is no statistically significant difference in accuracy.\n"

    # F1 Macro comparison
    t_stat_f1, p_value_f1 = ttest_rel(lr_scores_f1, dt_scores_f1)
    results_log += f"\nF1 Macro Score Comparison:\n"
    results_log += f"  Logistic Regression Mean F1 Macro: {np.mean(lr_scores_f1):.4f}\n"
    results_log += f"  Decision Tree Mean F1 Macro: {np.mean(dt_scores_f1):.4f}\n"
    results_log += f"  T-statistic: {t_stat_f1:.4f}, P-value: {p_value_f1:.4f}\n"
    print(f"F1 Macro Score Comparison: T-statistic: {t_stat_f1:.4f}, P-value: {p_value_f1:.4f}")
    if p_value_f1 < 0.05:
        results_log += "  Conclusion: There is a statistically significant difference in F1 Macro score.\n"
    else:
        results_log += "  Conclusion: There is no statistically significant difference in F1 Macro score.\n"

except FileNotFoundError:
    error_msg = f"ERROR: Training data file {train_file_path} not found."
    print(error_msg)
    results_log += error_msg + "\n"
except Exception as e:
    error_msg = f"ERROR: An error occurred: {e}"
    print(error_msg)
    results_log += error_msg + "\n"

# Save overall log
summary_file_path = os.path.join(output_dir, "statistical_significance_summary.txt")
with open(summary_file_path, "w", encoding="utf-8") as f:
    f.write(results_log)

print(f"\n--- Statistical Significance Testing Complete. Summary saved to {summary_file_path} ---")


