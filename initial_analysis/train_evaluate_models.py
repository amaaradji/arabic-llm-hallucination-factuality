import pandas as pd
import os
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

# Define input and output directories
tsv_dir = "/home/ubuntu/zip_contents/"
output_dir = "/home/ubuntu/predictive_modeling_results/"
os.makedirs(output_dir, exist_ok=True)

# Training data file
train_file_name = "Arabic LLMs Hallucination-OSACT2024-Train.txt"
train_file_path = os.path.join(tsv_dir, train_file_name)

results_log = "Predictive Modeling Results\n\n"

print(f"--- Loading and Preprocessing Training Data: {train_file_name} ---")
results_log += f"--- Loading and Preprocessing Training Data: {train_file_name} ---\n"

def extract_pos_tag(word_pos_string):
    if pd.isna(word_pos_string):
        return "other" # Default if NaN
    word_pos_string = str(word_pos_string)
    parts = word_pos_string.split("_")
    for part in parts:
        if part in ["noun", "verb", "adj"]:
            return part
    return "other"

try:
    # Corrected pd.read_csv call
    df_train = pd.read_csv(train_file_path, sep='\t', engine='python', quoting=3)

    # Preprocessing
    df_train["pos_extracted"] = df_train["word_pos"].apply(extract_pos_tag)
    df_train["readability"] = pd.to_numeric(df_train["readability"], errors='coerce')
    # Fill NaN readability with a common value or drop, for now fill with median or mode if any exist
    if df_train["readability"].isnull().any():
        median_readability = df_train["readability"].median()
        df_train["readability"].fillna(median_readability, inplace=True)
        results_log += f"Filled NaN readability values with median: {median_readability}\n"
    df_train["readability"] = df_train["readability"].astype(int)

    df_train["model_normalized"] = df_train["model"].astype(str).str.lower().replace({"gpt4": "gpt-4", "chatgpt": "gpt-3.5"})
    
    # Define features (X) and target (y)
    # Predictors: {readability, POS, model}
    feature_cols = ["readability", "pos_extracted", "model_normalized"]
    X = df_train[feature_cols].copy()
    y = df_train["label"].copy()

    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    target_labels = le_target.classes_
    results_log += f"Target labels: {target_labels}\n"

    # Preprocessing for features: OneHotEncode categorical features
    # Readability can be treated as is, or binned/scaled if desired. For now, as is.
    categorical_features = ["pos_extracted", "model_normalized"]
    numerical_features = ["readability"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            # ('num', 'passthrough', numerical_features) # readability is already 1-4, can be passthrough or scaled
        ],
        remainder='passthrough' # Keep readability as is
    )

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(solver='liblinear', multi_class='auto', max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['accuracy', 'f1_macro']

    for model_name, model_instance in models.items():
        print(f"\n--- Evaluating Model: {model_name} ---")
        results_log += f"\n--- Evaluating Model: {model_name} ---\n"

        # Create pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model_instance)])

        # Perform cross-validation
        cv_results = cross_validate(pipeline, X, y_encoded, cv=cv, scoring=scoring_metrics, return_train_score=False, return_estimator=True)
        
        avg_accuracy = np.mean(cv_results['test_accuracy'])
        avg_f1_macro = np.mean(cv_results['test_f1_macro'])

        results_log += f"Average Accuracy: {avg_accuracy:.4f}\n"
        results_log += f"Average Macro-F1 Score: {avg_f1_macro:.4f}\n"
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average Macro-F1 Score: {avg_f1_macro:.4f}")

        # Aggregate confusion matrices
        # Get predictions for each fold to build confusion matrix
        all_y_true = []
        all_y_pred = []
        for train_idx, test_idx in cv.split(X, y_encoded):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y_encoded[train_idx], y_encoded[test_idx]
            
            pipeline.fit(X_train_fold, y_train_fold)
            y_pred_fold = pipeline.predict(X_test_fold)
            
            all_y_true.extend(y_test_fold)
            all_y_pred.extend(y_pred_fold)
            
        cm = confusion_matrix(all_y_true, all_y_pred, labels=np.arange(len(target_labels)))
        results_log += f"Overall Confusion Matrix (Labels: {target_labels}):\n{cm}\n"
        print(f"Overall Confusion Matrix (Labels: {target_labels}):\n{cm}")

        # Save results
        with open(os.path.join(output_dir, f"{model_name}_metrics.txt"), "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Average Macro-F1 Score: {avg_f1_macro:.4f}\n")
            f.write(f"Overall Confusion Matrix (Labels: {target_labels}):\n{cm}\n")
        np.savetxt(os.path.join(output_dir, f"{model_name}_confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
        results_log += f"Metrics saved to {output_dir}{model_name}_metrics.txt\n"
        results_log += f"Confusion matrix saved to {output_dir}{model_name}_confusion_matrix.csv\n"

except FileNotFoundError:
    error_msg = f"ERROR: Training data file {train_file_path} not found."
    print(error_msg)
    results_log += error_msg + "\n"
except Exception as e:
    error_msg = f"ERROR: An error occurred during model training/evaluation: {e}"
    print(error_msg)
    results_log += error_msg + "\n"

# Save overall log
with open(os.path.join(output_dir, "predictive_modeling_summary.txt"), "w") as f:
    f.write(results_log)

print(f"\n--- Predictive Modeling Complete. Summary saved to {output_dir}predictive_modeling_summary.txt ---")


