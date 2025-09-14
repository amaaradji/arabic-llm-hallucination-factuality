import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Define input and output directories
tsv_dir = "/home/ubuntu/zip_contents/"
output_dir = "/home/ubuntu/feature_analysis_results/"
os.makedirs(output_dir, exist_ok=True)

def extract_pos_tag(word_pos_string):
    if pd.isna(word_pos_string):
        return "other"
    word_pos_string = str(word_pos_string)
    parts = word_pos_string.split("_")
    for part in parts:
        if part in ["noun", "verb", "adj"]:
            return part
    return "other"

def plot_feature_importance(importance_df, title, filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df.sort_values('importance', ascending=False))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def analyze_feature_importance():
    print("--- Starting Feature Importance Analysis ---")
    
    # Load and preprocess data
    train_file_path = os.path.join(tsv_dir, "Arabic LLMs Hallucination-OSACT2024-Train.txt")
    df = pd.read_csv(train_file_path, sep='\t', engine='python', quoting=3)
    
    # Preprocess features
    df["pos_extracted"] = df["word_pos"].apply(extract_pos_tag)
    df["readability"] = pd.to_numeric(df["readability"], errors='coerce').fillna(df["readability"].median()).astype(int)
    df["model_normalized"] = df["model"].astype(str).str.lower().replace({"gpt4": "gpt-4", "chatgpt": "gpt-3.5"})
    
    # Define features and target
    feature_cols = ["readability", "pos_extracted", "model_normalized"]
    X = df[feature_cols].copy()
    y = df["label"].copy()
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Define preprocessing
    categorical_features = ["pos_extracted", "model_normalized"]
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Define models
    models = {
        "LogisticRegression": LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nAnalyzing feature importance for {model_name}")
        
        # Create and fit pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        
        # 1. Model-specific feature importance
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            model_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            plot_feature_importance(model_importance, 
                                  f'{model_name} Feature Importance', 
                                  f'{model_name}_feature_importance.png')
            results[f'{model_name}_model_importance'] = model_importance
        
        # 2. Permutation Importance
        result = permutation_importance(pipeline, X_test, y_test, 
                                      n_repeats=10, random_state=42)
        
        # Get feature names after preprocessing
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        
        perm_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': result.importances_mean,
            'std': result.importances_std
        })
        
        plot_feature_importance(perm_importance, 
                              f'{model_name} Permutation Importance', 
                              f'{model_name}_permutation_importance.png')
        
        results[f'{model_name}_permutation_importance'] = perm_importance
        
        # Save results to CSV
        perm_importance.to_csv(os.path.join(output_dir, f'{model_name}_permutation_importance.csv'))
        if f'{model_name}_model_importance' in results:
            results[f'{model_name}_model_importance'].to_csv(
                os.path.join(output_dir, f'{model_name}_feature_importance.csv'))
    
    # Generate summary report
    summary = "Feature Importance Analysis Summary\n\n"
    for model_name in models.keys():
        summary += f"\n{model_name}:\n"
        if f'{model_name}_model_importance' in results:
            summary += "Model-specific Feature Importance:\n"
            summary += results[f'{model_name}_model_importance'].to_string() + "\n\n"
        summary += "Permutation Importance:\n"
        summary += results[f'{model_name}_permutation_importance'].to_string() + "\n\n"
    
    # Save summary
    with open(os.path.join(output_dir, 'feature_importance_summary.txt'), 'w') as f:
        f.write(summary)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    analyze_feature_importance() 