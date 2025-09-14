#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
# Configure font sizes for better readability
plt.rcParams.update({
    "font.size": 14,           # Base font size
    "axes.titlesize": 16,      # Title font size
    "axes.labelsize": 14,      # Axis label font size
    "xtick.labelsize": 12,     # X-axis tick label font size
    "ytick.labelsize": 12,     # Y-axis tick label font size
    "legend.fontsize": 12,     # Legend font size
    "figure.titlesize": 18,    # Figure title font size
    "axes.linewidth": 1.5,     # Axis line width
    "grid.linewidth": 0.8,     # Grid line width
    "lines.linewidth": 2.0,    # Line width
    "patch.linewidth": 1.0     # Patch line width
})
plt.style.use('default')
sns.set_palette("husl")

# Define directories
feature_dir = "../feature_engineered_data/"
output_dir = "../ml_analysis_results/"
plots_dir = "../ml_analysis_plots/"

# Create output directories
import os
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the engineered features data"""
    print("=== Loading and Preparing Data ===")
    
    # Load engineered features
    train_df = pd.read_csv(os.path.join(feature_dir, "train_engineered_features.csv"))
    dev_df = pd.read_csv(os.path.join(feature_dir, "dev_engineered_features.csv"))
    
    # Combine train and dev for better training
    combined_df = pd.concat([train_df, dev_df], ignore_index=True)
    
    # Separate features and labels
    feature_columns = [col for col in combined_df.columns if not col.startswith('label_') and col != 'claim_id']
    X = combined_df[feature_columns]
    
    # Create target variable (multiclass)
    label_col = combined_df[['label_fc', 'label_nf', 'label_fi']].idxmax(axis=1)
    label_mapping = {'label_fc': 'FC', 'label_nf': 'NF', 'label_fi': 'FI'}
    y = label_col.map(label_mapping)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {pd.Series(y).value_counts()}")
    print(f"Features used: {len(feature_columns)}")
    
    return X, y_encoded, le, feature_columns

def perform_variation_analysis(X, y, feature_columns):
    """Perform variation analysis on features"""
    print("\n=== Performing Variation Analysis ===")
    
    # Calculate coefficient of variation for each feature
    cv_results = {}
    for col in feature_columns:
        mean_val = X[col].mean()
        std_val = X[col].std()
        cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
        cv_results[col] = cv
    
    # Sort by coefficient of variation
    cv_sorted = sorted(cv_results.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 10 features with highest variation:")
    for feature, cv in cv_sorted[:10]:
        print(f"  {feature}: {cv:.2f}%")
    
    print("\nBottom 10 features with lowest variation:")
    for feature, cv in cv_sorted[-10:]:
        print(f"  {feature}: {cv:.2f}%")
    
    # Create variation analysis plot
    plt.figure(figsize=(15, 8))
    features = [x[0] for x in cv_sorted]
    cv_values = [x[1] for x in cv_sorted]
    
    plt.bar(range(len(features)), cv_values, alpha=0.7)
    plt.title('Feature Variation Analysis (Coefficient of Variation)')
    plt.xlabel('Features')
    plt.ylabel('Coefficient of Variation (%)')
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_variation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save variation analysis results
    variation_df = pd.DataFrame(cv_sorted, columns=['Feature', 'Coefficient_of_Variation'])
    variation_df.to_csv(os.path.join(output_dir, 'variation_analysis.csv'), index=False)
    
    return cv_sorted

def train_models(X, y, feature_columns):
    """Train multiple ML models"""
    print("\n=== Training ML Models ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models (including 3 new ones)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        if name in ['SVM', 'Neural Network']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_test': y_test,
            'model': model
        }
        
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results, X_test, y_test

def create_evaluation_plots(results, output_dir, plots_dir):
    """Create comprehensive evaluation plots"""
    print("\n=== Creating Evaluation Plots ===")
    
    # 1. Model Performance Comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in results.keys()]
        axes[i].bar(results.keys(), values, alpha=0.7, color=sns.color_palette("husl", len(results)))
        axes[i].set_title(f'{metric.capitalize()} Score Comparison')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cross-validation Results
    plt.figure(figsize=(12, 6))
    cv_means = [results[name]['cv_mean'] for name in results.keys()]
    cv_stds = [results[name]['cv_std'] for name in results.keys()]
    
    x_pos = np.arange(len(results.keys()))
    plt.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color=sns.color_palette("husl", len(results)))
    plt.xlabel('Models')
    plt.ylabel('Cross-validation Accuracy')
    plt.title('Cross-validation Results (5-fold)')
    plt.xticks(x_pos, results.keys(), rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
        plt.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cross_validation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrices
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()
    
    for i, (name, result) in enumerate(results.items()):
        if i < 9:  # Limit to 9 plots
            cm = confusion_matrix(result['y_test'], result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Model Ranking
    plt.figure(figsize=(12, 8))
    
    # Create ranking based on F1 score
    model_ranking = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    model_names = [name for name, _ in model_ranking]
    f1_scores = [result['f1'] for _, result in model_ranking]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = plt.bar(range(len(model_names)), f1_scores, color=colors, alpha=0.7)
    
    plt.xlabel('Models (Ranked by F1 Score)')
    plt.ylabel('F1 Score')
    plt.title('Model Ranking by F1 Score')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Performance Heatmap
    plt.figure(figsize=(10, 8))
    
    # Create performance matrix
    performance_data = []
    for name in results.keys():
        performance_data.append([
            results[name]['accuracy'],
            results[name]['precision'],
            results[name]['recall'],
            results[name]['f1'],
            results[name]['cv_mean']
        ])
    
    performance_df = pd.DataFrame(performance_data, 
                                index=results.keys(),
                                columns=['Accuracy', 'Precision', 'Recall', 'F1', 'CV_Accuracy'])
    
    sns.heatmap(performance_df, annot=True, cmap='YlOrRd', fmt='.3f', cbar_kws={'label': 'Score'})
    plt.title('Model Performance Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_results(results, output_dir):
    """Save detailed results to CSV files"""
    print("\n=== Saving Detailed Results ===")
    
    # Create summary dataframe
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1'],
            'CV_Mean': result['cv_mean'],
            'CV_Std': result['cv_std']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('F1_Score', ascending=False)
    summary_df.to_csv(os.path.join(output_dir, 'model_performance_summary.csv'), index=False)
    
    # Save detailed classification reports
    with open(os.path.join(output_dir, 'detailed_classification_reports.txt'), 'w') as f:
        f.write("DETAILED CLASSIFICATION REPORTS\n")
        f.write("=" * 50 + "\n\n")
        
        for name, result in results.items():
            f.write(f"\n{name.upper()}\n")
            f.write("-" * 30 + "\n")
            f.write(classification_report(result['y_test'], result['y_pred']))
            f.write("\n" + "=" * 50 + "\n")
    
    print(f"Results saved to: {output_dir}")

def main():
    """Main execution function"""
    print("=== Advanced ML Analysis with Variation Analysis ===")
    
    # Load and prepare data
    X, y, le, feature_columns = load_and_prepare_data()
    
    # Perform variation analysis
    variation_results = perform_variation_analysis(X, y, feature_columns)
    
    # Train models
    results, X_test, y_test = train_models(X, y, feature_columns)
    
    # Create evaluation plots
    create_evaluation_plots(results, output_dir, plots_dir)
    
    # Save detailed results
    save_detailed_results(results, output_dir)
    
    # Print final summary
    print("\n=== FINAL SUMMARY ===")
    print("Top 3 models by F1 score:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for i, (name, result) in enumerate(sorted_results[:3]):
        print(f"{i+1}. {name}: F1={result['f1']:.4f}, Accuracy={result['accuracy']:.4f}")
    
    print(f"\nAll results and plots saved to:")
    print(f"  Results: {output_dir}")
    print(f"  Plots: {plots_dir}")

if __name__ == "__main__":
    main() 