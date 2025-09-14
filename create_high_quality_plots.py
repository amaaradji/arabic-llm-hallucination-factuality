#!/usr/bin/env python
"""
Script to create high-quality plots with large, readable fonts
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import os

# Set high-quality plotting parameters
matplotlib.rcParams.update({
    'font.size': 18,           # Base font size
    'axes.titlesize': 20,      # Title font size
    'axes.labelsize': 18,      # Axis label font size
    'xtick.labelsize': 16,     # X-axis tick label font size
    'ytick.labelsize': 16,     # Y-axis tick label font size
    'legend.fontsize': 16,     # Legend font size
    'figure.titlesize': 22,    # Figure title font size
    'axes.linewidth': 2.0,     # Axis line width
    'grid.linewidth': 1.2,     # Grid line width
    'lines.linewidth': 3.0,    # Line width
    'patch.linewidth': 2.0,    # Patch line width
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})

def create_sample_plots():
    """Create sample high-quality plots"""
    
    # Create output directory
    output_dir = "high_quality_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sample correlation heatmap
    print("Creating correlation heatmap...")
    np.random.seed(42)
    data = np.random.randn(10, 10)
    corr_matrix = np.corrcoef(data)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix', fontweight='bold', pad=20)
    plt.xlabel('Features', fontweight='bold')
    plt.ylabel('Features', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap_sample.png")
    plt.close()
    
    # 2. Sample model performance comparison
    print("Creating model performance comparison...")
    models = ['Logistic Regression', 'Random Forest', 'SVM', 'Neural Network', 'Naive Bayes']
    accuracy = [0.85, 0.88, 0.82, 0.87, 0.79]
    f1_score = [0.83, 0.86, 0.80, 0.85, 0.77]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, f1_score, width, label='F1-Score', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance Comparison', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=14)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_performance_sample.png")
    plt.close()
    
    # 3. Sample confusion matrix
    print("Creating confusion matrix...")
    from sklearn.metrics import confusion_matrix
    
    # Sample data
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    
    cm = confusion_matrix(y_true, y_pred)
    classes = ['FC', 'NF', 'FI']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'shrink': 0.8})
    plt.title('Confusion Matrix - Sample Model', fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_sample.png")
    plt.close()
    
    print(f"\n✓ Sample plots created in {output_dir}/")
    print("These plots demonstrate the improved font sizes and quality.")

if __name__ == "__main__":
    create_sample_plots()
