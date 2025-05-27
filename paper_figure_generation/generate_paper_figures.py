import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_model_comparison(model_names, accuracy_scores, f1_scores, filename):
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, accuracy_scores, width, label="Accuracy")
    rects2 = ax.bar(x + width/2, f1_scores, width, label="Macro-F1 Score")

    ax.set_ylabel("Scores")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

# Data from predictive_modeling_summary.txt
# Logistic Regression
cm_lr = np.array([[3862, 0, 0],
                  [1301, 0, 0],
                  [1837, 0, 0]])
accuracy_lr = 0.5517
f1_lr = 0.2370

# Decision Tree
cm_dt = np.array([[3634, 2, 226],
                  [1227, 0, 74],
                  [1588, 0, 249]])
accuracy_dt = 0.5547
f1_dt = 0.3044

class_labels = ["FC", "FI", "NF"]
model_names = ["Logistic Regression", "Decision Tree"]
accuracy_scores = [accuracy_lr, accuracy_dt]
f1_scores = [f1_lr, f1_dt]

output_dir = "/home/ubuntu/latex_paper/figures"

# Create plots
plot_confusion_matrix(cm_lr, class_labels, "Confusion Matrix - Logistic Regression", f"{output_dir}/confusion_matrix_lr.png")
plot_confusion_matrix(cm_dt, class_labels, "Confusion Matrix - Decision Tree", f"{output_dir}/confusion_matrix_dt.png")
plot_model_comparison(model_names, accuracy_scores, f1_scores, f"{output_dir}/model_performance_comparison.png")

print("All plots generated.")

