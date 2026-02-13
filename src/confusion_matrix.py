# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,recall_score

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

    print(f"--- {title} Report ---")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy Score: {accuracy_score(y_true, y_pred):.6f}")
    recall = recall_score(y_true, y_pred)
    print(f"We identified {recall * 100:.2f}% of all fraudulent transactions.")