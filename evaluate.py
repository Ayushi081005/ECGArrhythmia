import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    roc_curve, 
    auc,
    roc_auc_score,
    precision_recall_fscore_support,
    cohen_kappa_score
)
from sklearn.preprocessing import label_binarize

# Import from our other .py files
import config
from model import CNN_LSTM
from preprocess import get_test_data_for_eval # Use the specific test data loader

def evaluate_model():
    print("\n" + "="*70)
    print("FINAL MODEL EVALUATION")
    print("="*70)
    
    # 1. Load test data and scaler
    # Note: We load X_test, y_test separately for sklearn metrics
    test_loader, _, y_true = get_test_data_for_eval()
    # Ensure y_true is a numpy array
    y_true = np.array(y_true)
    
    if test_loader is None:
        return # Error loading scaler, stop evaluation

    # 2. Initialize and load best model
    model = CNN_LSTM().to(config.DEVICE)
    try:
        # Load model with map_location so CPU-only machines work
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_PATH}.")
        print("Please run train.py first to create and save the model.")
        return
        
    model.eval()
    
    # 3. Get predictions
    y_pred_list = []
    y_probs_list = []
    
    print("Running model on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            y_pred_list.extend(predicted.cpu().numpy())
            y_probs_list.extend(probabilities.cpu().numpy())

    y_pred = np.array(y_pred_list)
    y_probs = np.array(y_probs_list)
    
    class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unclassifiable']
    
    # 4. Calculate and print metrics
    overall_acc = accuracy_score(y_true, y_pred) * 100
    print(f'\n📊 OVERALL TEST ACCURACY: {overall_acc:.2f}%\n')
    
    print("="*70)
    print("PER-CLASS PERFORMANCE")
    print("="*70)
    for i in range(config.N_CLASSES):
        class_mask = (y_true == i)
        class_total = int(np.sum(class_mask))
        if class_total == 0:
            # Avoid dividing by zero and calling accuracy on empty arrays
            print(f"{class_names[i]:30s}: {0:4d}/{class_total:4d} correct (   N/A )")
            continue

        class_correct = int(np.sum((y_true[class_mask] == y_pred[class_mask])))
        class_acc = (class_correct / class_total) * 100.0
        print(f"{class_names[i]:30s}: {class_correct:4d}/{class_total:4d} correct ({class_acc:6.2f}%)")

    # 5. Confusion Matrix
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(config.EVAL_CM_PATH)
    print(f"Confusion matrix saved to {config.EVAL_CM_PATH}")
    # plt.show() # Uncomment if running interactively

    # 6. Classification Report
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # 7. Aggregate Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    print("\n" + "="*70)
    print("AGGREGATE METRICS")
    print("="*70)
    print(f"Weighted Precision: {precision*100:.2f}%")
    print(f"Weighted Recall:    {recall*100:.2f}%")
    print(f"Weighted F1-Score:  {f1*100:.2f}%")
    print(f"Cohen's Kappa:      {kappa:.4f}")

    # 8. ROC Curves
    print("\n" + "="*70)
    print("ROC CURVES - MULTICLASS")
    print("="*70)
    y_true_bin = label_binarize(y_true, classes=list(range(config.N_CLASSES)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    print("Area Under Curve (AUC) per class:")
    roc_aucs = []
    for i in range(config.N_CLASSES):
        # roc_curve requires at least one positive and one negative sample for the class
        positives = int(np.sum(y_true_bin[:, i] == 1))
        negatives = int(np.sum(y_true_bin[:, i] == 0))
        if positives == 0:
            print(f"  {class_names[i]:30s}: No positive samples in y_true; skipping ROC/AUC")
            roc_aucs.append(np.nan)
            continue
        if negatives == 0:
            print(f"  {class_names[i]:30s}: No negative samples in y_true; skipping ROC/AUC")
            roc_aucs.append(np.nan)
            continue

        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
        except ValueError as e:
            print(f"  {class_names[i]:30s}: ROC error: {e}; skipping")
            roc_aucs.append(np.nan)
            continue

        roc_aucs.append(roc_auc)
        print(f"  {class_names[i]:30s}: {roc_auc:.4f}")
        # Full view
        ax1.plot(fpr, tpr, color=colors[i], lw=3, alpha=0.8,
                 label=f'{class_names[i]} (AUC = {roc_auc:.4f})')
        # Zoomed view
        ax2.plot(fpr, tpr, color=colors[i], lw=3, alpha=0.8,
                 label=f'{class_names[i]} (AUC = {roc_auc:.4f})')

    # Formatting Full View
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5000)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves - Full View', fontsize=13, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Formatting Zoomed View
    ax2.plot([0, 0.2], [0.8, 1.0], 'k--', lw=2, alpha=0.3)
    ax2.set_xlim([0.0, 0.2])
    ax2.set_ylim([0.8, 1.0])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curves - Zoomed View', fontsize=13, fontweight='bold')
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.EVAL_ROC_PATH)
    print(f"ROC curves plot saved to {config.EVAL_ROC_PATH}")
    # plt.show() # Uncomment if running interactively
    
    # Macro/micro AUC: wrap in try/except because they can fail when classes are missing
    try:
        macro_auc = roc_auc_score(y_true_bin, y_probs, average='macro')
        micro_auc = roc_auc_score(y_true_bin, y_probs, average='micro')
        print(f"\n  {'Macro-average AUC':30s}: {macro_auc:.4f}")
        print(f"  {'Micro-average AUC':30s}: {micro_auc:.4f}")
    except ValueError as e:
        print(f"Could not compute macro/micro AUC: {e}")
    
    print("\n" + "="*70)
    print("MODEL EVALUATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    evaluate_model()
