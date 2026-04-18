import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class IDSEvaluator:
    def __init__(self, model_path='models/best_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
    def load_test_data(self):
        """Load test data"""
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
        return X_test, y_test
    
    def evaluate(self, threshold=0.5):
        """Comprehensive evaluation"""
        X_test, y_test = self.load_test_data()
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test).flatten()
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Classification report
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'Attack'],
                                   digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\nDetailed Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"FPR:       {fpr:.4f}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        # Plot ROC curve
        self.plot_roc_curve(y_test, y_pred_proba)
        
        # Plot Precision-Recall curve
        self.plot_precision_recall_curve(y_test, y_pred_proba)
        
        # Find optimal threshold
        optimal_threshold = self.find_optimal_threshold(y_test, y_pred_proba)
        
        return y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300)
        print("\nConfusion matrix saved to results/confusion_matrix.png")
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/roc_curve.png', dpi=300)
        print("ROC curve saved to results/roc_curve.png")
        plt.show()
    
    def plot_precision_recall_curve(self, y_test, y_pred_proba):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/precision_recall_curve.png', dpi=300)
        print("Precision-Recall curve saved to results/precision_recall_curve.png")
        plt.show()
    
    def find_optimal_threshold(self, y_test, y_pred_proba):
        """Find optimal threshold that maximizes F1-score"""
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
        print(f"F1-Score at optimal threshold: {optimal_f1:.4f}")
        
        # Plot F1 vs Threshold
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, f1_scores, lw=2)
        plt.axvline(optimal_threshold, color='r', linestyle='--',
                   label=f'Optimal threshold = {optimal_threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel('F1-Score')
        plt.title('F1-Score vs Classification Threshold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/threshold_optimization.png', dpi=300)
        print("Threshold optimization plot saved")
        plt.show()
        
        return optimal_threshold


if __name__ == "__main__":
    evaluator = IDSEvaluator('models/best_model.h5')
    y_pred, y_pred_proba = evaluator.evaluate(threshold=0.5)