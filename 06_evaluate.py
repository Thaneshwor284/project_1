import os
import json
import numpy as np
import joblib
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms

def load_traditional_models():
    """Load SVM and KNN models"""
    print("Loading traditional ML models...")
    
    svm_model = joblib.load("models/svm_model.pkl")
    knn_model = joblib.load("models/knn_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    
    print("[OK] Traditional ML models loaded")
    return svm_model, knn_model, label_encoder

def load_cnn_results():
    """Load CNN results from JSON file"""
    print("Loading CNN results...")
    
    if os.path.exists("outputs/cnn_results.json"):
        with open("outputs/cnn_results.json", 'r') as f:
            cnn_results = json.load(f)
        print("[OK] CNN results loaded")
        return cnn_results
    else:
        print("⚠ CNN results file not found")
        return None

def load_test_features():
    """Load test features and labels"""
    print("Loading test features...")
    
    X_test = np.load("models/test_features.npy")
    y_test = np.load("models/test_labels.npy")
    
    print(f"[OK] Loaded {len(X_test)} test samples")
    return X_test, y_test

def evaluate_traditional_models(svm_model, knn_model, label_encoder, X_test, y_test):
    """Evaluate SVM and KNN models"""
    print("\n" + "=" * 60)
    print("EVALUATING TRADITIONAL ML MODELS")
    print("=" * 60)
    
    # Encode labels
    y_test_encoded = label_encoder.transform(y_test)
    
    # SVM Evaluation
    print("\nSVM Model:")
    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test_encoded, svm_predictions)
    print(f"Accuracy: {svm_accuracy:.4f}")
    print("\nClassification Report:")
    svm_report = classification_report(y_test_encoded, svm_predictions, 
                                       target_names=label_encoder.classes_, output_dict=True)
    print(classification_report(y_test_encoded, svm_predictions, 
                               target_names=label_encoder.classes_))
    
    # KNN Evaluation
    print("\nKNN Model:")
    knn_predictions = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test_encoded, knn_predictions)
    print(f"Accuracy: {knn_accuracy:.4f}")
    print("\nClassification Report:")
    knn_report = classification_report(y_test_encoded, knn_predictions,
                                       target_names=label_encoder.classes_, output_dict=True)
    print(classification_report(y_test_encoded, knn_predictions,
                               target_names=label_encoder.classes_))
    
    return {
        'svm': {'accuracy': svm_accuracy, 'report': svm_report},
        'knn': {'accuracy': knn_accuracy, 'report': knn_report}
    }

def plot_confusion_matrix(y_true, y_pred, classes, title, output_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Confusion matrix saved to {output_path}")

def generate_comparison_report(traditional_results, cnn_results, output_path):
    """Generate comparison report"""
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON REPORT")
    print("=" * 60)
    
    report = {
        'model_comparison': {
            'SVM': {
                'accuracy': float(traditional_results['svm']['accuracy']),
                'precision': float(traditional_results['svm']['report']['macro avg']['precision']),
                'recall': float(traditional_results['svm']['report']['macro avg']['recall']),
                'f1_score': float(traditional_results['svm']['report']['macro avg']['f1-score'])
            },
            'KNN': {
                'accuracy': float(traditional_results['knn']['accuracy']),
                'precision': float(traditional_results['knn']['report']['macro avg']['precision']),
                'recall': float(traditional_results['knn']['report']['macro avg']['recall']),
                'f1_score': float(traditional_results['knn']['report']['macro avg']['f1-score'])
            },
            'CNN': {
                'accuracy': float(cnn_results.get('test_accuracy', 0.0)),
                'precision': 0.0,  # Will be filled if CNN evaluation is implemented
                'recall': 0.0,
                'f1_score': 0.0
            }
        },
        'best_model': 'CNN' if cnn_results.get('test_accuracy', 0) > max(
            traditional_results['svm']['accuracy'], 
            traditional_results['knn']['accuracy']
        ) else ('SVM' if traditional_results['svm']['accuracy'] > traditional_results['knn']['accuracy'] else 'KNN')
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[OK] Comparison report saved to {output_path}")
    
    # Print summary
    print("\nModel Comparison Summary:")
    print("-" * 60)
    for model_name, metrics in report['model_comparison'].items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    print(f"\nBest Model: {report['best_model']}")
    
    return report

def main():
    print("=" * 60)
    print("MODEL EVALUATION AND COMPARISON")
    print("=" * 60)
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Load traditional ML models
    try:
        svm_model, knn_model, label_encoder = load_traditional_models()
        
        # Load test features
        X_test, y_test = load_test_features()
        
        # Evaluate traditional models
        traditional_results = evaluate_traditional_models(
            svm_model, knn_model, label_encoder, X_test, y_test
        )
        
        # Plot confusion matrices
        y_test_encoded = label_encoder.transform(y_test)
        svm_predictions = svm_model.predict(X_test)
        knn_predictions = knn_model.predict(X_test)
        
        plot_confusion_matrix(
            y_test_encoded, svm_predictions, label_encoder.classes_,
            "SVM Confusion Matrix", "outputs/svm_confusion_matrix.png"
        )
        
        plot_confusion_matrix(
            y_test_encoded, knn_predictions, label_encoder.classes_,
            "KNN Confusion Matrix", "outputs/knn_confusion_matrix.png"
        )
        
    except Exception as e:
        print(f"Warning: Could not evaluate traditional ML models: {e}")
        traditional_results = None
    
    # Load CNN results if available
    cnn_results = load_cnn_results()
    if cnn_results is None:
        cnn_results = {}
    
    # Generate comparison report
    if traditional_results:
        comparison_report = generate_comparison_report(
            traditional_results, cnn_results, "outputs/model_comparison.json"
        )
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nGenerated outputs:")
    print("- outputs/svm_confusion_matrix.png")
    print("- outputs/knn_confusion_matrix.png")
    print("- outputs/model_comparison.json")

if __name__ == "__main__":
    main()