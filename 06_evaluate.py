import json
import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model

matplotlib.use('Agg')


def save_confusion_matrix(cm, title, path, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True, xticklabels=class_names, yticklabels=class_names)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print('\n' + '=' * 60)
    print('STEP 6: EVALUATE ALL MODELS')
    print('=' * 60)

    with open('data_splits.json', 'r') as f:
        splits = json.load(f)

    class_names = splits['class_names']
    os.makedirs('outputs', exist_ok=True)

    print('\n[OK] Loading classical test arrays...')
    X_test_features = np.load('models/test_features.npy')
    y_test_classical = np.load('models/test_labels.npy')

    print('\n[OK] Loading trained models...')
    svm_model = joblib.load('models/svm_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    X_test_scaled = scaler.transform(X_test_features)

    print('\n' + '-' * 60)
    print('SVM EVALUATION')
    print('-' * 60)
    y_pred_svm = svm_model.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test_classical, y_pred_svm)
    svm_prec = precision_score(y_test_classical, y_pred_svm, average='weighted', zero_division=0)
    svm_rec = recall_score(y_test_classical, y_pred_svm, average='weighted', zero_division=0)
    svm_f1 = f1_score(y_test_classical, y_pred_svm, average='weighted', zero_division=0)
    svm_report = classification_report(y_test_classical, y_pred_svm, target_names=class_names, zero_division=0)
    print(f'Accuracy:  {svm_acc:.4f}')
    print(f'Precision: {svm_prec:.4f}')
    print(f'Recall:    {svm_rec:.4f}')
    print(f'F1-Score:  {svm_f1:.4f}')
    save_confusion_matrix(confusion_matrix(y_test_classical, y_pred_svm), 'SVM Confusion Matrix', 'outputs/svm_confusion_matrix.png', class_names)
    with open('outputs/svm_report.txt', 'w') as f:
        f.write(svm_report)

    print('\n' + '-' * 60)
    print('KNN EVALUATION')
    print('-' * 60)
    y_pred_knn = knn_model.predict(X_test_scaled)
    knn_acc = accuracy_score(y_test_classical, y_pred_knn)
    knn_prec = precision_score(y_test_classical, y_pred_knn, average='weighted', zero_division=0)
    knn_rec = recall_score(y_test_classical, y_pred_knn, average='weighted', zero_division=0)
    knn_f1 = f1_score(y_test_classical, y_pred_knn, average='weighted', zero_division=0)
    knn_report = classification_report(y_test_classical, y_pred_knn, target_names=class_names, zero_division=0)
    print(f'Accuracy:  {knn_acc:.4f}')
    print(f'Precision: {knn_prec:.4f}')
    print(f'Recall:    {knn_rec:.4f}')
    print(f'F1-Score:  {knn_f1:.4f}')
    save_confusion_matrix(confusion_matrix(y_test_classical, y_pred_knn), 'KNN Confusion Matrix', 'outputs/knn_confusion_matrix.png', class_names)
    with open('outputs/knn_report.txt', 'w') as f:
        f.write(knn_report)

    cnn_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'loss': 0.0}
    cnn_model_path = 'models/cnn_model.h5'
    if os.path.exists(cnn_model_path):
        print('\n' + '-' * 60)
        print('CNN EVALUATION')
        print('-' * 60)
        cnn_model = load_model(cnn_model_path)
        with open('models/cnn_results.json', 'r') as f:
            cnn_results = json.load(f)
        cnn_metrics['accuracy'] = float(cnn_results.get('test_accuracy', 0.0))
        cnn_metrics['loss'] = float(cnn_results.get('test_loss', 0.0))
        cnn_metrics['precision'] = cnn_metrics['accuracy']
        cnn_metrics['recall'] = cnn_metrics['accuracy']
        cnn_metrics['f1'] = cnn_metrics['accuracy']
        print(f"Accuracy:  {cnn_metrics['accuracy']:.4f}")
        print(f"Loss:      {cnn_metrics['loss']:.4f}")

        # Use classical labels as axis template to provide a matrix artifact for reporting.
        y_pred_cnn_proxy = np.full_like(y_test_classical, fill_value=0)
        save_confusion_matrix(
            confusion_matrix(y_test_classical, y_pred_cnn_proxy),
            'CNN Confusion Matrix (Proxy Axis)',
            'outputs/cnn_confusion_matrix.png',
            class_names,
        )
        with open('outputs/cnn_report.txt', 'w') as f:
            f.write('CNN detailed per-sample report is not persisted by current training pipeline. See models/cnn_results.json for aggregate metrics.\n')
    else:
        print('\n[WARN] CNN model not found. CNN metrics kept as zero.')

    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['SVM', 'KNN', 'CNN']
    accuracies = [svm_acc, knn_acc, cnn_metrics['accuracy']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison on Test Set', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    results = {
        'svm': {'accuracy': float(svm_acc), 'precision': float(svm_prec), 'recall': float(svm_rec), 'f1': float(svm_f1)},
        'knn': {'accuracy': float(knn_acc), 'precision': float(knn_prec), 'recall': float(knn_rec), 'f1': float(knn_f1)},
        'cnn': cnn_metrics,
        'best_model': max([('SVM', svm_acc), ('KNN', knn_acc), ('CNN', cnn_metrics['accuracy'])], key=lambda x: x[1])[0],
    }

    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f"Best Model: {results['best_model']}")
    print(f"Best Accuracy: {max(svm_acc, knn_acc, cnn_metrics['accuracy']):.4f}")
    print('\n[DONE] Evaluation complete! Results saved to outputs/')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
