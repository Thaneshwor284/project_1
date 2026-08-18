import json
import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use('Agg')


def main():
    print('\n' + '=' * 60)
    print('STEP 4: TRAIN SVM & KNN CLASSIFIERS')
    print('=' * 60)

    print('\n✓ Loading features...')
    X_train = np.load('models/train_features.npy')
    y_train = np.load('models/train_labels.npy')
    X_test = np.load('models/test_features.npy')
    y_test = np.load('models/test_labels.npy')

    with open('data_splits.json', 'r') as f:
        splits = json.load(f)

    class_names = splits['class_names']
    os.makedirs('outputs', exist_ok=True)

    print(f'  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features')
    print(f'  Test:  {X_test.shape[0]} samples')

    print('\n✓ Normalizing features...')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print('\n✓ Training SVM classifier...')
    svm_model = SVC(kernel='rbf', C=100, gamma='scale', probability=False)
    svm_model.fit(X_train_scaled, y_train)

    y_pred_svm = svm_model.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)

    print(f'\n  SVM Accuracy: {svm_accuracy:.4f}')
    print('\n  Classification Report (SVM):')
    svm_report = classification_report(y_test, y_pred_svm, target_names=class_names, zero_division=0)
    print(svm_report)

    print('\n✓ Training KNN classifier...')
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)

    y_pred_knn = knn_model.predict(X_test_scaled)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)

    print(f'\n  KNN Accuracy: {knn_accuracy:.4f}')
    print('\n  Classification Report (KNN):')
    knn_report = classification_report(y_test, y_pred_knn, target_names=class_names, zero_division=0)
    print(knn_report)

    print('\n✓ Saving models...')
    joblib.dump(svm_model, 'models/svm_model.pkl')
    joblib.dump(knn_model, 'models/knn_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    results = {
        'svm_accuracy': float(svm_accuracy),
        'knn_accuracy': float(knn_accuracy),
        'num_features': X_train.shape[1],
        'num_classes': len(class_names),
        'class_names': class_names
    }

    with open('models/classical_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open('outputs/classical_svm_report.txt', 'w') as f:
        f.write(svm_report)
    with open('outputs/classical_knn_report.txt', 'w') as f:
        f.write(knn_report)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=True, xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f'SVM Confusion Matrix (Acc: {svm_accuracy:.4f})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    cm_knn = confusion_matrix(y_test, y_pred_knn)
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=True, xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(f'KNN Confusion Matrix (Acc: {knn_accuracy:.4f})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('outputs/confusion_matrices_classical.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print('\n✅ Classical models trained and saved!')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
