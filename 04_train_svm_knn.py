import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import json

def load_features_and_labels(split_name):
    """Load features and labels for a specific split"""
    features_path = f"models/{split_name}_features.npy"
    labels_path = f"models/{split_name}_labels.npy"
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Feature files not found for {split_name}")
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    print(f"Loaded {split_name}: {features.shape[0]} samples, {features.shape[1]} features")
    return features, labels

def train_svm(X_train, y_train):
    """Train SVM classifier"""
    print("\nTraining SVM classifier...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    print("[OK] SVM training completed")
    return svm

def train_knn(X_train, y_train):
    """Train KNN classifier"""
    print("\nTraining KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)
    print("[OK] KNN training completed")
    return knn

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and print metrics"""
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy

def save_model(model, model_path):
    """Save trained model"""
    joblib.dump(model, model_path)
    print(f"[OK] Model saved to {model_path}")

def save_training_results(results, output_path):
    """Save training results to JSON"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved to {output_path}")

def main():
    print("=" * 60)
    print("TRAINING SVM AND KNN CLASSIFIERS")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_features_and_labels('train')
    
    # Load validation data
    print("Loading validation data...")
    X_val, y_val = load_features_and_labels('val')
    
    # Load test data
    print("Loading test data...")
    X_test, y_test = load_features_and_labels('test')
    
    # Encode labels
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Save label encoder
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    print("[OK] Label encoder saved")
    
    # Train SVM
    svm_model = train_svm(X_train, y_train_encoded)
    
    # Train KNN
    knn_model = train_knn(X_train, y_train_encoded)
    
    # Evaluate SVM
    svm_accuracy = evaluate_model(svm_model, X_val, y_val_encoded, "SVM")
    
    # Evaluate KNN
    knn_accuracy = evaluate_model(knn_model, X_val, y_val_encoded, "KNN")
    
    # Save models
    save_model(svm_model, "models/svm_model.pkl")
    save_model(knn_model, "models/knn_model.pkl")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    svm_test_accuracy = evaluate_model(svm_model, X_test, y_test_encoded, "SVM (Test)")
    knn_test_accuracy = evaluate_model(knn_model, X_test, y_test_encoded, "KNN (Test)")
    
    # Save results
    results = {
        'svm_val_accuracy': float(svm_accuracy),
        'knn_val_accuracy': float(knn_accuracy),
        'svm_test_accuracy': float(svm_test_accuracy),
        'knn_test_accuracy': float(knn_test_accuracy),
        'classes': label_encoder.classes_.tolist()
    }
    save_training_results(results, "outputs/traditional_ml_results.json")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"SVM Test Accuracy: {svm_test_accuracy:.4f}")
    print(f"KNN Test Accuracy: {knn_test_accuracy:.4f}")

if __name__ == "__main__":
    main()