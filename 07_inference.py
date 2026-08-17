import os
import sys
import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import argparse
from skimage import feature

class SimpleCNN(nn.Module):
    """Simple CNN for tomato leaf disease classification"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def preprocess_image_for_traditional(image_path, target_size=(224, 224)):
    """Preprocess image for traditional ML models"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def extract_color_features(img):
    """Extract color-based features"""
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate mean and std for each channel
    features = []
    for channel in range(3):
        features.append(np.mean(hsv[:,:,channel]))
        features.append(np.std(hsv[:,:,channel]))
    
    return np.array(features)

def extract_texture_features(img):
    """Extract texture features using LBP"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
    
    # Histogram of LBP
    hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
    hist = hist.astype('float') / hist.sum()
    
    return hist

def extract_shape_features(img):
    """Extract shape-based features"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.array([0, 0, 0])
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    return np.array([area, perimeter, circularity])

def extract_all_features(img):
    """Extract all feature groups"""
    color_feats = extract_color_features(img)
    texture_feats = extract_texture_features(img)
    shape_feats = extract_shape_features(img)
    
    # Combine all features
    all_features = np.concatenate([color_feats, texture_feats, shape_feats])
    
    return all_features

def preprocess_image_for_cnn(image_path, target_size=(128, 128)):
    """Preprocess image for CNN model"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def predict_with_traditional_models(image_path, svm_model, knn_model, label_encoder):
    """Make predictions using traditional ML models"""
    print("Preprocessing image for traditional ML models...")
    
    # Preprocess and extract features
    img = preprocess_image_for_traditional(image_path)
    features = extract_all_features(img)
    features = features.reshape(1, -1)  # Reshape for single sample
    
    # SVM prediction
    svm_pred_encoded = svm_model.predict(features)[0]
    svm_pred_label = label_encoder.inverse_transform([svm_pred_encoded])[0]
    svm_proba = None
    
    if hasattr(svm_model, 'predict_proba'):
        svm_proba = svm_model.predict_proba(features)[0]
        svm_confidence = np.max(svm_proba)
    else:
        svm_confidence = 1.0  # No probability available
    
    # KNN prediction
    knn_pred_encoded = knn_model.predict(features)[0]
    knn_pred_label = label_encoder.inverse_transform([knn_pred_encoded])[0]
    knn_proba = None
    
    if hasattr(knn_model, 'predict_proba'):
        knn_proba = knn_model.predict_proba(features)[0]
        knn_confidence = np.max(knn_proba)
    else:
        knn_confidence = 1.0  # No probability available
    
    return {
        'svm': {
            'prediction': svm_pred_label,
            'confidence': float(svm_confidence)
        },
        'knn': {
            'prediction': knn_pred_label,
            'confidence': float(knn_confidence)
        }
    }

def predict_with_cnn(image_path, cnn_model, label_encoder, device='cpu'):
    """Make prediction using CNN model"""
    print("Preprocessing image for CNN model...")
    
    # Preprocess image
    img_tensor = preprocess_image_for_cnn(image_path)
    img_tensor = img_tensor.to(device)
    
    # Make prediction
    cnn_model.eval()
    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    pred_label = label_encoder.inverse_transform([predicted.item()])[0]
    
    return {
        'cnn': {
            'prediction': pred_label,
            'confidence': float(confidence.item())
        }
    }

def load_models():
    """Load all trained models"""
    print("Loading trained models...")
    
    # Load traditional ML models
    try:
        svm_model = joblib.load("models/svm_model.pkl")
        knn_model = joblib.load("models/knn_model.pkl")
        label_encoder_traditional = joblib.load("models/label_encoder.pkl")
        print("[OK] Traditional ML models loaded")
        traditional_loaded = True
    except Exception as e:
        print(f"Warning: Could not load traditional ML models: {e}")
        traditional_loaded = False
        svm_model = None
        knn_model = None
        label_encoder_traditional = None
    
    # Load CNN model
    try:
        checkpoint = torch.load("models/cnn_model.pth", map_location='cpu')
        num_classes = checkpoint['num_classes']
        cnn_model = SimpleCNN(num_classes=num_classes)
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        cnn_model.eval()
        label_encoder_cnn = checkpoint['label_encoder']
        print("[OK] CNN model loaded")
        cnn_loaded = True
    except Exception as e:
        print(f"Warning: Could not load CNN model: {e}")
        cnn_loaded = False
        cnn_model = None
        label_encoder_cnn = None
    
    return {
        'traditional_loaded': traditional_loaded,
        'svm_model': svm_model,
        'knn_model': knn_model,
        'label_encoder_traditional': label_encoder_traditional,
        'cnn_loaded': cnn_loaded,
        'cnn_model': cnn_model,
        'label_encoder_cnn': label_encoder_cnn
    }

def main():
    print("=" * 60)
    print("TOMATO LEAF DISEASE INFERENCE")
    print("=" * 60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict tomato leaf disease from image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['svm', 'knn', 'cnn', 'all'],
                       help='Which model to use for prediction')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)
    
    print(f"\nInput image: {args.image_path}")
    
    # Load models
    models = load_models()
    
    predictions = {}
    
    # Make predictions based on requested model
    if args.model in ['svm', 'knn', 'all']:
        if models['traditional_loaded']:
            try:
                traditional_preds = predict_with_traditional_models(
                    args.image_path,
                    models['svm_model'],
                    models['knn_model'],
                    models['label_encoder_traditional']
                )
                predictions.update(traditional_preds)
            except Exception as e:
                print(f"Error in traditional ML prediction: {e}")
    
    if args.model in ['cnn', 'all']:
        if models['cnn_loaded']:
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                cnn_preds = predict_with_cnn(
                    args.image_path,
                    models['cnn_model'],
                    models['label_encoder_cnn'],
                    device
                )
                predictions.update(cnn_preds)
            except Exception as e:
                print(f"Error in CNN prediction: {e}")
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    if not predictions:
        print("No predictions could be made. Please check model files.")
        sys.exit(1)
    
    for model_name, result in predictions.items():
        print(f"\n{model_name.upper()} Model:")
        print(f"  Predicted Class: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")
    
    # Determine consensus prediction if multiple models used
    if len(predictions) > 1:
        print("\n" + "-" * 60)
        all_predictions = [result['prediction'] for result in predictions.values()]
        from collections import Counter
        most_common = Counter(all_predictions).most_common(1)[0]
        print(f"Consensus Prediction: {most_common[0]}")
        print(f"Agreement: {most_common[1]}/{len(predictions)} models")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()