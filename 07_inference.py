import argparse
import json

import cv2
import joblib
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)


def preprocess_for_cnn(img_path):
    """Preprocess image for CNN inference."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {img_path}')
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    return np.array([img])


def preprocess_for_classical(img_path):
    """Extract handcrafted features for classical models."""
    from skimage import feature

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {img_path}')
    img = cv2.resize(img, IMG_SIZE)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_feats = []
    for channel in range(3):
        color_feats.append(np.mean(hsv[:, :, channel]))
        color_feats.append(np.std(hsv[:, :, channel]))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
    hist = hist.astype('float') / hist.sum()

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        shape_feats = [0, 0, 0]
    else:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        shape_feats = [area, perimeter, circularity]

    return np.concatenate([color_feats, hist, shape_feats])


def predict_image(img_path, model_name=None):
    with open('data_splits.json') as f:
        splits = json.load(f)
    class_names = splits['class_names']

    svm_model = joblib.load('models/svm_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    cnn_model = load_model('models/cnn_model.h5')

    feature_vector = preprocess_for_classical(img_path)
    feature_vector = scaler.transform([feature_vector])

    svm_pred = svm_model.predict(feature_vector)[0]
    knn_pred = knn_model.predict(feature_vector)[0]

    cnn_img = preprocess_for_cnn(img_path)
    cnn_prob = cnn_model.predict(cnn_img, verbose=0)
    cnn_pred = int(np.argmax(cnn_prob[0]))
    cnn_confidence = float(cnn_prob[0][cnn_pred])

    outputs = {
        'svm': class_names[svm_pred],
        'knn': class_names[knn_pred],
        'cnn': class_names[cnn_pred],
        'cnn_confidence': cnn_confidence,
    }

    if model_name is not None:
        model_name = model_name.lower()
        if model_name not in {'svm', 'knn', 'cnn'}:
            raise ValueError('Model must be one of: svm, knn, cnn')
        print(f'{model_name.upper()} Prediction: {outputs[model_name]}')
        if model_name == 'cnn':
            print(f'CNN Confidence: {cnn_confidence:.4f}')
        return outputs

    print('SVM Prediction:       ' + outputs['svm'])
    print('KNN Prediction:       ' + outputs['knn'])
    print(f'CNN Prediction:       {outputs["cnn"]} (Confidence: {cnn_confidence:.4f})')
    return outputs


def main():
    parser = argparse.ArgumentParser(description='Predict tomato disease from an image.')
    parser.add_argument('image_path', nargs='?', default=None, help='Path to the image to classify.')
    parser.add_argument('--model', choices=['svm', 'knn', 'cnn'], default=None, help='Optional model to use for the prediction.')
    args = parser.parse_args()

    if args.image_path is None:
        with open('data_splits.json') as f:
            splits = json.load(f)
        args.image_path = splits['test'][0]['path']

    print('\n' + '=' * 60)
    print('INFERENCE ON NEW IMAGES')
    print('=' * 60)
    print(f'\n✓ Testing on: {args.image_path}')
    predict_image(args.image_path, model_name=args.model)
    print('\n✅ Inference complete!')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
