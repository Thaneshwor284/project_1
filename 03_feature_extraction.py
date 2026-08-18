import json
import os
from pathlib import Path

import cv2
import numpy as np
from skimage import feature
from tqdm import tqdm

IMG_SIZE = (224, 224)


def load_and_resize(img_path):
    """Load and resize image."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    return cv2.resize(img, IMG_SIZE)


def extract_color_features(img):
    """Extract color-based features in HSV space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    for channel in range(3):
        features.append(np.mean(hsv[:, :, channel]))
        features.append(np.std(hsv[:, :, channel]))
    return np.array(features)


def extract_texture_features(img):
    """Extract texture features using LBP."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
    hist = hist.astype('float') / hist.sum()
    return hist


def extract_shape_features(img):
    """Extract shape-based features."""
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


def extract_features_from_image(img):
    """Extract all feature groups from a resized image."""
    color_feats = extract_color_features(img)
    texture_feats = extract_texture_features(img)
    shape_feats = extract_shape_features(img)
    return np.concatenate([color_feats, texture_feats, shape_feats])


def resolve_preprocessed_path(item, split_name):
    """Return the saved preprocessed image path if it exists; otherwise fall back to original."""
    filename = Path(item['path']).name
    candidates = [
        Path('outputs') / 'preprocessed' / split_name / item['class_name'] / filename,
        Path('outputs') / 'preprocessed' / split_name / str(item['label']) / filename,
        Path(item['path'])
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(Path(item['path']))


def main():
    print('\n' + '=' * 60)
    print('STEP 3: FEATURE EXTRACTION')
    print('=' * 60)

    if not os.path.exists('data_splits.json'):
        print('❌ data_splits.json not found!')
        return

    with open('data_splits.json', 'r') as f:
        splits = json.load(f)

    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    for split_name in ['train', 'val', 'test']:
        print(f'\n[OK] Extracting features for {split_name} set...')

        split_data = splits[split_name]
        features_list = []
        labels_list = []

        for item in tqdm(split_data, desc=split_name):
            try:
                preprocessed_path = resolve_preprocessed_path(item, split_name)
                img = cv2.imread(preprocessed_path)
                if img is None:
                    raise FileNotFoundError(f'Cannot read image: {preprocessed_path}')
                img = cv2.resize(img, IMG_SIZE)
                features = extract_features_from_image(img)
                features_list.append(features)
                labels_list.append(item['label'])
            except Exception as exc:
                print(f"  ⚠️  Skipped: {item['path']} - {exc}")
                continue

        features_array = np.array(features_list)
        labels_array = np.array(labels_list)

        np.save(f'models/{split_name}_features.npy', features_array)
        np.save(f'models/{split_name}_labels.npy', labels_array)

        print(f'  Extracted {len(features_list)} samples')
        print(f'  Feature shape: {features_array.shape}')

    print('\n[OK] Feature extraction complete!')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
