import cv2
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

IMG_SIZE = (224, 224)


def load_and_resize(img_path):
    """Load image and resize to a fixed standard size."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return cv2.resize(img, IMG_SIZE)


def apply_gaussian_blur(img, kernel_size=5):
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def rgb_to_hsv(img):
    """Convert BGR to HSV color space."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def segment_leaf(img_bgr):
    """Segment the leaf mask using green-range HSV thresholding."""
    blurred = apply_gaussian_blur(img_bgr)
    hsv = rgb_to_hsv(blurred)

    lower_green = np.array([25, 40, 40], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if mask.sum() == 0:
        segmented = img_bgr.copy()
    else:
        segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    return segmented, mask


def preprocess_image(img_path):
    """Complete preprocessing pipeline."""
    img = load_and_resize(img_path)
    img = apply_gaussian_blur(img)
    hsv = rgb_to_hsv(img)
    segmented, mask = segment_leaf(img)
    return img, hsv, segmented, mask


def visualize_preprocessing(original, hsv, segmented, mask, class_name, output_path):
    """Save a preprocessing visualization for one class."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(hsv)
    axes[0, 1].set_title('HSV Color Space', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('Leaf Mask', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Segmented Leaf', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle(f'Class: {class_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_preprocessed_image(item, split_name):
    """Save the filtered image into outputs/preprocessed/<split>/<class_name>/filename."""
    input_path = item['path']
    output_dir = Path('outputs') / 'preprocessed' / split_name / item['class_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(input_path).name

    try:
        original, _, segmented, _ = preprocess_image(input_path)
        cv2.imwrite(str(output_path), segmented)
        if not output_path.exists():
            cv2.imwrite(str(output_path), original)
        return str(output_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to preprocess {input_path}: {exc}") from exc


def main():
    print('\n' + '=' * 60)
    print('STEP 2: IMAGE PREPROCESSING')
    print('=' * 60)

    if not os.path.exists('data_splits.json'):
        print('❌ data_splits.json not found. Run 01_data_preparation.py first!')
        return

    os.makedirs('outputs', exist_ok=True)

    with open('data_splits.json', 'r') as f:
        splits = json.load(f)

    for split_name in ['train', 'val', 'test']:
        split_dir = Path('outputs') / 'preprocessed' / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for item in splits[split_name]:
            try:
                save_preprocessed_image(item, split_name)
            except Exception as exc:
                print(f'  ⚠️  Failed to process {item["path"]}: {exc}')

    print('\n✓ Saving sample preprocessing visualizations...')
    class_samples = {}
    for item in splits['train']:
        class_name = item['class_name']
        if class_name not in class_samples:
            class_samples[class_name] = item

    for class_name, sample in class_samples.items():
        try:
            original, hsv, segmented, mask = preprocess_image(sample['path'])
            output_file = Path('outputs') / f'preprocess_{class_name}.png'
            visualize_preprocessing(original, hsv, segmented, mask, class_name, str(output_file))
        except Exception as exc:
            print(f'  ❌ Error for {class_name}: {exc}')

    print('\n✅ Preprocessing complete. Processed images saved under outputs/preprocessed/')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
