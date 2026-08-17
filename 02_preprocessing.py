import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_required_folders():
    """Create required folders if they don't exist"""
    folders = ['outputs/preprocessed/train', 'outputs/preprocessed/val', 'outputs/preprocessed/test']
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created/verified folder: {folder}")

def load_data_splits(splits_path):
    """Load data splits from JSON file"""
    print(f"Loading data splits from: {splits_path}")
    
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Data splits file not found: {splits_path}")
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    print(f"[OK] Loaded splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess a single image"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def save_preprocessed_image(img, output_path):
    """Save preprocessed image"""
    # Convert back to uint8 and scale to [0, 255]
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Convert RGB to BGR for saving with OpenCV
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, img_bgr)

def process_split(split_data, split_name, output_base_dir):
    """Process a single data split"""
    print(f"\nProcessing {split_name} split...")
    
    output_dir = os.path.join(output_base_dir, split_name)
    processed_count = 0
    failed_count = 0
    
    for item in tqdm(split_data, desc=f"Processing {split_name}"):
        image_path = item['path']
        label = item['label']
        
        # Preprocess image
        img = preprocess_image(image_path)
        
        if img is None:
            failed_count += 1
            continue
        
        # Create label folder
        label_dir = os.path.join(output_dir, label)
        Path(label_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        filename = os.path.basename(image_path)
        output_path = os.path.join(label_dir, filename)
        
        # Save preprocessed image
        save_preprocessed_image(img, output_path)
        processed_count += 1
    
    print(f"[OK] {split_name}: Processed {processed_count} images, Failed {failed_count}")
    return processed_count, failed_count

def main():
    print("=" * 60)
    print("IMAGE PREPROCESSING")
    print("=" * 60)
    
    # Paths
    splits_path = "data_splits.json"
    output_base_dir = "outputs/preprocessed"
    
    # Create required folders
    create_required_folders()
    
    # Load data splits
    splits = load_data_splits(splits_path)
    
    # Process each split
    total_processed = 0
    total_failed = 0
    
    for split_name in ['train', 'val', 'test']:
        processed, failed = process_split(splits[split_name], split_name, output_base_dir)
        total_processed += processed
        total_failed += failed
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print(f"Total processed: {total_processed}")
    print(f"Total failed: {total_failed}")
    print("=" * 60)

if __name__ == "__main__":
    main()