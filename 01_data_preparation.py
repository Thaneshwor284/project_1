import os
import json
import random
from pathlib import Path
from collections import defaultdict

def create_required_folders():
    """Create required folders if they don't exist"""
    folders = ['models', 'outputs']
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)
        print(f"[OK] Created/verified folder: {folder}")

def scan_dataset(dataset_path):
    """Scan dataset and collect all image paths with labels"""
    print(f"Scanning dataset at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    class_images = defaultdict(list)
    total_images = 0
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            print(f"  Processing class: {class_name}")
            image_files = []
            
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(class_path, file_name)
                    image_files.append(image_path)
            
            if image_files:
                class_images[class_name] = image_files
                total_images += len(image_files)
                print(f"    Found {len(image_files)} images")
            else:
                print(f"    Warning: No images found in {class_name}")
    
    print(f"\nTotal images found: {total_images}")
    print(f"Total classes: {len(class_images)}")
    
    return class_images

def split_data(class_images, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data into train, validation, and test sets"""
    print(f"\nSplitting data (train: {train_ratio}, val: {val_ratio}, test: {test_ratio})")
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for class_name, images in class_images.items():
        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        splits['train'].extend([(img, class_name) for img in train_images])
        splits['val'].extend([(img, class_name) for img in val_images])
        splits['test'].extend([(img, class_name) for img in test_images])
        
        print(f"  {class_name}: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    
    # Shuffle splits
    random.shuffle(splits['train'])
    random.shuffle(splits['val'])
    random.shuffle(splits['test'])
    
    print(f"\nTotal split sizes:")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")
    
    return splits

def save_splits(splits, output_path):
    """Save data splits to JSON file"""
    print(f"\nSaving splits to: {output_path}")
    
    # Convert tuples to lists for JSON serialization
    serializable_splits = {
        'train': [{'path': path, 'label': label} for path, label in splits['train']],
        'val': [{'path': path, 'label': label} for path, label in splits['val']],
        'test': [{'path': path, 'label': label} for path, label in splits['test']]
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_splits, f, indent=2)
    
    print("[OK] Splits saved successfully")

def main():
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Paths
    dataset_path = "dataset/PlantVillage"
    output_path = "data_splits.json"
    
    # Create required folders
    create_required_folders()
    
    # Scan dataset
    class_images = scan_dataset(dataset_path)
    
    if not class_images:
        raise ValueError("No images found in dataset!")
    
    # Split data
    splits = split_data(class_images)
    
    # Save splits
    save_splits(splits, output_path)
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()