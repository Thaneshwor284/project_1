import os
import json
import random
from pathlib import Path

DATASET_DIR = "dataset/PlantVillage"
OUTPUT_JSON = "data_splits.json"
SEED = 42

TOMATO_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
]

def main():
    print("\n" + "="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60)
    
    data = []
    total_images = 0

    for label_idx, class_name in enumerate(TOMATO_CLASSES):
        class_path = os.path.join(DATASET_DIR, class_name)
        
        if not os.path.isdir(class_path):
            print(f"⚠️  Missing folder: {class_path}")
            continue

        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"✓ {class_name:<50} {len(images):>5} images")
        total_images += len(images)

        for img_file in images:
            data.append({
                "path": os.path.join(class_path, img_file).replace("\\", "/"),
                "label": label_idx,
                "class_name": class_name
            })

    print(f"\n📊 Total images found: {total_images}")

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(data)

    n = len(data)
    train_end = int(0.70 * n)
    val_end = train_end + int(0.15 * n)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    splits = {
        "class_names": TOMATO_CLASSES,
        "num_classes": len(TOMATO_CLASSES),
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\n📈 Data Split:")
    print(f"   Train: {len(train_data)} samples (70%)")
    print(f"   Val:   {len(val_data)} samples (15%)")
    print(f"   Test:  {len(test_data)} samples (15%)")
    print(f"\n✅ Data splits saved to: {OUTPUT_JSON}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
