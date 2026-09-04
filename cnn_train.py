from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetSplit:
    train_paths: list[str]
    train_labels: list[int]
    val_paths: list[str]
    val_labels: list[int]
    test_paths: list[str]
    test_labels: list[int]
    class_names: list[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def collect_image_paths(dataset_dir: Path) -> tuple[list[str], list[int], list[str]]:
    class_dirs = sorted([path for path in dataset_dir.iterdir() if path.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in dataset directory: {dataset_dir}")

    image_paths: list[str] = []
    labels: list[int] = []
    class_names = [class_dir.name for class_dir in class_dirs]

    for class_index, class_dir in enumerate(class_dirs):
        class_images = sorted(
            [
                file_path
                for file_path in class_dir.iterdir()
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        if not class_images:
            raise ValueError(f"No images found in class folder: {class_dir}")

        for file_path in class_images:
            image_paths.append(str(file_path))
            labels.append(class_index)

    return image_paths, labels, class_names


def create_stratified_splits(
    image_paths: list[str], labels: list[int], class_names: list[str], seed: int
) -> DatasetSplit:
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels,
    )

    return DatasetSplit(
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        class_names=class_names,
    )


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], antialias=True)
    image = tf.cast(image, tf.float32)
    return image, label


def build_dataset(
    paths: Iterable[str],
    labels: Iterable[int],
    image_size: int,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    path_tensor = tf.constant(list(paths))
    label_tensor = tf.constant(list(labels), dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((path_tensor, label_tensor))
    if training:
        dataset = dataset.shuffle(buffer_size=len(path_tensor), seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: _decode_and_resize(path, label, image_size),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def save_split_summary(split: DatasetSplit, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_names": split.class_names,
        "num_classes": len(split.class_names),
        "train_size": len(split.train_paths),
        "val_size": len(split.val_paths),
        "test_size": len(split.test_paths),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight CNN for tomato leaf disease detection.")
    parser.add_argument("--dataset-dir", default="dataset\\PlantVillage", help="Path to PlantVillage tomato dataset.")
    parser.add_argument("--results-dir", default="results", help="Directory for training outputs.")
    parser.add_argument("--image-size", type=int, default=128, help="Image size (square).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    image_paths, labels, class_names = collect_image_paths(dataset_dir)
    split = create_stratified_splits(image_paths=image_paths, labels=labels, class_names=class_names, seed=args.seed)

    build_dataset(split.train_paths, split.train_labels, args.image_size, args.batch_size, training=True)
    build_dataset(split.val_paths, split.val_labels, args.image_size, args.batch_size, training=False)
    build_dataset(split.test_paths, split.test_labels, args.image_size, args.batch_size, training=False)

    results_dir = Path(args.results_dir)
    save_split_summary(split, results_dir / "split_summary.json")

    print(f"Found {len(class_names)} classes and {len(image_paths)} images.")
    print(f"Train/Val/Test: {len(split.train_paths)}/{len(split.val_paths)}/{len(split.test_paths)}")
    print(f"Saved split summary to {results_dir / 'split_summary.json'}")


if __name__ == "__main__":
    main()
