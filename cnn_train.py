from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
matplotlib.use("Agg")


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


def build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            layers.RandomFlip(mode="horizontal_and_vertical"),
            layers.RandomRotation(factor=0.10),
            layers.RandomZoom(height_factor=0.10, width_factor=0.10),
            layers.RandomBrightness(factor=0.15),
        ],
        name="data_augmentation",
    )


def conv_block(inputs: tf.Tensor, filters: int, dropout_rate: float) -> tf.Tensor:
    x = layers.SeparableConv2D(filters, kernel_size=3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_model(image_size: int, num_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = build_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    x = conv_block(x, filters=32, dropout_rate=0.10)
    x = conv_block(x, filters=64, dropout_rate=0.15)
    x = conv_block(x, filters=128, dropout_rate=0.20)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="tomato_lightweight_cnn")
    return model


def plot_training_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight CNN for tomato leaf disease detection.")
    parser.add_argument("--dataset-dir", default="dataset\\PlantVillage", help="Path to PlantVillage tomato dataset.")
    parser.add_argument("--results-dir", default="results", help="Directory for training outputs.")
    parser.add_argument("--image-size", type=int, default=128, help="Image size (square).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs (recommended: 50-100).")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    image_paths, labels, class_names = collect_image_paths(dataset_dir)
    split = create_stratified_splits(image_paths=image_paths, labels=labels, class_names=class_names, seed=args.seed)

    results_dir = Path(args.results_dir)
    train_ds = build_dataset(split.train_paths, split.train_labels, args.image_size, args.batch_size, training=True)
    val_ds = build_dataset(split.val_paths, split.val_labels, args.image_size, args.batch_size, training=False)
    build_dataset(split.test_paths, split.test_labels, args.image_size, args.batch_size, training=False)

    save_split_summary(split, results_dir / "split_summary.json")

    model = build_model(image_size=args.image_size, num_classes=len(class_names))
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_model_path = results_dir / "best_model.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        CSVLogger(str(results_dir / "epoch_metrics.csv"), append=False),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    plot_training_history(history, results_dir / "training_curves.png")

    print(f"Found {len(class_names)} classes and {len(image_paths)} images.")
    print(f"Train/Val/Test: {len(split.train_paths)}/{len(split.val_paths)}/{len(split.test_paths)}")
    print(f"Saved split summary to {results_dir / 'split_summary.json'}")
    print(f"Saved best model to {best_model_path}")
    print(f"Saved epoch metrics to {results_dir / 'epoch_metrics.csv'}")
    print(f"Saved training plot to {results_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
