from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from cnn_train import build_dataset, collect_image_paths, create_stratified_splits, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert and evaluate the trained tomato CNN as TFLite.")
    parser.add_argument("--model", default="results/best_model.keras")
    parser.add_argument("--output", default="results/best_model.tflite")
    parser.add_argument("--dataset-dir", default="dataset/PlantVillage")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_test_paths(dataset_dir: Path, seed: int) -> tuple[list[str], list[int], list[str]]:
    paths, labels, class_names = collect_image_paths(dataset_dir)
    split = create_stratified_splits(paths, labels, class_names, seed)
    return split.test_paths, split.test_labels, class_names


def evaluate_tflite(
    interpreter: tf.lite.Interpreter,
    test_paths: list[str],
    labels: list[int],
    image_size: int,
) -> float:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details["index"]
    output_index = output_details["index"]
    input_dtype = input_details["dtype"]
    input_scale, input_zero_point = input_details["quantization"]
    predictions = []

    for path in test_paths:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [image_size, image_size], antialias=True)
        batch = image.numpy().astype(np.float32)[None, ...]
        if input_dtype != np.float32:
            batch = np.round(batch / input_scale + input_zero_point).astype(input_dtype)
        interpreter.set_tensor(input_index, batch)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        output_scale, output_zero_point = output_details["quantization"]
        if output.dtype != np.float32:
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        predictions.append(output)

    y_pred = np.argmax(np.concatenate(predictions, axis=0), axis=1)
    return float(np.mean(y_pred == np.asarray(labels, dtype=np.int32)))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    results_dir = Path(args.results_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path)
    test_paths, test_labels, class_names = load_test_paths(dataset_dir, args.seed)
    test_dataset = build_dataset(test_paths, test_labels, args.image_size, args.batch_size, training=False)
    keras_loss, keras_accuracy = model.evaluate(test_dataset, verbose=0)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()
    tflite_accuracy = evaluate_tflite(interpreter, test_paths, test_labels, args.image_size)
    accuracy_delta = tflite_accuracy - float(keras_accuracy)
    summary = (
        f"Keras model: {model_path}\n"
        f"TFLite model: {output_path}\n"
        "Quantization: dynamic range (Optimize.DEFAULT, no representative dataset)\n"
        f"Dataset: {dataset_dir}\n"
        f"Test samples: {len(test_labels)}\n"
        f"Classes: {len(class_names)}\n"
        f"Keras test loss: {float(keras_loss):.6f}\n"
        f"Keras test accuracy: {float(keras_accuracy):.6f}\n"
        f"TFLite test accuracy: {tflite_accuracy:.6f}\n"
        f"Accuracy delta (TFLite - Keras): {accuracy_delta:+.6f}\n"
        f"Absolute accuracy loss: {abs(accuracy_delta):.6f}\n"
        f"TFLite size (bytes): {len(tflite_model)}\n"
    )
    (results_dir / "tflite_conversion_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()