from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one tomato leaf image with the TFLite model.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("results/best_model.tflite"))
    parser.add_argument("--classes", type=Path, default=Path("results/split_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = json.loads(args.classes.read_text(encoding="utf-8"))["class_names"]
    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    height, width = input_details["shape"][1:3]

    image = Image.open(args.image).convert("RGB").resize((width, height))
    input_data = np.asarray(image, dtype=np.float32)[None, ...]
    if input_details["dtype"] != np.float32:
        scale, zero_point = input_details["quantization"]
        input_data = np.round(input_data / scale + zero_point).astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details["index"])[0]
    if output_details["dtype"] != np.float32:
        scale, zero_point = output_details["quantization"]
        scores = (scores.astype(np.float32) - zero_point) * scale
    predicted_index = int(np.argmax(scores))
    print(f"Predicted class: {classes[predicted_index]}")
    print(f"Confidence: {float(scores[predicted_index]):.4f}")


if __name__ == "__main__":
    main()