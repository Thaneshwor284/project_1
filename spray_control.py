from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from gpiozero import OutputDevice


CONFIDENCE_THRESHOLD = 0.80
HEALTHY_CLASS = "Tomato_healthy"
SPRAY_DURATION_SEC = 2.0
RELAY_PIN = 17


def configure_logging() -> logging.Logger:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("spray_control")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        logs_dir / "spray_control.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify camera frames and control a spray pump.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the TFLite model.")
    parser.add_argument("--classes", type=Path, required=True, help="Path to the class-name JSON file.")
    parser.add_argument("--camera-index", type=int, required=True, help="OpenCV camera device index.")
    return parser.parse_args()


def load_model_and_classes(args: argparse.Namespace, logger: logging.Logger):
    if not args.model.is_file():
        logger.critical("TFLite model file is missing: %s", args.model)
        sys.exit(1)
    if not args.classes.is_file():
        logger.critical("Class metadata file is missing: %s", args.classes)
        sys.exit(1)

    try:
        class_names = json.loads(args.classes.read_text(encoding="utf-8"))["class_names"]
        interpreter = tf.lite.Interpreter(model_path=str(args.model))
        interpreter.allocate_tensors()
    except Exception:
        logger.exception("Failed to load TFLite model or class metadata")
        sys.exit(1)

    return interpreter, class_names


def predict_frame(frame: np.ndarray, interpreter, class_names: list[str]) -> tuple[str, float]:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    height, width = input_details["shape"][1:3]
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
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
    return str(class_names[predicted_index]), float(scores[predicted_index])


def spray(relay: OutputDevice, logger: logging.Logger) -> None:
    try:
        relay.on()
        logger.info("Spray activated for %.2f seconds", SPRAY_DURATION_SEC)
        time.sleep(SPRAY_DURATION_SEC)
    except Exception:
        logger.exception("Spray activation failed")
    finally:
        relay.off()
        logger.info("Spray relay turned off")


def main() -> None:
    logger = configure_logging()
    args = parse_args()
    interpreter, class_names = load_model_and_classes(args, logger)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    if len(class_names) != output_details["shape"][-1]:
        logger.critical("Class metadata count does not match model output count")
        sys.exit(1)

    try:
        relay = OutputDevice(RELAY_PIN, active_high=False, initial_value=False)
        camera = cv2.VideoCapture(args.camera_index)
    except Exception:
        logger.exception("Failed to initialize relay or camera")
        sys.exit(1)

    if not camera.isOpened():
        logger.error("Could not open camera at index %d", args.camera_index)

    logger.info("Spray controller started")
    try:
        while camera.isOpened():
            try:
                captured, frame = camera.read()
            except Exception:
                logger.exception("Camera capture failed; skipping cycle")
                continue
            if not captured or frame is None:
                logger.error("Camera capture failed; skipping cycle")
                continue

            try:
                predicted_class, confidence = predict_frame(frame, interpreter, class_names)
            except Exception:
                logger.exception("Prediction failed; skipping cycle")
                continue
            logger.info("Prediction: %s (confidence %.4f)", predicted_class, confidence)
            if predicted_class != HEALTHY_CLASS and confidence >= CONFIDENCE_THRESHOLD:
                spray(relay, logger)
    except KeyboardInterrupt:
        logger.info("Spray controller stopped")
    finally:
        camera.release()
        relay.off()
        relay.close()


if __name__ == "__main__":
    main()