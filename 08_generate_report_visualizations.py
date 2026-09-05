import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model

matplotlib.use("Agg")

LOGGER = logging.getLogger("viz")
IMG_SIZE = (224, 224)
NUM_CLASSES = 10


@dataclass
class Artifacts:
    class_names: List[str]
    y_true: Optional[np.ndarray]
    y_pred_knn: Optional[np.ndarray]
    y_pred_cnn: Optional[np.ndarray]
    y_score_cnn: Optional[np.ndarray]
    y_score_knn: Optional[np.ndarray]
    image_paths: Optional[List[str]]
    features: Optional[np.ndarray]
    cnn_history: Optional[Dict[str, List[float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready visualizations for tomato leaf disease models."
    )

    parser.add_argument("--y-true", default="models/test_labels.npy", help="Path to y_true.npy")
    parser.add_argument("--y-pred-knn", default="", help="Path to y_pred_knn.npy (optional)")
    parser.add_argument("--y-pred-cnn", default="", help="Path to y_pred_cnn.npy (optional)")
    parser.add_argument("--y-score-cnn", default="", help="Path to y_score_cnn.npy (optional)")
    parser.add_argument("--class-names", default="", help="Path to class_names.json (optional)")
    parser.add_argument("--cnn-history", default="", help="Path to cnn_history.json (optional)")
    parser.add_argument("--features", default="models/test_features.npy", help="Path to features.npy (optional)")
    parser.add_argument("--image-paths", default="", help="Path to image_paths.npy or json list (optional)")

    parser.add_argument("--data-splits", default="data_splits.json", help="Path to data_splits.json")
    parser.add_argument("--knn-model", default="models/knn_model.pkl", help="Path to KNN model")
    parser.add_argument("--scaler", default="models/scaler.pkl", help="Path to scaler.pkl")
    parser.add_argument("--cnn-model", default="models/cnn_model.h5", help="Path to CNN model")

    parser.add_argument("--output-dir", default="outputs/visualizations", help="Output directory")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--max-misclassified", type=int, default=12, help="Max images in misclassification grid")
    parser.add_argument("--sort-by-f1", action="store_true", help="Sort per-class bars by F1 descending")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_class_names(args: argparse.Namespace) -> List[str]:
    if args.class_names and os.path.exists(args.class_names):
        data = _read_json(args.class_names)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "class_names" in data and isinstance(data["class_names"], list):
                return data["class_names"]

    if os.path.exists(args.data_splits):
        splits = _read_json(args.data_splits)
        if "class_names" in splits:
            return splits["class_names"]

    raise FileNotFoundError("Could not resolve class_names from class_names.json or data_splits.json")


def _load_image_paths(args: argparse.Namespace) -> Optional[List[str]]:
    if args.image_paths and os.path.exists(args.image_paths):
        if args.image_paths.endswith(".npy"):
            arr = np.load(args.image_paths, allow_pickle=True)
            return [str(x) for x in arr.tolist()]
        data = _read_json(args.image_paths)
        if isinstance(data, list):
            return [str(x) for x in data]

    if os.path.exists(args.data_splits):
        splits = _read_json(args.data_splits)
        test_items = splits.get("test", [])
        if test_items:
            return [item["path"] for item in test_items if "path" in item]

    return None


def _load_cnn_history(path: str) -> Optional[Dict[str, List[float]]]:
    if not path or not os.path.exists(path):
        return None
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    return data


def _prepare_cnn_input(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    valid_paths: List[str] = []
    images: List[np.ndarray] = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE).astype("float32") / 255.0
        images.append(img)
        valid_paths.append(p)
    if not images:
        raise ValueError("No valid images were loaded for CNN inference.")
    return np.array(images), np.array(valid_paths)


def compute_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict:
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def load_artifacts(args: argparse.Namespace) -> Artifacts:
    class_names = _load_class_names(args)
    image_paths = _load_image_paths(args)

    y_true = None
    if args.y_true and os.path.exists(args.y_true):
        y_true = np.load(args.y_true)
    else:
        LOGGER.warning("Missing y_true file: %s", args.y_true)

    features = None
    if args.features and os.path.exists(args.features):
        features = np.load(args.features)

    y_pred_knn = None
    y_score_knn = None
    if args.y_pred_knn and os.path.exists(args.y_pred_knn):
        y_pred_knn = np.load(args.y_pred_knn)
    elif features is not None and os.path.exists(args.knn_model) and os.path.exists(args.scaler):
        LOGGER.info("Computing KNN predictions from model + features...")
        knn_model = joblib.load(args.knn_model)
        scaler = joblib.load(args.scaler)
        X_scaled = scaler.transform(features)
        y_pred_knn = knn_model.predict(X_scaled)
        if hasattr(knn_model, "predict_proba"):
            y_score_knn = knn_model.predict_proba(X_scaled)
    else:
        LOGGER.warning("KNN predictions unavailable (missing y_pred_knn.npy or required model/features files).")

    y_pred_cnn = None
    y_score_cnn = None
    if args.y_score_cnn and os.path.exists(args.y_score_cnn):
        y_score_cnn = np.load(args.y_score_cnn)
        y_pred_cnn = np.argmax(y_score_cnn, axis=1)
    elif args.y_pred_cnn and os.path.exists(args.y_pred_cnn):
        y_pred_cnn = np.load(args.y_pred_cnn)
    elif os.path.exists(args.cnn_model) and image_paths is not None:
        LOGGER.info("Computing CNN predictions from model + test images...")
        cnn_model = load_model(args.cnn_model)
        X_cnn, valid_paths = _prepare_cnn_input(image_paths)
        y_score_cnn = cnn_model.predict(X_cnn, verbose=0)
        y_pred_cnn = np.argmax(y_score_cnn, axis=1)

        if y_true is not None and len(y_true) != len(y_pred_cnn):
            LOGGER.warning(
                "y_true length (%s) differs from CNN predictions (%s). Aligning to minimum length.",
                len(y_true),
                len(y_pred_cnn),
            )
            n = min(len(y_true), len(y_pred_cnn))
            y_true = y_true[:n]
            y_pred_cnn = y_pred_cnn[:n]
            y_score_cnn = y_score_cnn[:n] if y_score_cnn is not None else None
            valid_paths = valid_paths[:n]

        image_paths = [str(x) for x in valid_paths.tolist()]
    else:
        LOGGER.warning("CNN predictions unavailable (missing artifacts and/or cnn_model.h5).")

    cnn_history = _load_cnn_history(args.cnn_history)
    return Artifacts(
        class_names=class_names,
        y_true=y_true,
        y_pred_knn=y_pred_knn,
        y_pred_cnn=y_pred_cnn,
        y_score_cnn=y_score_cnn,
        y_score_knn=y_score_knn,
        image_paths=image_paths,
        features=features,
        cnn_history=cnn_history,
    )


def plot_confusion_matrix_with_percentages(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    out_path: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=True,
        linewidths=0.5,
        linecolor="lightgray",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    for i in range(min(cm.shape[0], cm.shape[1])):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor="white", lw=2.0))

    ax.set_title(f"Confusion Matrix — {model_name} (Counts and % by True Class)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metrics(report_dict: Dict, class_names: List[str], model_name: str, out_path: str, sort_by_f1: bool) -> None:
    rows = []
    for cls in class_names:
        if cls in report_dict:
            rows.append(
                {
                    "class": cls,
                    "precision": report_dict[cls]["precision"],
                    "recall": report_dict[cls]["recall"],
                    "f1": report_dict[cls]["f1-score"],
                }
            )

    if not rows:
        LOGGER.warning("No per-class metrics to plot for %s", model_name)
        return

    if sort_by_f1:
        rows = sorted(rows, key=lambda x: x["f1"], reverse=True)

    classes = [r["class"] for r in rows]
    precision_vals = np.array([r["precision"] for r in rows])
    recall_vals = np.array([r["recall"] for r in rows])
    f1_vals = np.array([r["f1"] for r in rows])

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(16, 8))
    bars_p = ax.bar(x - w, precision_vals, width=w, label="Precision", color="#1f77b4")
    bars_r = ax.bar(x, recall_vals, width=w, label="Recall", color="#ff7f0e")
    bars_f = ax.bar(x + w, f1_vals, width=w, label="F1", color="#2ca02c")

    for bars in (bars_p, bars_r, bars_f):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2.0, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Class Precision / Recall / F1 — {model_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Optional[Dict[str, List[float]]], out_path: str) -> None:
    if history is None:
        LOGGER.warning("CNN history missing; creating placeholder training-curves figure.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "CNN history not found\nProvide --cnn-history to plot real curves.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title("CNN Training Curves")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    train_loss = history.get("train_loss", history.get("loss"))
    val_loss = history.get("val_loss")
    train_acc = history.get("train_acc", history.get("accuracy"))
    val_acc = history.get("val_acc", history.get("val_accuracy"))

    if not all([train_loss, val_loss, train_acc, val_acc]):
        LOGGER.warning("CNN history keys incomplete; creating placeholder training-curves figure.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "CNN history keys missing\nRequired: train_loss, val_loss, train_acc, val_acc",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title("CNN Training Curves")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = int(np.argmin(val_loss) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[0].axvline(best_epoch, color="red", linestyle="--", alpha=0.6)
    axes[0].annotate("best val", (best_epoch, val_loss[best_epoch - 1]), xytext=(best_epoch, val_loss[best_epoch - 1] + 0.05))
    axes[0].set_title("CNN Training Curves")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, marker="o", label="Train Acc")
    axes[1].plot(epochs, val_acc, marker="o", label="Val Acc")
    axes[1].axvline(best_epoch, color="red", linestyle="--", alpha=0.6)
    axes[1].annotate("best val", (best_epoch, val_acc[best_epoch - 1]), xytext=(best_epoch, min(1.0, val_acc[best_epoch - 1] + 0.05)))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(epochs)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_summary(model_metrics: Dict[str, Dict[str, float]], out_path: str) -> None:
    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    models = list(model_metrics.keys())

    x = np.arange(len(metric_names))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, model_name in enumerate(models):
        vals = [model_metrics[model_name].get(m, 0.0) for m in metric_names]
        bars = ax.bar(x + i * width - (len(models) - 1) * width / 2, vals, width=width, label=model_name)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2.0, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Overall Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1_comparison(knn_f1: List[float], cnn_f1: List[float], class_names: List[str], out_path: str, sort_by_avg: bool) -> None:
    if len(knn_f1) != len(class_names) or len(cnn_f1) != len(class_names):
        LOGGER.warning("Per-class F1 arrays are not aligned; skipping per-class F1 comparison.")
        return

    idx = np.arange(len(class_names))
    if sort_by_avg:
        avg = (np.array(knn_f1) + np.array(cnn_f1)) / 2.0
        idx = np.argsort(avg)[::-1]

    names = [class_names[i] for i in idx]
    knn_vals = [knn_f1[i] for i in idx]
    cnn_vals = [cnn_f1[i] for i in idx]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - w / 2, knn_vals, width=w, label="KNN", color="#4E79A7")
    bars2 = ax.bar(x + w / 2, cnn_vals, width=w, label="CNN", color="#F28E2B")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2.0, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-Class F1 Comparison: KNN vs CNN")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_curves(y_true: np.ndarray, y_score: np.ndarray, class_names: List[str], out_roc: str, out_pr: str, model_name: str) -> None:
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    if y_score.shape[1] != n_classes:
        LOGGER.warning("y_score columns (%s) != number of classes (%s); skipping ROC/PR.", y_score.shape[1], n_classes)
        return

    fpr: Dict = {}
    tpr: Dict = {}
    roc_auc: Dict = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_score[:, i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = roc_auc_score(y_true_bin, y_score, average="micro", multi_class="ovr")
    roc_auc["macro"] = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")

    fig, ax = plt.subplots(figsize=(12, 9))
    for i, cls_name in enumerate(class_names):
        ax.plot(fpr[i], tpr[i], lw=1.5, label=f"{cls_name} (AUC={roc_auc[i]:.3f})")

    ax.plot(fpr["micro"], tpr["micro"], lw=3, linestyle="--", label=f"micro-average (AUC={roc_auc['micro']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves per Class — {model_name}")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_roc, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 9))
    for i, cls_name in enumerate(class_names):
        p, r, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
        ax.plot(r, p, lw=1.5, label=f"{cls_name} (AP={ap:.3f})")

    p_micro, r_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
    ap_micro = average_precision_score(y_true_bin, y_score, average="micro")
    ax.plot(r_micro, p_micro, lw=3, linestyle="--", label=f"micro-average (AP={ap_micro:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {model_name}")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_pr, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_tsne(features: np.ndarray, labels: np.ndarray, class_names: List[str], out_path: str, random_state: int, perplexity: float) -> None:
    if features is None or labels is None:
        LOGGER.warning("Features or labels missing; skipping t-SNE plot.")
        return

    n = min(len(features), len(labels))
    X = features[:n]
    y = labels[:n]

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    if X.shape[0] < 5:
        LOGGER.warning("Not enough samples for t-SNE; skipping.")
        return

    effective_perplexity = min(perplexity, max(5.0, (X.shape[0] - 1) / 3.0))
    tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto", perplexity=effective_perplexity)
    emb = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(12, 9))
    palette = sns.color_palette("tab10", n_colors=len(class_names))
    for i, cls_name in enumerate(class_names):
        m = y == i
        if np.any(m):
            ax.scatter(emb[m, 0], emb[m, 1], s=18, alpha=0.75, color=palette[i], label=cls_name)

    ax.set_title("Feature Space (t-SNE) — test features")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def show_misclassified_images(
    image_paths: Optional[List[str]],
    y_true: Optional[np.ndarray],
    y_pred: Optional[np.ndarray],
    class_names: List[str],
    out_path: str,
    y_score: Optional[np.ndarray],
    max_images: int,
    model_name: str,
) -> None:
    if image_paths is None or y_true is None or y_pred is None:
        LOGGER.warning("Missing image paths / labels / predictions for misclassification grid.")
        return

    n = min(len(image_paths), len(y_true), len(y_pred))
    mis_idx = [i for i in range(n) if y_true[i] != y_pred[i]]
    if not mis_idx:
        LOGGER.warning("No misclassified samples found.")
        return

    mis_idx = mis_idx[:max_images]
    cols = 4
    rows = int(np.ceil(len(mis_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for k, idx in enumerate(mis_idx):
        r, c = divmod(k, cols)
        ax = axes[r, c]

        img = cv2.imread(image_paths[idx])
        if img is None:
            ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        true_name = class_names[int(y_true[idx])]
        pred_name = class_names[int(y_pred[idx])]
        pred_prob = float(np.max(y_score[idx])) if y_score is not None and idx < len(y_score) else np.nan
        caption = f"True: {true_name}\nPred: {pred_name} | prob: {pred_prob:.2f}"
        ax.set_title(caption, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Example Misclassifications — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_dashboard(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_path: str, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    support = cm.sum(axis=1)
    per_class_acc = np.divide(np.diag(cm), support, out=np.zeros_like(support, dtype=float), where=support != 0)

    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    pairs = []
    for i in range(off_diag.shape[0]):
        for j in range(off_diag.shape[1]):
            if off_diag[i, j] > 0:
                pairs.append((f"{class_names[i]} -> {class_names[j]}", int(off_diag[i, j])))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:5]

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0
    ann = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    sns.heatmap(cm, annot=ann, fmt="", cmap="Blues", ax=axes[0, 0], cbar=False, xticklabels=class_names, yticklabels=class_names)
    axes[0, 0].set_title("Confusion Matrix (Counts + %)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    y_pos = np.arange(len(class_names))
    axes[0, 1].barh(y_pos, per_class_acc, color="#76B7B2")
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(class_names)
    axes[0, 1].set_xlim(0, 1.0)
    axes[0, 1].set_title("Per-Class Accuracy")
    axes[0, 1].grid(axis="x", alpha=0.3)

    if pairs:
        names = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        axes[1, 0].barh(np.arange(len(names)), vals, color="#E15759")
        axes[1, 0].set_yticks(np.arange(len(names)))
        axes[1, 0].set_yticklabels(names)
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_title("Top-5 Confused Class Pairs")
    else:
        axes[1, 0].text(0.5, 0.5, "No off-diagonal errors", ha="center", va="center")
        axes[1, 0].set_title("Top-5 Confused Class Pairs")

    axes[1, 1].pie(support, labels=class_names, autopct="%1.1f%%", startangle=90)
    axes[1, 1].set_title("Class Distribution")

    fig.suptitle(f"Error Analysis Dashboard — {model_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_logging()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    artifacts = load_artifacts(args)
    y_true = artifacts.y_true
    class_names = artifacts.class_names

    if y_true is None:
        raise FileNotFoundError("y_true is required. Provide --y-true or ensure models/test_labels.npy exists.")

    model_metrics: Dict[str, Dict[str, float]] = {}
    default_dashboard_written = False

    if artifacts.y_pred_knn is not None:
        y_pred_knn = artifacts.y_pred_knn[: len(y_true)]
        plot_confusion_matrix_with_percentages(
            y_true,
            y_pred_knn,
            class_names,
            "KNN",
            os.path.join(args.output_dir, "confusion_matrix_knn.png"),
        )
        report_knn = compute_classification_report(y_true, y_pred_knn, class_names)
        plot_per_class_metrics(
            report_knn,
            class_names,
            "KNN",
            os.path.join(args.output_dir, "per_class_metrics_knn.png"),
            args.sort_by_f1,
        )

        model_metrics["KNN"] = {
            "accuracy": accuracy_score(y_true, y_pred_knn),
            "macro_precision": precision_score(y_true, y_pred_knn, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred_knn, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred_knn, average="macro", zero_division=0),
        }

        plot_error_dashboard(
            y_true,
            y_pred_knn,
            class_names,
            os.path.join(args.output_dir, "error_dashboard_knn.png"),
            "KNN",
        )
        if not default_dashboard_written:
            plot_error_dashboard(
                y_true,
                y_pred_knn,
                class_names,
                os.path.join(args.output_dir, "error_dashboard.png"),
                "KNN",
            )
            default_dashboard_written = True

    if artifacts.y_pred_cnn is not None:
        y_pred_cnn = artifacts.y_pred_cnn[: len(y_true)]
        y_score_cnn = artifacts.y_score_cnn[: len(y_true)] if artifacts.y_score_cnn is not None else None

        plot_confusion_matrix_with_percentages(
            y_true,
            y_pred_cnn,
            class_names,
            "CNN",
            os.path.join(args.output_dir, "confusion_matrix_cnn.png"),
        )
        report_cnn = compute_classification_report(y_true, y_pred_cnn, class_names)
        plot_per_class_metrics(
            report_cnn,
            class_names,
            "CNN",
            os.path.join(args.output_dir, "per_class_metrics_cnn.png"),
            args.sort_by_f1,
        )

        model_metrics["CNN"] = {
            "accuracy": accuracy_score(y_true, y_pred_cnn),
            "macro_precision": precision_score(y_true, y_pred_cnn, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred_cnn, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred_cnn, average="macro", zero_division=0),
        }

        if y_score_cnn is not None:
            plot_roc_pr_curves(
                y_true,
                y_score_cnn,
                class_names,
                os.path.join(args.output_dir, "roc_curves.png"),
                os.path.join(args.output_dir, "pr_curves.png"),
                model_name="CNN",
            )

        show_misclassified_images(
            artifacts.image_paths,
            y_true,
            y_pred_cnn,
            class_names,
            os.path.join(args.output_dir, "misclassifications.png"),
            y_score_cnn,
            args.max_misclassified,
            model_name="CNN",
        )

        plot_error_dashboard(
            y_true,
            y_pred_cnn,
            class_names,
            os.path.join(args.output_dir, "error_dashboard_cnn.png"),
            "CNN",
        )
        plot_error_dashboard(
            y_true,
            y_pred_cnn,
            class_names,
            os.path.join(args.output_dir, "error_dashboard.png"),
            "CNN",
        )
        default_dashboard_written = True

    if model_metrics:
        plot_model_comparison_summary(model_metrics, os.path.join(args.output_dir, "model_comparison.png"))

    if artifacts.y_pred_knn is not None and artifacts.y_pred_cnn is not None:
        knn_report = compute_classification_report(y_true, artifacts.y_pred_knn[: len(y_true)], class_names)
        cnn_report = compute_classification_report(y_true, artifacts.y_pred_cnn[: len(y_true)], class_names)
        knn_f1 = [knn_report[c]["f1-score"] for c in class_names if c in knn_report]
        cnn_f1 = [cnn_report[c]["f1-score"] for c in class_names if c in cnn_report]
        if len(knn_f1) == len(class_names) and len(cnn_f1) == len(class_names):
            plot_per_class_f1_comparison(
                knn_f1,
                cnn_f1,
                class_names,
                os.path.join(args.output_dir, "per_class_f1_comparison.png"),
                sort_by_avg=args.sort_by_f1,
            )

    if artifacts.features is not None:
        plot_feature_tsne(
            artifacts.features,
            y_true,
            class_names,
            os.path.join(args.output_dir, "tsne_features.png"),
            random_state=args.random_state,
            perplexity=args.tsne_perplexity,
        )

    plot_training_curves(
        artifacts.cnn_history,
        os.path.join(args.output_dir, "cnn_training_curves.png"),
    )

    LOGGER.info("Visualization generation complete. Outputs saved in: %s", args.output_dir)


if __name__ == "__main__":
    main()
