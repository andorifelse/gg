from argparse import ArgumentParser
from pathlib import Path
import colorsys
import json

import numpy as np
from PIL import Image


def id2rgb(label, max_num_obj=256):
    if not 0 <= label <= max_num_obj:
        raise ValueError(f"ID should be in range(0, {max_num_obj})")

    rgb = np.zeros((3,), dtype=np.uint8)
    if label == 0:
        return rgb

    golden_ratio = 1.6180339887
    h = (label * golden_ratio) % 1.0
    s = 0.5 + (label % 2) * 0.5
    l = 0.5
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r * 255), int(g * 255), int(b * 255)
    return rgb


def build_color_map(max_num_classes):
    return {
        tuple(id2rgb(class_id, max_num_classes).tolist()): class_id
        for class_id in range(max_num_classes + 1)
    }


def image_to_labels(image_path, color_to_label):
    image = np.array(Image.open(image_path).convert("RGB"))
    flat = image.reshape(-1, 3)
    labels = np.zeros((flat.shape[0],), dtype=np.int32)

    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    for color_index, color in enumerate(unique_colors):
        color_tuple = tuple(color.tolist())
        if color_tuple not in color_to_label:
            raise ValueError(
                f"Unknown color {color_tuple} found in {image_path}. "
                "This image does not match the id2rgb mapping used by render.py."
            )
        labels[inverse == color_index] = color_to_label[color_tuple]

    return labels.reshape(image.shape[:2])


def collect_pairs(gt_dir, pred_dir):
    gt_paths = sorted(path for path in gt_dir.iterdir() if path.is_file())
    pairs = []
    for gt_path in gt_paths:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        pairs.append((gt_path, pred_path))
    return pairs


def update_confusion_matrix(confusion_matrix, gt_labels, pred_labels, ignore_label):
    valid_mask = np.ones_like(gt_labels, dtype=bool)
    if ignore_label is not None:
        valid_mask &= gt_labels != ignore_label

    gt_flat = gt_labels[valid_mask].reshape(-1)
    pred_flat = pred_labels[valid_mask].reshape(-1)

    for gt_class, pred_class in zip(gt_flat, pred_flat):
        confusion_matrix[gt_class, pred_class] += 1


def compute_metrics(confusion_matrix, ignore_label):
    classes = np.arange(confusion_matrix.shape[0])
    if ignore_label is not None:
        classes = classes[classes != ignore_label]

    class_iou = {}
    class_accuracy = {}
    class_dice = {}

    total_correct = 0
    total_pixels = 0
    fw_iou_numerator = 0.0

    valid_ious = []
    valid_accuracies = []
    valid_dices = []

    for class_id in classes:
        tp = confusion_matrix[class_id, class_id]
        gt_count = confusion_matrix[class_id, :].sum()
        pred_count = confusion_matrix[:, class_id].sum()
        union = gt_count + pred_count - tp

        total_correct += tp
        total_pixels += gt_count

        if gt_count > 0:
            acc = tp / gt_count
            class_accuracy[int(class_id)] = float(acc)
            valid_accuracies.append(acc)

        if union > 0:
            iou = tp / union
            class_iou[int(class_id)] = float(iou)
            valid_ious.append(iou)
            fw_iou_numerator += gt_count * iou

        if gt_count + pred_count > 0:
            dice = (2 * tp) / (gt_count + pred_count)
            class_dice[int(class_id)] = float(dice)
            valid_dices.append(dice)

    metrics = {
        "pixel_accuracy": float(total_correct / total_pixels) if total_pixels > 0 else 0.0,
        "mean_accuracy": float(np.mean(valid_accuracies)) if valid_accuracies else 0.0,
        "mean_iou": float(np.mean(valid_ious)) if valid_ious else 0.0,
        "frequency_weighted_iou": float(fw_iou_numerator / total_pixels) if total_pixels > 0 else 0.0,
        "mean_dice": float(np.mean(valid_dices)) if valid_dices else 0.0,
        "per_class_iou": class_iou,
        "per_class_accuracy": class_accuracy,
        "per_class_dice": class_dice,
    }
    return metrics


def evaluate_method(method_dir, max_num_classes, ignore_label):
    gt_dir = method_dir / "gt_objects_color"
    pred_dir = method_dir / "objects_pred"
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth folder not found: {gt_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction folder not found: {pred_dir}")

    color_to_label = build_color_map(max_num_classes)
    pairs = collect_pairs(gt_dir, pred_dir)
    confusion_matrix = np.zeros((max_num_classes + 1, max_num_classes + 1), dtype=np.int64)
    per_view = {}

    for gt_path, pred_path in pairs:
        gt_labels = image_to_labels(gt_path, color_to_label)
        pred_labels = image_to_labels(pred_path, color_to_label)
        if gt_labels.shape != pred_labels.shape:
            raise ValueError(
                f"Shape mismatch for {gt_path.name}: "
                f"gt {gt_labels.shape} vs pred {pred_labels.shape}"
            )

        image_confusion = np.zeros_like(confusion_matrix)
        update_confusion_matrix(image_confusion, gt_labels, pred_labels, ignore_label)
        update_confusion_matrix(confusion_matrix, gt_labels, pred_labels, ignore_label)
        per_view[gt_path.name] = compute_metrics(image_confusion, ignore_label)

    return compute_metrics(confusion_matrix, ignore_label), per_view


def evaluate(model_paths, split, method, max_num_classes, ignore_label):
    full_results = {}
    per_view_results = {}

    for model_path in model_paths:
        model_dir = Path(model_path)
        split_dir = model_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        full_results[model_path] = {}
        per_view_results[model_path] = {}

        if method:
            method_names = [method]
        else:
            method_names = sorted(path.name for path in split_dir.iterdir() if path.is_dir())

        for method_name in method_names:
            method_dir = split_dir / method_name
            metrics, per_view = evaluate_method(method_dir, max_num_classes, ignore_label)
            full_results[model_path][method_name] = metrics
            per_view_results[model_path][method_name] = per_view

            print(f"Scene: {model_path}")
            print(f"Method: {method_name}")
            print(f"  Pixel Acc: {metrics['pixel_accuracy']:.7f}")
            print(f"  Mean Acc : {metrics['mean_accuracy']:.7f}")
            print(f"  mIoU     : {metrics['mean_iou']:.7f}")
            print(f"  fwIoU    : {metrics['frequency_weighted_iou']:.7f}")
            print(f"  Mean Dice: {metrics['mean_dice']:.7f}")
            print("")

        result_name = f"segmentation_{split}_results.json"
        per_view_name = f"segmentation_{split}_per_view.json"
        with open(model_dir / result_name, "w") as f:
            json.dump(full_results[model_path], f, indent=2)
        with open(model_dir / per_view_name, "w") as f:
            json.dump(per_view_results[model_path], f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate segmentation masks rendered by Gaussian Grouping")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--max_num_classes", type=int, default=255)
    parser.add_argument("--ignore_label", type=int, default=0)
    args = parser.parse_args()

    evaluate(
        model_paths=args.model_paths,
        split=args.split,
        method=args.method,
        max_num_classes=args.max_num_classes,
        ignore_label=args.ignore_label,
    )
