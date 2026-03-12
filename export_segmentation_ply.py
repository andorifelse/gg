import os
import re
import colorsys
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.sh_utils import RGB2SH


def id2rgb(label):
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


def find_latest_iteration(point_cloud_root):
    if not os.path.isdir(point_cloud_root):
        raise FileNotFoundError(f"Point cloud directory not found: {point_cloud_root}")

    max_iter = None
    pattern = re.compile(r"iteration_(\d+)$")
    for name in os.listdir(point_cloud_root):
        full_path = os.path.join(point_cloud_root, name)
        if not os.path.isdir(full_path):
            continue
        match = pattern.match(name)
        if match is None:
            continue
        value = int(match.group(1))
        if max_iter is None or value > max_iter:
            max_iter = value

    if max_iter is None:
        raise RuntimeError(f"No iteration_xxx folders found in: {point_cloud_root}")
    return max_iter


def load_cfg_args(model_path):
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        return None

    with open(cfg_path, "r") as f:
        content = f.read().strip()

    if not content:
        return None

    try:
        cfg = eval(content, {"Namespace": Namespace})
    except Exception as exc:
        raise RuntimeError(f"Failed to parse cfg_args at {cfg_path}: {exc}")

    return cfg


def resolve_sh_degree(model_path, sh_degree_arg):
    if sh_degree_arg is not None:
        return sh_degree_arg

    cfg = load_cfg_args(model_path)
    if cfg is not None and hasattr(cfg, "sh_degree"):
        return cfg.sh_degree

    raise ValueError(
        "Could not determine sh_degree automatically. "
        "Please pass --sh_degree explicitly."
    )


def predict_gaussian_labels(gaussians, classifier):
    with torch.no_grad():
        features_3d = gaussians._objects_dc.permute(2, 0, 1).contiguous().unsqueeze(0)
        logits = classifier(features_3d).squeeze(0)
        probs = torch.softmax(logits, dim=0)

        labels = torch.argmax(probs, dim=0).squeeze(-1)
        confidence = torch.max(probs, dim=0).values.squeeze(-1)

    return labels, confidence


def write_ascii_point_ply(path, xyz, rgb, labels, confidence, opacity):
    num_points = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int label\n")
        f.write("property float confidence\n")
        f.write("property float opacity\n")
        f.write("end_header\n")

        for index in range(num_points):
            x, y, z = xyz[index]
            r, g, b = rgb[index]
            label = int(labels[index])
            conf = float(confidence[index])
            alpha = float(opacity[index])

            f.write(
                f"{x:.6f} {y:.6f} {z:.6f} "
                f"{int(r)} {int(g)} {int(b)} "
                f"{label} {conf:.6f} {alpha:.6f}\n"
            )


def paint_gaussians_with_segmentation(gaussians, rgb):
    device = gaussians._features_dc.device
    rgb_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).to(device)
    sh_dc = RGB2SH(rgb_tensor)

    with torch.no_grad():
        if gaussians._features_dc.shape[1:] == (1, 3):
            gaussians._features_dc.copy_(sh_dc.unsqueeze(1))
        elif gaussians._features_dc.shape[1:] == (3, 1):
            gaussians._features_dc.copy_(sh_dc.unsqueeze(-1))
        else:
            raise RuntimeError(
                f"Unexpected _features_dc shape: {tuple(gaussians._features_dc.shape)}"
            )

        gaussians._features_rest.zero_()


def export_segmentation_ply(model_path, iteration, sh_degree=None, output_dir=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to load the Gaussian model in this repo.")

    point_cloud_root = os.path.join(model_path, "point_cloud")
    if iteration < 0:
        iteration = find_latest_iteration(point_cloud_root)

    iter_dir = os.path.join(point_cloud_root, f"iteration_{iteration}")
    point_cloud_path = os.path.join(iter_dir, "point_cloud.ply")
    classifier_path = os.path.join(iter_dir, "classifier.pth")

    if not os.path.exists(point_cloud_path):
        raise FileNotFoundError(f"Point cloud file not found: {point_cloud_path}")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier file not found: {classifier_path}")

    sh_degree = resolve_sh_degree(model_path, sh_degree)

    if output_dir is None or output_dir == "":
        output_dir = os.path.join(iter_dir, "segmentation_ply")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Gaussian model from: {point_cloud_path}")
    print(f"Loading classifier from: {classifier_path}")
    print(f"Using sh_degree: {sh_degree}")

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(point_cloud_path)

    classifier_state = torch.load(classifier_path, map_location="cuda")
    if "weight" not in classifier_state:
        raise RuntimeError(f"Unexpected classifier state_dict format in: {classifier_path}")

    num_classes = classifier_state["weight"].shape[0]
    in_channels = classifier_state["weight"].shape[1]

    if in_channels != gaussians.num_objects:
        raise RuntimeError(
            f"Classifier input channels ({in_channels}) do not match "
            f"gaussians.num_objects ({gaussians.num_objects})."
        )

    classifier = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1).cuda()
    classifier.load_state_dict(classifier_state)
    classifier.eval()

    labels, confidence = predict_gaussian_labels(gaussians, classifier)

    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    labels_np = labels.detach().cpu().numpy().astype(np.int32)
    confidence_np = confidence.detach().cpu().numpy().astype(np.float32)
    opacity = gaussians.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32)

    rgb = np.stack([id2rgb(int(label)) for label in labels_np], axis=0)

    point_ply_path = os.path.join(output_dir, f"segmentation_points_{iteration}.ply")
    gaussian_ply_path = os.path.join(output_dir, f"segmentation_gaussians_{iteration}.ply")

    write_ascii_point_ply(
        point_ply_path,
        xyz,
        rgb,
        labels_np,
        confidence_np,
        opacity,
    )

    paint_gaussians_with_segmentation(gaussians, rgb)
    gaussians.save_ply(gaussian_ply_path)

    unique_labels, counts = np.unique(labels_np, return_counts=True)
    print(f"Done. Total gaussians: {xyz.shape[0]}")
    print(f"Predicted classes used: {len(unique_labels)}")
    print(f"Saved point cloud PLY: {point_ply_path}")
    print(f"Saved gaussian PLY: {gaussian_ply_path}")
    print("Top class counts:")
    top_pairs = sorted(zip(unique_labels.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)[:10]
    for label, count in top_pairs:
        print(f"  class {label}: {count}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Export 3D segmentation PLY without loading Scene")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=None)
    parser.add_argument("--output_dir", "--output", dest="output_dir", type=str, default="")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    safe_state(args.quiet)

    export_segmentation_ply(
        model_path=args.model_path,
        iteration=args.iteration,
        sh_degree=args.sh_degree,
        output_dir=args.output_dir,
    )
