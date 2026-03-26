import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def generate_interior_regions(gt_obj, erode_ks=5, min_region=20):
    """
    按照你原函数中的逻辑，为每个类别生成内部核心区域

    Args:
        gt_obj: torch.Tensor, [H, W], long
        erode_ks: int, erosion kernel size, odd number
        min_region: int, ignore tiny regions

    Returns:
        results: dict
            {
                cls_id: {
                    "mask": [H, W] bool tensor,
                    "interior": [H, W] bool tensor
                },
                ...
            }
    """
    results = {}
    pad = erode_ks // 2

    present_classes = torch.unique(gt_obj)
    for cls_id in present_classes.tolist():
        mask = (gt_obj == cls_id).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        if mask.sum() < min_region:
            continue

        # erosion(mask) ≈ min-pooling = -max_pool(-x)
        interior = -F.max_pool2d(-mask, kernel_size=erode_ks, stride=1, padding=pad)
        interior = (interior > 0.99).squeeze(0).squeeze(0)  # [H,W] bool

        if interior.sum() < min_region:
            continue

        results[cls_id] = {
            "mask": mask.squeeze(0).squeeze(0).bool(),
            "interior": interior
        }

    return results


def colorize_label_map(label_map):
    """
    给 label map 上色，方便可视化
    label_map: [H, W] numpy array
    """
    h, w = label_map.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)

    rng = np.random.default_rng(12345)
    unique_ids = np.unique(label_map)

    palette = {}
    for cls_id in unique_ids:
        if cls_id == 0:
            palette[cls_id] = np.array([0, 0, 0], dtype=np.uint8)
        else:
            palette[cls_id] = rng.integers(0, 255, size=3, dtype=np.uint8)

    for cls_id in unique_ids:
        color_map[label_map == cls_id] = palette[cls_id]

    return color_map


def overlay_interior_on_mask(mask, interior, mask_color=(180, 180, 180), interior_color=(255, 0, 0)):
    """
    生成 overlay 图：
    - 原 mask 用灰色
    - interior 用红色
    """
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    vis[mask] = np.array(mask_color, dtype=np.uint8)
    vis[interior] = np.array(interior_color, dtype=np.uint8)

    return vis


def show_and_save_results(gt_np, results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 保存整体 GT 彩色图
    gt_color = colorize_label_map(gt_np)
    cv2.imwrite(os.path.join(save_dir, "gt_color.png"), cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR))

    # 画总览图
    n_cls = len(results)
    if n_cls == 0:
        print("No valid interior regions found.")
        return

    fig, axes = plt.subplots(n_cls, 4, figsize=(16, 4 * n_cls))
    if n_cls == 1:
        axes = axes[None, :]

    for row_idx, (cls_id, data) in enumerate(results.items()):
        mask = data["mask"].cpu().numpy()
        interior = data["interior"].cpu().numpy()

        overlay = overlay_interior_on_mask(mask, interior)

        # 保存单类结果
        cv2.imwrite(
            os.path.join(save_dir, f"class_{cls_id}_mask.png"),
            (mask.astype(np.uint8) * 255)
        )
        cv2.imwrite(
            os.path.join(save_dir, f"class_{cls_id}_interior.png"),
            (interior.astype(np.uint8) * 255)
        )
        cv2.imwrite(
            os.path.join(save_dir, f"class_{cls_id}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        axes[row_idx, 0].imshow(gt_color)
        axes[row_idx, 0].set_title(f"GT Color Map")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(mask, cmap="gray")
        axes[row_idx, 1].set_title(f"Class {cls_id} Mask")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(interior, cmap="gray")
        axes[row_idx, 2].set_title(f"Class {cls_id} Interior")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(overlay)
        axes[row_idx, 3].set_title(f"Class {cls_id} Overlay")
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_classes_overview.png"), dpi=200)
    plt.show()


def main():
    # =========================
    # 1. 改这里：你的 GT 路径
    # =========================
    gt_path = "/home/wzc/test/DSCF0920.png"   # 例如一张单通道标签图
    save_dir = "/home/wzc/test/interior_vis_results"

    # 参数
    erode_ks = 5
    min_region = 20

    # =========================
    # 2. 读取 GT
    # =========================
    # 以原始方式读取，确保标签值不被改动
    gt_np = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

    if gt_np is None:
        raise FileNotFoundError(f"Cannot read gt image: {gt_path}")

    # 如果读出来是 3 通道，默认取第一个通道
    if gt_np.ndim == 3:
        print("Warning: gt image has 3 channels, using the first channel as label map.")
        gt_np = gt_np[:, :, 0]

    # 转成 torch.long
    gt_obj = torch.from_numpy(gt_np.astype(np.int64))

    print("GT shape:", gt_obj.shape)
    print("Unique class ids:", torch.unique(gt_obj).tolist())

    # =========================
    # 3. 生成 interior
    # =========================
    results = generate_interior_regions(
        gt_obj=gt_obj,
        erode_ks=erode_ks,
        min_region=min_region
    )

    print(f"Valid classes with interior: {list(results.keys())}")
    for cls_id, data in results.items():
        mask_area = int(data["mask"].sum().item())
        interior_area = int(data["interior"].sum().item())
        print(f"Class {cls_id}: mask area = {mask_area}, interior area = {interior_area}")

    # =========================
    # 4. 可视化 + 保存
    # =========================
    show_and_save_results(gt_np, results, save_dir)
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
