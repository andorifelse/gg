# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
from os import makedirs
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from render import feature_to_rgb, visualize_obj
from ext.grounded_sam_ours_v2 import florence_sam_output, load_florence2_model


def select_obj_soft_score(logits, text_mask, score_thresh=0.05):
    prob = torch.softmax(logits, dim=0)
    text_mask = text_mask.bool().to(prob.device)

    if text_mask.sum() == 0:
        return torch.empty(0, dtype=torch.long, device=prob.device)

    inside_score = prob[:, text_mask].mean(dim=1)
    if (~text_mask).sum() > 0:
        outside_score = prob[:, ~text_mask].mean(dim=1)
    else:
        outside_score = torch.zeros_like(inside_score)

    class_scores = inside_score - outside_score
    selected_obj_ids = torch.nonzero(class_scores > score_thresh, as_tuple=False).squeeze(1)
    selected_obj_ids = selected_obj_ids[selected_obj_ids != 0]
    return selected_obj_ids


def postprocess_mask(mask):
    if mask.max() == 0:
        return mask

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    classifier,
    florence_model,
    florence_processor,
    sam_predictor,
    text_prompt,
    threshold=0.2,
):
    render_path = os.path.join(model_path, name, f"ours_{iteration}_text_v2", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}_text_v2", "gt")
    colormask_path = os.path.join(model_path, name, f"ours_{iteration}_text_v2", "objects_feature16")
    pred_obj_path = os.path.join(model_path, name, f"ours_{iteration}_text_v2", "test_mask")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)

    results0 = render(views[0], gaussians, pipeline, background)
    rendering0 = results0["render"]
    rendering_obj0 = results0["render_object"]
    logits0 = classifier(rendering_obj0)

    image0 = (rendering0.permute(1, 2, 0) * 255).cpu().numpy().astype("uint8")
    text_mask, annotated_frame_with_mask = florence_sam_output(
        florence_model,
        florence_processor,
        sam_predictor,
        text_prompt,
        image0,
        device="cuda",
    )
    Image.fromarray(annotated_frame_with_mask).save(os.path.join(render_path[:-8], "florence-sam---" + text_prompt + ".png"))
    selected_obj_ids = select_obj_soft_score(logits0, text_mask)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        pred_obj_img_path = os.path.join(pred_obj_path, str(idx))
        makedirs(pred_obj_img_path, exist_ok=True)

        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)

        if len(selected_obj_ids) > 0:
            prob = torch.softmax(logits, dim=0)
            pred_obj_mask = prob[selected_obj_ids, :, :] > threshold
            pred_obj_mask = pred_obj_mask.any(dim=0)
            pred_obj_mask = (pred_obj_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            pred_obj_mask = postprocess_mask(pred_obj_mask)
        else:
            pred_obj_mask = torch.zeros_like(view.objects).cpu().numpy()

        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, "{0:05d}".format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_img_path, text_prompt + ".png"))
        print(os.path.join(pred_obj_img_path, text_prompt + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, florence_model_id: str):
    with torch.no_grad():
        dataset.eval = True
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        num_classes = dataset.num_classes
        print("Num classes: ", num_classes)

        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(
            torch.load(
                os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter), "classifier.pth")
            )
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        florence_model, florence_processor = load_florence2_model(florence_model_id, device="cuda")

        sam_checkpoint = "Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device="cuda")
        sam_predictor = SamPredictor(sam)

        if "figurines" in dataset.model_path:
            positive_input = "green apple;green toy chair;old camera;porcelain hand;red apple;red toy chair;rubber duck with red hat"
        elif "ramen" in dataset.model_path:
            positive_input = "chopsticks;egg;glass of water;pork belly;wavy noodles in bowl;yellow bowl"
        elif "teatime" in dataset.model_path:
            positive_input = "apple;bag of cookies;coffee mug;cookies on a plate;paper napkin;plate;sheep;spoon handle;stuffed bear;tea in a glass"
        else:
            raise NotImplementedError

        positives = positive_input.split(";")
        print("Text prompts:    ", positives)
        print("Florence-2 model:", florence_model_id)

        for text_prompt in positives:
            if not skip_train:
                render_set(
                    dataset.model_path,
                    "train",
                    scene.loaded_iter,
                    scene.getTrainCameras(),
                    gaussians,
                    pipeline,
                    background,
                    classifier,
                    florence_model,
                    florence_processor,
                    sam_predictor,
                    text_prompt,
                )
            if not skip_test:
                render_set(
                    dataset.model_path,
                    "test",
                    scene.loaded_iter,
                    scene.getTestCameras(),
                    gaussians,
                    pipeline,
                    background,
                    classifier,
                    florence_model,
                    florence_processor,
                    sam_predictor,
                    text_prompt,
                )


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script with Florence-2 + SAM")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--florence_model_id", type=str, default="microsoft/Florence-2-large-ft")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.florence_model_id,
    )
