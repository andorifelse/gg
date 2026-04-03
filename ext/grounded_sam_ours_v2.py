import cv2
import numpy as np
import torch
from PIL import Image


def load_florence2_model(model_id="microsoft/Florence-2-large-ft", device="cuda"):
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Florence-2 is expected to run here with "
            "`transformers==4.49.0` plus `trust_remote_code=True`. "
            "Install it with `pip install transformers==4.49.0 accelerate sentencepiece timm huggingface_hub`."
        ) from exc

    # Florence-2 mixed precision is fragile on this stack. Force float32 first.
    torch_dtype = torch.float32

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model = model.to(device)
    model = model.float()
    model.eval()
    return model, processor


def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")
    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def draw_boxes(image, boxes, labels=None):
    annotated = image.copy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (30, 255, 144), 2)
        if labels is not None and idx < len(labels):
            cv2.putText(
                annotated,
                str(labels[idx]),
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (30, 255, 144),
                1,
                cv2.LINE_AA,
            )
    return annotated


def _extract_detection_result(parsed_answer):
    if "<OPEN_VOCABULARY_DETECTION>" in parsed_answer:
        return parsed_answer["<OPEN_VOCABULARY_DETECTION>"]
    return parsed_answer


def _get_boxes_and_labels(parsed_answer):
    parsed_answer = _extract_detection_result(parsed_answer)
    boxes = parsed_answer.get("bboxes", [])
    labels = parsed_answer.get("labels", [])
    return boxes, labels


def florence_sam_output(
    florence_model,
    florence_processor,
    sam_predictor,
    text_prompt,
    image,
    task_prompt="<OPEN_VOCABULARY_DETECTION>",
    device="cuda",
):
    image_source = image
    pil_image = Image.fromarray(image_source).convert("RGB")

    prompt = f"{task_prompt} {text_prompt}"
    inputs = florence_processor(text=prompt, images=pil_image, return_tensors="pt")
    model_dtype = next(florence_model.parameters()).dtype
    model_device = next(florence_model.parameters()).device
    inputs = {
        k: (
            v.to(device=model_device, dtype=model_dtype)
            if hasattr(v, "to") and torch.is_floating_point(v)
            else v.to(model_device)
            if hasattr(v, "to")
            else v
        )
        for k, v in inputs.items()
    }

    generated_ids = florence_model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=3,
        do_sample=False,
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(pil_image.width, pil_image.height),
    )

    boxes, labels = _get_boxes_and_labels(parsed_answer)
    annotated_frame = draw_boxes(image_source, boxes, labels)

    sam_predictor.set_image(image_source)
    h, w, _ = image_source.shape

    if len(boxes) > 0:
        boxes_xyxy = torch.tensor(boxes, dtype=torch.float32, device=device)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    else:
        masks = torch.zeros((1, 1, h, w), dtype=torch.bool, device=device)

    annotated_frame_with_mask = annotated_frame
    for idx in range(len(masks)):
        annotated_frame_with_mask = show_mask(masks[idx][0].cpu().numpy(), annotated_frame_with_mask)

    return torch.sum(masks, dim=0).squeeze().bool(), annotated_frame_with_mask
