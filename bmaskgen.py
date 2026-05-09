import argparse
import os
import inspect
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamModel, SamProcessor


def load_models(device):
    dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    ).to(device)

    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

    return dino_processor, dino_model, sam_processor, sam_model


def text_to_binary_mask(
    image, text_prompt,
    dino_processor, dino_model, sam_processor, sam_model,
    device, box_threshold=0.3, text_threshold=0.3,
):
    image = image.convert("RGB")
    W, H = image.size

    text_prompt = text_prompt.lower()
    if not text_prompt.endswith("."):
        text_prompt += "."

    # Grounding DINO
    inputs = dino_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    post_process_fn = dino_processor.post_process_grounded_object_detection
    sig_params = inspect.signature(post_process_fn).parameters

    kwargs = {"target_sizes": [(H, W)]}
    if "threshold" in sig_params:
        kwargs["threshold"] = box_threshold
    elif "box_threshold" in sig_params:
        kwargs["box_threshold"] = box_threshold
    if "text_threshold" in sig_params:
        kwargs["text_threshold"] = text_threshold

    if "input_ids" in sig_params:
        results = post_process_fn(outputs, input_ids=inputs.input_ids, **kwargs)[0]
    else:
        results = post_process_fn(outputs, inputs.input_ids, **kwargs)[0]

    boxes = results["boxes"].cpu().numpy().tolist()
    scores = results["scores"].cpu().numpy().tolist()
    labels = results.get("text_labels", results.get("labels", ["?"] * len(boxes)))

    print(f"검출된 객체 수: {len(boxes)}")
    for box, score, label in zip(boxes, scores, labels):
        print(f"  - {label}: {score:.3f}, box={[round(b, 1) for b in box]}")

    if len(boxes) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # SAM
    sam_inputs = sam_processor(image, input_boxes=[boxes], return_tensors="pt").to(device)
    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs, multimask_output=False)

    masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
    )[0]

    masks_np = masks.numpy()[:, 0, :, :]
    combined = np.any(masks_np, axis=0)
    binary_mask = combined.astype(np.uint8) * 255

    assert binary_mask.shape == (H, W), f"Mask {binary_mask.shape} != image {(H, W)}"
    return binary_mask


def main():
    parser = argparse.ArgumentParser(description="Text-prompted binary mask generator (Grounding DINO + SAM)")
    parser.add_argument("--image", "-i", required=True, help="입력 이미지 경로")
    parser.add_argument("--prompt", "-p", required=True, help="텍스트 프롬프트 (예: 'a cat. a dog.')")
    parser.add_argument("--output", "-o", default=None,
                        help="출력 경로. 미지정 시 현재 디렉토리에 '<입력파일명>_mask.png'으로 저장")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.3)
    args = parser.parse_args()


    if args.output is None:
        base = os.path.splitext(os.path.basename(args.image))[0]
        args.output = os.path.join(os.getcwd(), f"{base}_mask.png")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("모델 로드 중...")
    dino_processor, dino_model, sam_processor, sam_model = load_models(device)

    image = Image.open(args.image)
    print(f"입력 이미지: {args.image} (size={image.size})")

    binary_mask = text_to_binary_mask(
        image, args.prompt,
        dino_processor, dino_model, sam_processor, sam_model,
        device, args.box_threshold, args.text_threshold,
    )

    Image.fromarray(binary_mask).save(args.output)
    print(f"저장 완료: {args.output} (size: {binary_mask.shape[::-1]})")


if __name__ == "__main__":
    main()
