#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

OPTION_LETTERS = list("ABCDEFGH")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def split_model(model_path: str):
    from transformers import AutoConfig

    device_map = {}
    world_size = max(1, torch.cuda.device_count())
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    num_layers = config.llm_config.num_hidden_layers
    if world_size == 1:
        return "auto"
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            if layer_cnt >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


def load_model(model_path: str, use_flash_attn: bool = False):
    device_map = split_model(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
        local_files_only=True,
        device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False, local_files_only=True
    )
    tokenizer.model_max_length = 32768
    return model, tokenizer


def adapt_image_token_count(model, input_size: int) -> None:
    """Keep InternVL chat placeholders aligned with lowered image resolution."""
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    patch_size = int(getattr(vision_config, "patch_size", getattr(model, "patch_size", 14)) or 14)
    downsample_ratio = float(getattr(config, "downsample_ratio", getattr(model, "downsample_ratio", 0.5)) or 0.5)
    if patch_size <= 0 or input_size <= 0:
        return
    expected = int((input_size // patch_size) ** 2 * (downsample_ratio ** 2))
    current = getattr(model, "num_image_token", None)
    if expected <= 0 or current == expected:
        return
    force_image_size = getattr(config, "force_image_size", None)
    logging.info(
        "Adapting InternVL num_image_token for input_size=%d: %s -> %d "
        "(force_image_size=%s, patch_size=%d, downsample_ratio=%.3f)",
        input_size,
        current,
        expected,
        force_image_size,
        patch_size,
        downsample_ratio,
    )
    model.num_image_token = expected


def extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    letter_class = "A-H"
    patterns = [
        rf"(?:Best\s+)?[Aa]nswer\s*:\s*([{letter_class}])",
        rf"\b([{letter_class}])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()
    return None


def ask_model(model, tokenizer, pixel_values, num_patches_list, question, max_new_tokens=64):
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    response = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )
    return response


def ask_model_textonly(model, tokenizer, question, max_new_tokens=64):
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    response = model.chat(
        tokenizer,
        None,
        question,
        generation_config,
        num_patches_list=[],
        history=None,
        return_history=False,
    )
    return response


def build_base_question(question: str, options: list[str], num_frames: int) -> str:
    allowed_letters = OPTION_LETTERS[: len(options)]
    letter_hint = ", ".join(allowed_letters)
    if num_frames > 0:
        frame_prefix = "".join(f"Frame{i+1}: <image>\n" for i in range(num_frames))
        context_line = "Select the best answer to the following multiple-choice question based on the video.\n"
    else:
        frame_prefix = ""
        context_line = "Select the best answer to the following multiple-choice question.\n"
    option_lines = "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return (
        frame_prefix
        + context_line
        + f"Respond with only the letter ({letter_hint}) of the correct option.\n\n"
        + question
        + "\n"
        + option_lines
    )


def load_cached_frames(cache_dir: Path, video_id: str, num_frames: int):
    if num_frames == 0:
        return [], {"duration_sec": 0.0, "extracted_fps": 0.0, "frame_count": 0}

    frame_dir = cache_dir / video_id
    metadata_path = frame_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata for {video_id}: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No cached frames found for {video_id}")

    total = len(frame_paths)
    if num_frames >= total:
        selected_paths = frame_paths
    elif num_frames == 1:
        selected_paths = [frame_paths[0]]
    else:
        selected_indices = sorted(
            {
                min(total - 1, round(i * (total - 1) / (num_frames - 1)))
                for i in range(num_frames)
            }
        )
        selected_paths = [frame_paths[i] for i in selected_indices]
    return selected_paths, metadata


def preprocess_cached_frames(frame_paths, input_size=448, max_num=1):
    transform = build_transform(input_size=input_size)
    pixel_values_list, num_patches_list = [], []
    for frame_path in frame_paths:
        with Image.open(frame_path) as img:
            tiles = dynamic_preprocess(
                img.convert("RGB"),
                image_size=input_size,
                use_thumbnail=True,
                max_num=max_num,
            )
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list) if pixel_values_list else None
    return pixel_values, num_patches_list


def group_samples_by_video(dataset):
    grouped = defaultdict(list)
    for sample in dataset:
        grouped[sample.get("video_id") or Path(sample["views"]["full"]).stem].append(sample)
    return grouped


def run_base(model, tokenizer, sample, pixel_values, num_patches_list, num_frames: int):
    question = build_base_question(sample["question"], sample["options"], num_frames)
    if num_frames == 0:
        text = ask_model_textonly(model, tokenizer, question)
        pred = extract_answer(text)
        return {"raw_text": text, "pred_letter": pred}

    pv = pixel_values.to(torch.bfloat16).cuda()
    try:
        text = ask_model(model, tokenizer, pv, num_patches_list, question)
    finally:
        del pv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pred = extract_answer(text)
    return {"raw_text": text, "pred_letter": pred}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--method", choices=["base"], required=True)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--num-rows", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--use-flash-attn", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dataset = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    if args.start_index > 0 and args.num_rows:
        dataset = dataset[args.start_index: args.start_index + args.num_rows]
        args.start_index = 0
    elif args.num_rows:
        dataset = dataset[:args.num_rows]

    model, tokenizer = load_model(args.model_path, use_flash_attn=args.use_flash_attn)
    adapt_image_token_count(model, args.input_size)
    cache_dir = Path(args.cache_dir)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    existing_results = []
    if out_path.exists() and args.start_index == 0:
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    existing_ids.add(row["sample_id"])
                    existing_results.append(row)
                except Exception:
                    pass
        if existing_ids:
            logging.info("Resuming: found %d existing results", len(existing_ids))

    grouped = group_samples_by_video(dataset)
    ordered_video_ids = []
    seen = set()
    for sample in dataset:
        video_id = sample.get("video_id") or Path(sample["views"]["full"]).stem
        if video_id not in seen:
            ordered_video_ids.append(video_id)
            seen.add(video_id)

    results = list(existing_results)
    done_count = len(existing_results)
    mode = "a" if existing_ids else "w"

    with out_path.open(mode, encoding="utf-8") as f:
        for video_id in ordered_video_ids:
            group = grouped[video_id]
            pending = [sample for sample in group if sample["sample_id"] not in existing_ids]
            if not pending:
                continue

            try:
                frame_paths, metadata = load_cached_frames(cache_dir, video_id, args.num_frames)
                if args.num_frames == 0:
                    pixel_values, num_patches_list = None, []
                    actual_frame_count = 0
                    logging.info("Video group %s: text-only cached path", video_id)
                else:
                    pixel_values, num_patches_list = preprocess_cached_frames(
                        frame_paths,
                        input_size=args.input_size,
                        max_num=1,
                    )
                    actual_frame_count = len(num_patches_list)
                    logging.info(
                        "Video group %s: cached_frames=%d duration=%.1fs input_size=%d",
                        video_id,
                        actual_frame_count,
                        float(metadata.get("duration_sec", 0.0) or 0.0),
                        args.input_size,
                    )
            except Exception as e:
                logging.error("Error loading cache for %s: %s", video_id, e)
                for sample in pending:
                    row = {
                        "sample_id": sample["sample_id"],
                        "correct_letter": sample["correct_letter"],
                        "task_type": sample.get("task_type", ""),
                        "duration_category": sample.get("duration_category", ""),
                        "video_id": video_id,
                        "error": f"cache_load_error: {e}",
                        "base_correct": 0,
                        "final_correct": 0,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    results.append(row)
                    done_count += 1
                continue

            for sample in pending:
                logging.info(
                    "[%d/%d] %s video=%s frames=%d input_size=%d method=%s",
                    done_count + 1,
                    len(dataset),
                    sample["sample_id"],
                    video_id,
                    actual_frame_count,
                    args.input_size,
                    args.method,
                )
                try:
                    base = run_base(
                        model,
                        tokenizer,
                        sample,
                        pixel_values,
                        num_patches_list,
                        actual_frame_count,
                    )
                    row = {
                        "sample_id": sample["sample_id"],
                        "video_id": video_id,
                        "correct_letter": sample["correct_letter"],
                        "task_type": sample.get("task_type", ""),
                        "duration_category": sample.get("duration_category", ""),
                        "base_pred": base["pred_letter"],
                        "base_text": base["raw_text"],
                        "base_correct": int(base["pred_letter"] == sample["correct_letter"]) if base["pred_letter"] else 0,
                        "final_pred": base["pred_letter"],
                        "final_correct": int(base["pred_letter"] == sample["correct_letter"]) if base["pred_letter"] else 0,
                        "cached_frame_count": actual_frame_count,
                        "input_size": args.input_size,
                    }
                except Exception as e:
                    logging.error("Error on %s: %s", sample["sample_id"], e)
                    row = {
                        "sample_id": sample["sample_id"],
                        "video_id": video_id,
                        "correct_letter": sample["correct_letter"],
                        "task_type": sample.get("task_type", ""),
                        "duration_category": sample.get("duration_category", ""),
                        "error": str(e),
                        "base_correct": 0,
                        "final_correct": 0,
                    }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                results.append(row)
                done_count += 1

                if done_count % 50 == 0:
                    bc = sum(r.get("base_correct", 0) for r in results)
                    fc = sum(r.get("final_correct", 0) for r in results)
                    logging.info(
                        "  -- Progress: %d/%d base=%d/%d=%.3f final=%d/%d=%.3f",
                        done_count,
                        len(dataset),
                        bc,
                        done_count,
                        bc / done_count,
                        fc,
                        done_count,
                        fc / done_count,
                    )

            del pixel_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total = len(results)
    base_correct = sum(r.get("base_correct", 0) for r in results)
    final_correct = sum(r.get("final_correct", 0) for r in results)
    errors = sum(1 for r in results if "error" in r)
    summary = {
        "method": args.method,
        "num_frames": args.num_frames,
        "input_size": args.input_size,
        "total": total,
        "errors": errors,
        "base_correct": base_correct,
        "base_acc": base_correct / total if total else 0.0,
        "final_correct": final_correct,
        "final_acc": final_correct / total if total else 0.0,
        "delta": final_correct - base_correct,
    }
    summary["base_pred_distribution"] = dict(Counter(r.get("base_pred", "?") for r in results if "base_pred" in r))
    by_type = defaultdict(lambda: {"total": 0, "base": 0, "final": 0})
    for r in results:
        tt = r.get("task_type", "unknown")
        by_type[tt]["total"] += 1
        by_type[tt]["base"] += r.get("base_correct", 0)
        by_type[tt]["final"] += r.get("final_correct", 0)
    summary["by_task_type"] = dict(by_type)
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Summary: %s", out_path.with_suffix(".summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
