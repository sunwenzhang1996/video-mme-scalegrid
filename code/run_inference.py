#!/usr/bin/env python3
"""
Cached Video-MME evaluation.

Uses offline frame cache instead of online video decoding. The input dataset
remains sample-level JSON, but samples are grouped by video_id so one cached
frame sequence can be reused for all questions tied to the same video.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

OPTION_LETTERS = list("ABCDEFGH")
INSTRUCT_PROMPT_V2 = (
    "Select the best answer to the following multiple-choice question based on the video.\n"
    "Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option."
)
MODEL_CLASS_MAP = {
    "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
    "qwen3_vl": Qwen3VLForConditionalGeneration,
}


def load_model(model_path, model_family="qwen2_5_vl"):
    model_cls = MODEL_CLASS_MAP[model_family]
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = 3136
    model.eval()
    return model, processor


def build_option_block(options):
    return "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))


def build_mcqa_prompt(question, options):
    allowed_letters = OPTION_LETTERS[: len(options)]
    letter_hint = ", ".join(allowed_letters)
    lines = [
        "Select the best answer to the following multiple-choice question based on the video.",
        f"Respond with only the letter ({letter_hint}) of the correct option.",
        "",
        question,
    ]
    for i, opt in enumerate(options):
        lines.append(f"{OPTION_LETTERS[i]}. {opt}")
    return "\n".join(lines)


def build_videomme_v2_prompt(question, options, subtitle_text="", subtitle_mode="none"):
    if subtitle_mode in {"concat", "only"}:
        prefix = (
            "These are the frames of a video.\n"
            f"This video's subtitles are listed below:\n{subtitle_text}\n"
        )
    elif subtitle_mode == "interleave":
        prefix = (
            "These are the frames of a video with corresponding subtitles shown between frames.\n"
            "The subtitles indicate what is being said during the time interval between adjacent frames.\n"
            f"{subtitle_text}\n"
        )
    else:
        prefix = "These are the frames of a video."

    return "\n".join(
        [
            prefix,
            "",
            f"Question: {question}",
            build_option_block(options),
            INSTRUCT_PROMPT_V2,
        ]
    )


def extract_answer_from_text(text):
    letter_class = "A-H"
    patterns = [
        rf"(?:Best\s+)?[Aa]nswer\s*:\s*([{letter_class}])",
        rf"\b([{letter_class}])\s*$",
        rf"\b([{letter_class}])\s*[.\)]",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text or "")
        if matches:
            return matches[-1].upper()
    return None


def build_video_content(video_frames, sample_fps, raw_fps, max_pixels):
    return {
        "type": "video",
        "video": video_frames,
        "sample_fps": sample_fps,
        "raw_fps": raw_fps,
        "min_pixels": min(max_pixels, 3136),
        "max_pixels": max_pixels,
    }


def build_messages(prompt, video_frames, sample_fps, raw_fps, max_pixels):
    content = []
    if video_frames:
        content.append(build_video_content(video_frames, sample_fps, raw_fps, max_pixels))
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def load_subtitle_entries(subtitle_dir: Path | None, video_id: str):
    if subtitle_dir is None:
        return []
    path = subtitle_dir / f"{video_id}.jsonl"
    if not path.exists():
        logging.warning(f"Missing subtitle for {video_id}: {path}")
        return []
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            entries.append(
                {
                    "text": text,
                    "start_time": float(row.get("start_time", 0.0) or 0.0),
                    "end_time": float(row.get("end_time", 0.0) or 0.0),
                }
            )
    return entries


def subtitle_concat_all(entries, max_chars=12000):
    text = " ".join(entry["text"] for entry in entries).strip()
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half].rstrip() + "\n...[subtitle truncated]...\n" + text[-half:].lstrip()


def group_subtitle_segments(entries, gap_threshold=0.5):
    if not entries:
        return []
    segments = []
    current = [entries[0]]
    for prev, cur in zip(entries, entries[1:]):
        gap = cur["start_time"] - prev["end_time"]
        prev_ends_sentence = prev["text"].rstrip().endswith((".", "!", "?"))
        if gap > gap_threshold or (prev_ends_sentence and gap > 0.1):
            segments.append(
                {
                    "text": " ".join(item["text"] for item in current),
                    "start_time": current[0]["start_time"],
                    "end_time": current[-1]["end_time"],
                }
            )
            current = [cur]
        else:
            current.append(cur)
    if current:
        segments.append(
            {
                "text": " ".join(item["text"] for item in current),
                "start_time": current[0]["start_time"],
                "end_time": current[-1]["end_time"],
            }
        )
    return segments


def segments_between_timestamps(segments, start_time, end_time):
    return [
        seg
        for seg in segments
        if seg["end_time"] >= start_time and seg["start_time"] < end_time
    ]


def subtitle_interleave_text(entries, frame_timestamps, duration_sec, max_chars=12000):
    segments = group_subtitle_segments(entries)
    if not segments:
        return ""
    lines = []
    for idx, start in enumerate(frame_timestamps):
        end = frame_timestamps[idx + 1] if idx + 1 < len(frame_timestamps) else duration_sec
        for seg in segments_between_timestamps(segments, start, end):
            lines.append(
                f"[Subtitle {seg['start_time']:.2f}s - {seg['end_time']:.2f}s]: {seg['text']}"
            )
    text = "\n".join(lines).strip()
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half].rstrip() + "\n...[subtitle truncated]...\n" + text[-half:].lstrip()


def run_text_generation(model, processor, messages, max_new_tokens=160):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def run_mcqa_cached(
    model,
    processor,
    video_frames,
    question,
    options,
    sample_fps,
    raw_fps,
    max_pixels,
    prompt_style="cached",
    inference_mode="prob",
    subtitle_text="",
    subtitle_mode="none",
):
    if prompt_style == "videomme_v2":
        prompt = build_videomme_v2_prompt(question, options, subtitle_text, subtitle_mode)
    else:
        prompt = build_mcqa_prompt(question, options)
    messages = build_messages(prompt, video_frames, sample_fps, raw_fps, max_pixels)

    if inference_mode == "generate":
        text = run_text_generation(model, processor, messages)
        pred = extract_answer_from_text(text)
        return {
            "pred_letter": pred,
            "option_probs": {},
            "answer_margin": 0.0,
            "raw_text": text,
        }

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if not video_frames:
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :].float().cpu()
    probs = torch.softmax(logits, dim=-1)

    option_probs = {}
    for i, letter in enumerate(OPTION_LETTERS[: len(options)]):
        token_ids = []
        for variant in [letter, f" {letter}", letter.lower(), f" {letter.lower()}"]:
            tids = processor.tokenizer.encode(variant, add_special_tokens=False)
            if len(tids) == 1:
                token_ids.append(tids[0])
        option_probs[letter] = max(float(probs[tid]) for tid in token_ids) if token_ids else 0.0

    ranked = sorted(option_probs.items(), key=lambda x: x[1], reverse=True)
    return {
        "pred_letter": ranked[0][0],
        "option_probs": option_probs,
        "answer_margin": ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1],
    }


def select_frame_indices(total: int, num_frames: int, sampling: str, seed: int, video_id: str) -> list[int]:
    if num_frames >= total:
        return list(range(total))
    if num_frames == 1:
        return [0]
    if sampling == "uniform":
        return sorted(
            {
                min(total - 1, round(i * (total - 1) / (num_frames - 1)))
                for i in range(num_frames)
            }
        )
    if sampling == "dense_start":
        # Keep the same frame count but restrict coverage to the first third.
        end = max(num_frames - 1, (total - 1) // 3)
        return sorted(
            {
                min(end, round(i * end / (num_frames - 1)))
                for i in range(num_frames)
            }
        )
    if sampling == "random":
        stable = sum((i + 1) * ord(ch) for i, ch in enumerate(video_id))
        rng = random.Random(seed + stable)
        return sorted(rng.sample(range(total), k=min(num_frames, total)))
    raise ValueError(f"unknown frame sampling strategy: {sampling}")


def load_cached_video(cache_dir: Path, video_id: str, num_frames: int, sampling: str = "uniform", seed: int = 42):
    if num_frames == 0:
        return [], 0.0, 0.0, {"duration_sec": 0.0, "extracted_fps": 0.0, "frame_count": 0}

    frame_dir = cache_dir / video_id
    metadata_path = frame_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata for {video_id}: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No cached frames found for {video_id}")

    total = len(frame_paths)
    selected_indices = select_frame_indices(total, num_frames, sampling, seed, video_id)
    metadata["selected_frame_indices"] = selected_indices
    selected_paths = [frame_paths[i] for i in selected_indices]

    frames = []
    for path in selected_paths:
        with Image.open(path) as img:
            frames.append(img.convert("RGB").copy())

    duration = float(metadata.get("duration_sec", 0.0) or 0.0)
    raw_fps = float(metadata.get("extracted_fps", 1.0) or 1.0)
    sample_fps = max(len(frames) / duration, 1e-6) if duration > 0 else raw_fps
    timestamps = metadata.get("timestamps_sec") or []
    if timestamps:
        metadata["selected_timestamps_sec"] = [
            float(timestamps[min(idx, len(timestamps) - 1)]) for idx in selected_indices
        ]
    else:
        metadata["selected_timestamps_sec"] = [
            idx / raw_fps if raw_fps > 0 else 0.0 for idx in selected_indices
        ]
    return frames, sample_fps, raw_fps, metadata


def group_samples_by_video(dataset):
    grouped = defaultdict(list)
    for sample in dataset:
        grouped[sample.get("video_id") or Path(sample["views"]["full"]).stem].append(sample)
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--model-family",
        choices=["qwen2_5_vl", "qwen3_vl"],
        default="qwen2_5_vl",
        help="Model architecture family for loader dispatch.",
    )
    parser.add_argument("--method", choices=["base"], required=True)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=151200)
    parser.add_argument(
        "--frame-sampling",
        choices=["uniform", "dense_start", "random"],
        default="uniform",
        help="Frame selection strategy over cached frames.",
    )
    parser.add_argument("--frame-sampling-seed", type=int, default=42)
    parser.add_argument("--num-rows", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--prompt-style",
        choices=["cached", "videomme_v2"],
        default="cached",
        help="Use the historical cached prompt or the official Video-MME-v2 prompt shape.",
    )
    parser.add_argument(
        "--inference-mode",
        choices=["prob", "generate"],
        default="prob",
        help="Select next-token option probability or generation+regex extraction.",
    )
    parser.add_argument(
        "--subtitle-mode",
        choices=["none", "concat", "interleave", "only"],
        default="none",
        help="Subtitle injection mode for V2 protocol checks.",
    )
    parser.add_argument("--subtitle-dir", default=None)
    parser.add_argument("--max-subtitle-chars", type=int, default=12000)
    args = parser.parse_args()
    if args.subtitle_mode != "none" and not args.subtitle_dir:
        parser.error("--subtitle-dir is required unless --subtitle-mode=none")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dataset = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))

    if args.start_index > 0 and args.num_rows:
        dataset = dataset[args.start_index : args.start_index + args.num_rows]
        args.start_index = 0
    elif args.num_rows:
        dataset = dataset[: args.num_rows]

    model, processor = load_model(args.model_path, args.model_family)
    cache_dir = Path(args.cache_dir)
    subtitle_dir = Path(args.subtitle_dir) if args.subtitle_dir else None
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    existing_results = []
    if output_path.exists() and args.start_index == 0:
        with output_path.open("r", encoding="utf-8") as f:
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
            logging.info(f"Resuming: found {len(existing_ids)} existing results")

    grouped = group_samples_by_video(dataset)
    ordered_video_ids = []
    seen = set()
    for sample in dataset:
        video_id = sample.get("video_id") or Path(sample["views"]["full"]).stem
        if video_id not in seen:
            ordered_video_ids.append(video_id)
            seen.add(video_id)

    results = list(existing_results)
    mode = "a" if existing_ids else "w"
    done_count = len(existing_results)

    with output_path.open(mode, encoding="utf-8") as f:
        for video_id in ordered_video_ids:
            group = grouped[video_id]
            pending = [sample for sample in group if sample["sample_id"] not in existing_ids]
            if not pending:
                continue

            try:
                frames_to_load = 0 if args.subtitle_mode == "only" else args.max_frames
                if frames_to_load == 0:
                    frames, sample_fps, raw_fps, metadata = load_cached_video(
                        cache_dir, video_id, frames_to_load, args.frame_sampling, args.frame_sampling_seed
                    )
                    logging.info(f"Video group {video_id}: text-only cached path")
                else:
                    frames, sample_fps, raw_fps, metadata = load_cached_video(
                        cache_dir, video_id, frames_to_load, args.frame_sampling, args.frame_sampling_seed
                    )
                    logging.info(
                        f"Video group {video_id}: cached_frames={len(frames)} "
                        f"duration={metadata.get('duration_sec', 0):.1f}s sample_fps={sample_fps:.4f} "
                        f"sampling={args.frame_sampling}"
                    )
            except Exception as e:
                logging.error(f"Error loading cache for {video_id}: {e}")
                for sample in pending:
                    row = {
                        "sample_id": sample["sample_id"],
                        "correct_letter": sample["correct_letter"],
                        "task_type": sample.get("task_type", ""),
                        "duration_category": sample.get("duration_category", ""),
                        "video_id": video_id,
                        "error": f"cache_load_error: {e}",
                        "final_correct": 0,
                        "base_correct": 0,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    results.append(row)
                    done_count += 1
                continue

            for sample in pending:
                subtitle_entries = load_subtitle_entries(subtitle_dir, video_id)
                if args.subtitle_mode == "interleave":
                    subtitle_text = subtitle_interleave_text(
                        subtitle_entries,
                        metadata.get("selected_timestamps_sec") or [],
                        float(metadata.get("duration_sec", 0.0) or 0.0),
                        args.max_subtitle_chars,
                    )
                elif args.subtitle_mode in {"concat", "only"}:
                    subtitle_text = subtitle_concat_all(subtitle_entries, args.max_subtitle_chars)
                else:
                    subtitle_text = ""
                logging.info(
                    f"[{done_count + 1}/{len(dataset)}] {sample['sample_id']} "
                    f"video={video_id} frames={len(frames)} method={args.method}"
                )
                try:
                    base = run_mcqa_cached(
                        model,
                        processor,
                        frames,
                        sample["question"],
                        sample["options"],
                        sample_fps,
                        raw_fps,
                        args.max_pixels,
                        args.prompt_style,
                        args.inference_mode,
                        subtitle_text,
                        args.subtitle_mode,
                    )
                    pred_letter = base["pred_letter"]
                    row = {
                        "sample_id": sample["sample_id"],
                        "video_id": video_id,
                        "correct_letter": sample["correct_letter"],
                        "task_type": sample.get("task_type", ""),
                        "duration_category": sample.get("duration_category", ""),
                        "base_pred": pred_letter,
                        "base_correct": int(pred_letter == sample["correct_letter"]),
                        "base_option_probs": base["option_probs"],
                        "base_margin": base["answer_margin"],
                        "final_pred": pred_letter,
                        "final_correct": int(pred_letter == sample["correct_letter"]),
                        "cached_frame_count": len(frames),
                        "cached_sample_fps": sample_fps,
                        "cached_raw_fps": raw_fps,
                        "prompt_style": args.prompt_style,
                        "inference_mode": args.inference_mode,
                        "subtitle_mode": args.subtitle_mode,
                        "subtitle_char_count": len(subtitle_text),
                        "frame_sampling": args.frame_sampling,
                        "frame_sampling_seed": args.frame_sampling_seed,
                        "selected_frame_indices": metadata.get("selected_frame_indices", []),
                    }
                    if base.get("raw_text") is not None:
                        row["base_text"] = base["raw_text"]
                except Exception as e:
                    logging.error(f"Error on {sample['sample_id']}: {e}")
                    row = {
                        "sample_id": sample["sample_id"],
                        "video_id": video_id,
                        "correct_letter": sample["correct_letter"],
                        "task_type": sample.get("task_type", ""),
                        "duration_category": sample.get("duration_category", ""),
                        "error": str(e),
                        "final_correct": 0,
                        "base_correct": 0,
                    }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                results.append(row)
                done_count += 1

                if done_count % 50 == 0:
                    bc = sum(r.get("base_correct", 0) for r in results)
                    fc = sum(r.get("final_correct", 0) for r in results)
                    logging.info(
                        f"  -- Progress: {done_count}/{len(dataset)} "
                        f"base={bc}/{done_count}={bc / done_count:.3f} "
                        f"final={fc}/{done_count}={fc / done_count:.3f}"
                    )

    total = len(results)
    base_correct = sum(r.get("base_correct", 0) for r in results)
    final_correct = sum(r.get("final_correct", 0) for r in results)
    errors = sum(1 for r in results if "error" in r)
    summary = {
        "method": args.method,
        "total": total,
        "errors": errors,
        "base_correct": base_correct,
        "base_acc": base_correct / total if total else 0.0,
        "final_correct": final_correct,
        "final_acc": final_correct / total if total else 0.0,
        "delta": final_correct - base_correct,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
