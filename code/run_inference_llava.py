#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

OPTION_LETTERS = list("ABCDEFGH")


def load_model(model_path: str):
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    processor = LlavaNextVideoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


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


def build_messages(prompt, has_video):
    content = []
    if has_video:
        content.append({"type": "video"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def collect_option_token_ids(tokenizer, options):
    token_ids = {}
    for i, letter in enumerate(OPTION_LETTERS[: len(options)]):
        variants = [letter, f" {letter}", letter.lower(), f" {letter.lower()}"]
        ids = []
        for variant in variants:
            tids = tokenizer.encode(variant, add_special_tokens=False)
            if len(tids) == 1:
                ids.append(tids[0])
        token_ids[letter] = sorted(set(ids))
    return token_ids


def run_mcqa_llava(model, processor, video_frames, question, options):
    prompt = build_mcqa_prompt(question, options)
    messages = build_messages(prompt, bool(video_frames))
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    if video_frames:
        inputs = processor(
            text=[text],
            videos=[video_frames],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :].float().cpu()
    probs = torch.softmax(logits, dim=-1)
    token_ids = collect_option_token_ids(processor.tokenizer, options)
    option_probs = {}
    for letter, tids in token_ids.items():
        option_probs[letter] = max(float(probs[tid]) for tid in tids) if tids else 0.0
    ranked = sorted(option_probs.items(), key=lambda x: x[1], reverse=True)

    return {
        "pred_letter": ranked[0][0],
        "option_probs": option_probs,
        "answer_margin": ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1],
    }


def load_cached_video(cache_dir: Path, video_id: str, num_frames: int):
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

    frames = []
    for path in selected_paths:
        with Image.open(path) as img:
            frames.append(img.convert("RGB").copy())
    return frames, metadata


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
    parser.add_argument("--method", choices=["base"], required=True)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--num-rows", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.max_frames > 32:
        logging.warning("LLaVA-NeXT-Video context limit: max-frames capped at 32")
        args.max_frames = 32

    dataset = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    if args.start_index > 0 and args.num_rows:
        dataset = dataset[args.start_index : args.start_index + args.num_rows]
        args.start_index = 0
    elif args.num_rows:
        dataset = dataset[: args.num_rows]

    model, processor = load_model(args.model_path)
    cache_dir = Path(args.cache_dir)
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
                frames, metadata = load_cached_video(cache_dir, video_id, args.max_frames)
                if args.max_frames == 0:
                    logging.info(f"Video group {video_id}: text-only cached path")
                else:
                    logging.info(
                        f"Video group {video_id}: cached_frames={len(frames)} "
                        f"duration={metadata.get('duration_sec', 0):.1f}s"
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
                logging.info(
                    f"[{done_count + 1}/{len(dataset)}] {sample['sample_id']} "
                    f"video={video_id} frames={len(frames)} method={args.method}"
                )
                try:
                    base = run_mcqa_llava(
                        model,
                        processor,
                        frames,
                        sample["question"],
                        sample["options"],
                    )
                    row = {
                        "sample_id": sample["sample_id"],
                        "video_id": video_id,
                        "correct_letter": sample["correct_letter"],
                        "task_type": sample.get("task_type", ""),
                        "duration_category": sample.get("duration_category", ""),
                        "base_pred": base["pred_letter"],
                        "base_correct": int(base["pred_letter"] == sample["correct_letter"]),
                        "base_option_probs": base["option_probs"],
                        "base_margin": base["answer_margin"],
                        "final_pred": base["pred_letter"],
                        "final_correct": int(base["pred_letter"] == sample["correct_letter"]),
                        "cached_frame_count": len(frames),
                    }
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

    total = len(results)
    base_correct = sum(r.get("base_correct", 0) for r in results)
    final_correct = sum(r.get("final_correct", 0) for r in results)
    errors = sum(1 for r in results if "error" in r)
    summary = {
        "method": args.method,
        "max_frames": args.max_frames,
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
