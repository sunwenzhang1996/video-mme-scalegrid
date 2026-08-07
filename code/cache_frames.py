#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def probe_video(video_path: Path, ffprobe_bin: str) -> dict:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=index,codec_type,r_frame_rate,avg_frame_rate,width,height",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    payload = json.loads(result.stdout)
    duration_sec = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    video_stream = next((s for s in payload.get("streams", []) if s.get("codec_type") == "video"), {})
    return {
        "duration_sec": duration_sec,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "avg_frame_rate": video_stream.get("avg_frame_rate"),
        "r_frame_rate": video_stream.get("r_frame_rate"),
        "codec_name": video_stream.get("codec_name"),
    }


def existing_cache_valid(video_dir: Path, metadata_path: Path) -> tuple[bool, dict | None]:
    if not metadata_path.exists():
        return False, None
    try:
        metadata = json.loads(metadata_path.read_text())
        frame_paths = sorted(video_dir.glob("frame_*.jpg"))
        return len(frame_paths) == int(metadata.get("frame_count", -1)) and len(frame_paths) > 0, metadata
    except Exception:
        return False, None


def build_cache_for_video(
    video_id: str,
    video_path: Path,
    cache_dir: Path,
    target_fps: float,
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> dict:
    video_cache_dir = cache_dir / video_id
    video_cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = video_cache_dir / "metadata.json"

    valid, metadata = existing_cache_valid(video_cache_dir, metadata_path)
    if valid:
        return {
            "status": "exists",
            "video_id": video_id,
            "video_path": str(video_path),
            "frame_count": metadata["frame_count"],
        }

    for frame in video_cache_dir.glob("frame_*.jpg"):
        frame.unlink()

    probe = probe_video(video_path, ffprobe_bin)
    frame_pattern = str(video_cache_dir / "frame_%06d.jpg")
    attempts = []

    base_cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
    ]

    # Some ffmpeg builds do not auto-pick AV1 decoders reliably inside this
    # container, even though explicit decoders work. Prefer explicit decode
    # attempts for AV1 and then fall back to the generic path.
    if probe.get("codec_name") == "av1":
        attempts.append(
            (
                "av1_libdav1d",
                base_cmd
                + ["-c:v", "libdav1d", "-i", str(video_path), "-vf", f"fps={target_fps}", "-q:v", "2", frame_pattern],
            )
        )
        attempts.append(
            (
                "av1_libaom",
                base_cmd
                + ["-c:v", "libaom-av1", "-i", str(video_path), "-vf", f"fps={target_fps}", "-q:v", "2", frame_pattern],
            )
        )

    attempts.append(
        (
            "generic",
            base_cmd + ["-i", str(video_path), "-vf", f"fps={target_fps}", "-q:v", "2", frame_pattern],
        )
    )

    last_error = None
    decode_mode = None
    for attempt_name, cmd in attempts:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60 * 60)
            decode_mode = attempt_name
            last_error = None
            break
        except subprocess.CalledProcessError as exc:
            last_error = exc.stderr[-2000:]
        except subprocess.TimeoutExpired:
            last_error = f"{attempt_name}: ffmpeg timeout"

    if last_error is not None:
        return {
            "status": "failed",
            "video_id": video_id,
            "video_path": str(video_path),
            "codec_name": probe.get("codec_name"),
            "stderr": last_error,
        }

    frame_paths = sorted(video_cache_dir.glob("frame_*.jpg"))
    if not frame_paths:
        return {
            "status": "failed",
            "video_id": video_id,
            "video_path": str(video_path),
            "stderr": "no frames extracted",
        }

    metadata = {
        "video_id": video_id,
        "src_video_path": str(video_path),
        "duration_sec": probe["duration_sec"],
        "extracted_fps": target_fps,
        "decode_mode": decode_mode,
        "frame_count": len(frame_paths),
        "timestamps_sec": [i / target_fps for i in range(len(frame_paths))],
        "width": probe["width"],
        "height": probe["height"],
        "avg_frame_rate": probe["avg_frame_rate"],
        "r_frame_rate": probe["r_frame_rate"],
        "codec_name": probe["codec_name"],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return {
        "status": "built",
        "video_id": video_id,
        "video_path": str(video_path),
        "frame_count": len(frame_paths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset_json).read_text())
    video_dir = Path(args.video_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    videos = {}
    for sample in dataset:
        video_id = sample.get("video_id") or Path(sample["views"]["full"]).stem
        videos.setdefault(video_id, Path(sample["views"]["full"]))

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {
            ex.submit(
                build_cache_for_video,
                video_id,
                video_path,
                cache_dir,
                args.target_fps,
                args.ffmpeg_bin,
                args.ffprobe_bin,
            ): video_id
            for video_id, video_path in sorted(videos.items())
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    status_counts = {}
    for result in results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1

    manifest = {
        "dataset_json": args.dataset_json,
        "video_dir": str(video_dir),
        "cache_dir": str(cache_dir),
        "target_fps": args.target_fps,
        "total_videos": len(videos),
        "status_counts": status_counts,
        "results": sorted(results, key=lambda x: x["video_id"]),
    }
    Path(args.manifest_json).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
