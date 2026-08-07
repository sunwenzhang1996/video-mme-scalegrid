# Stable Curves, Unstable Items: Item-Level Scaling Heterogeneity in Video LLMs


## What's Released

- `labels/per_item.csv` — per-item correctness for frozen Video LLM evaluation cells.
- `labels/derived.csv` — derived annotations: `is_visually_confused`, `is_text_overwritten`, `trajectory_class`, `best_config`, `matched_grid_best_config`.
- `labels/aggregated/` — per-(model, split) JSON: scaling curves, churn/confusion summaries, and release manifest.
- `code/` — cached evaluation pipeline and cascade/timing reference scripts.
- `scripts/reproduce_table2.py` — one-shot script reproducing matched-grid oracle headroom and visual confusion.

## License

- Code: MIT (see `LICENSE`)
- Labels & annotations: CC-BY-4.0 (see `LICENSE-DATA`)
- Underlying videos are not redistributed: download Video-MME / MLVU under their original licenses.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/reproduce_table2.py --labels labels/per_item.csv
python scripts/reproduce_fig_churn.py --model Qwen2.5-VL-7B --split V1_short
```

## Schema (`per_item.csv`)

| Column | Type | Description |
|---|---|---|
| sample_id | str | Video-MME / MLVU question ID |
| model | str | Model family |
| split | str | One of `V1_short`, `V1_medium`, `V2_medium`, `MLVU` |
| config | str | e.g. `16f_151k`, `64f_151k`, `textonly`, or fallback-specific tag |
| frame_count | int | Frames sampled |
| pixel_count | int | Per-frame max pixels or model-default equivalent |
| predicted_option | str | Letter A-H when available |
| ground_truth_option | str | Letter A-H |
| correct | int | 0 or 1 |
| source_tag | str | Normalized source and protocol provenance tag |
| protocol_note | str | Caveats such as fallback input size or LLaVA context cap |
