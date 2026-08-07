#!/usr/bin/env python3
"""Create figure-ready churn decomposition data for one model/split."""
import argparse, json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--labels", default="labels/per_item.csv")
p.add_argument("--model", required=True)
p.add_argument("--split", required=True)
p.add_argument("--output", default=None)
args = p.parse_args()

df = pd.read_csv(args.labels)
df = df[(df.model == args.model) & (df.split == args.split)]
pivot = df.pivot_table(index="sample_id", columns="config", values="correct", aggfunc="max", fill_value=0)
order = [c for c in ["textonly", "8f_151k", "16f_151k", "32f_151k", "64f_151k", "128f_151k", "256f_151k"] if c in pivot.columns]
edges = []
for a, b in zip(order, order[1:]):
    both = pivot[[a, b]]
    edges.append({
        "from": a,
        "to": b,
        "wrong_to_correct": int(((both[a] == 0) & (both[b] == 1)).sum()),
        "correct_to_wrong": int(((both[a] == 1) & (both[b] == 0)).sum()),
        "stable_correct": int(((both[a] == 1) & (both[b] == 1)).sum()),
        "stable_wrong": int(((both[a] == 0) & (both[b] == 0)).sum()),
    })
out = Path(args.output or f"fig_churn_{args.model.replace('/', '_')}_{args.split}.json")
out.write_text(json.dumps({"model": args.model, "split": args.split, "edges": edges}, indent=2))
print(out)
