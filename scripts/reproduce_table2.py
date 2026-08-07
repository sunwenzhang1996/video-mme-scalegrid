#!/usr/bin/env python3
"""Reproduce matched-grid oracle headroom and visual confusion."""
import argparse
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--labels", default="labels/per_item.csv")
args = p.parse_args()

df = pd.read_csv(args.labels)
matched = ["textonly", "16f_151k", "32f_151k", "64f_151k"]
costs = {"textonly": 0, "16f_151k": 16, "32f_151k": 32, "64f_151k": 64}
df = df[df["config"].isin(matched)]

print(f"{'Model':<25}{'Split':<12}{'BestFixed':>10}{'Oracle':>10}{'Headroom':>10}{'Confusion':>10}")
for (model, split), g in df.groupby(["model", "split"]):
    pivot = g.pivot_table(index="sample_id", columns="config", values="correct", aggfunc="max", fill_value=0)
    if not all(c in pivot.columns for c in matched):
        continue
    pivot = pivot[matched]
    best_fixed = pivot.mean().max() * 100
    oracle = pivot.any(axis=1).mean() * 100
    confused = 0
    for _, row in pivot.iterrows():
        if any(row[lo] and not row[hi] for lo in matched for hi in matched if costs[lo] < costs[hi]):
            confused += 1
    confusion = confused / len(pivot) * 100
    print(f"{model:<25}{split:<12}{best_fixed:>10.1f}{oracle:>10.1f}{oracle-best_fixed:>+10.1f}{confusion:>10.1f}")
