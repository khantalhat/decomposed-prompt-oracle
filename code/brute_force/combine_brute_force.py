"""
Combine brute force results from both GPUs and produce a ranked summary.

Usage:
    python combine_brute_force.py

Reads:  brute_force_results_part1.csv  (GPU 1, genomes 0–2047)
        brute_force_results_part2.csv  (GPU 2, genomes 2048–4095)
Writes: brute_force_combined.csv       (all 4096, sorted by ordinal_score desc)
"""

import pandas as pd
import os

# Read/write in the paper results tree: code/brute_force/ -> ../../results/brute_force/
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "brute_force")
P1 = os.path.join(_DIR, "brute_force_results_part1.csv")
P2 = os.path.join(_DIR, "brute_force_results_part2.csv")
OUT = os.path.join(_DIR, "brute_force_combined.csv")

parts = []
for path in [P1, P2]:
    if os.path.exists(path):
        df = pd.read_csv(path)
        parts.append(df)
        print(f"Loaded {path}: {len(df)} rows")
    else:
        print(f"WARNING: {path} not found — skipping")

if not parts:
    print("No files found. Exiting.")
    exit(1)

combined = pd.concat(parts, ignore_index=True)
combined = combined.sort_values("ordinal_score", ascending=False).reset_index(drop=True)
combined.insert(0, "rank", combined.index + 1)

combined.to_csv(OUT, index=False)
print(f"\nCombined: {len(combined)} genomes total")
print(f"Saved to: {OUT}")

print("\n--- TOP 10 BY ORDINAL SCORE ---")
cols = ["rank", "genome", "exact_accuracy", "ordinal_score",
        "n_correct", "n_unrecognized", "prompt_preview"]
print(combined[cols].head(10).to_string(index=False))

print("\n--- TOP 10 BY EXACT ACCURACY ---")
top_exact = combined.sort_values("exact_accuracy", ascending=False).head(10)
print(top_exact[cols].to_string(index=False))

print("\n--- SUMMARY STATS ---")
print(f"Max exact accuracy : {combined['exact_accuracy'].max():.4f}")
print(f"Max ordinal score  : {combined['ordinal_score'].max():.4f}")
print(f"Mean exact accuracy: {combined['exact_accuracy'].mean():.4f}")
print(f"Mean ordinal score : {combined['ordinal_score'].mean():.4f}")
print(f"Std exact accuracy : {combined['exact_accuracy'].std():.4f}")
