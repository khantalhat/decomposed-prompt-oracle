"""
Offline Random-Search Control (derived from the brute-force oracle)
===================================================================
Because brute force evaluates ALL 4096 genomes, the outcome of a random search
is just "draw k of the 4096 and report the best score among them" — a lookup, no
model needed. So we compute the random-search control here OFFLINE (CPU only, no
GPU), Monte-Carloing many samplings to get the full best-of-k distribution
(mean, std, percentiles) rather than a point estimate from a handful of draws.

This is the CONTROL the evolution result is compared against, at matched budget k.

Reads : ../../results/brute_force/brute_force_combined.csv
Writes: ../../results/baselines/random_search_offline_summary.csv

Run:
    python random_search_offline.py --budget 400 --metric exact
    python random_search_offline.py --budget 400 --metric ordinal --trials 20000
"""

import argparse
import csv
import os
import random
import statistics

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_DIR, "..", "..", "results", "brute_force", "brute_force_combined.csv")
DEFAULT_OUT = os.path.join(_DIR, "..", "..", "results", "baselines", "random_search_offline_summary.csv")


def load_scores(csv_path, metric):
    col = "exact_accuracy" if metric == "exact" else "ordinal_score"
    scores = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores.append(float(row[col]))
    if not scores:
        raise SystemExit(f"No '{col}' values found in {csv_path}")
    return scores


def percentile(sorted_vals, pct):
    """Linear-interpolation percentile (pct in 0..100)."""
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def best_of_k_distribution(scores, k, trials, rng):
    n = len(scores)
    k = min(k, n)
    bests = []
    for _ in range(trials):
        idx = rng.sample(range(n), k)
        bests.append(max(scores[i] for i in idx))
    bests.sort()
    return bests


def summarize(bests, ceiling):
    mean = statistics.fmean(bests)
    std = statistics.pstdev(bests)
    return {
        "mean": mean, "std": std,
        "min": bests[0], "max": bests[-1],
        "p05": percentile(bests, 5),
        "p25": percentile(bests, 25),
        "p50": percentile(bests, 50),
        "p75": percentile(bests, 75),
        "p95": percentile(bests, 95),
        "frac_hit_ceiling": sum(1 for b in bests if b >= ceiling) / len(bests),
    }


def main():
    p = argparse.ArgumentParser(description="Offline random-search control from the brute-force oracle.")
    p.add_argument("--csv", default=DEFAULT_CSV, help="brute_force_combined.csv")
    p.add_argument("--metric", choices=["exact", "ordinal"], default="exact")
    p.add_argument("--budget", type=int, default=400,
                   help="Evolution's budget = mu + lambda*(generations-1) = 20+20*19 = 400.")
    p.add_argument("--trials", type=int, default=10000, help="Monte-Carlo samplings.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--sweep", default="50,100,200,400,800",
                   help="Comma-list of budgets to also report (the random-search curve).")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"Oracle not found: {args.csv}\n"
                         f"Re-run brute force (gpu1) + combine_brute_force.py first.")

    scores = load_scores(args.csv, args.metric)
    n = len(scores)
    ceiling = max(scores)
    rng = random.Random(args.seed)

    print("=" * 70)
    print(f"OFFLINE RANDOM SEARCH  metric={args.metric}  genomes={n}  trials={args.trials}")
    print("=" * 70)
    print(f"Oracle ceiling (max {args.metric}) : {ceiling:.4f}")
    print(f"P(a single random draw of k includes the global optimum) = k/N\n")

    sweep = sorted({int(x) for x in args.sweep.split(",")} | {args.budget})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for k in sweep:
        bests = best_of_k_distribution(scores, k, args.trials, rng)
        s = summarize(bests, ceiling)
        p_opt = min(k, n) / n
        marker = "  <-- evolution budget" if k == args.budget else ""
        print(f"budget k={k:<5} best-of-k: mean={s['mean']:.4f} ± {s['std']:.4f}  "
              f"p50={s['p50']:.4f}  p95={s['p95']:.4f}  "
              f"P(=ceiling)={s['frac_hit_ceiling']*100:5.1f}%  "
              f"P(opt in sample)={p_opt*100:4.1f}%{marker}")
        rows.append({
            "metric": args.metric, "budget": k, "trials": args.trials,
            "ceiling": round(ceiling, 6),
            "mean": round(s["mean"], 6), "std": round(s["std"], 6),
            "min": round(s["min"], 6), "max": round(s["max"], 6),
            "p05": round(s["p05"], 6), "p25": round(s["p25"], 6),
            "p50": round(s["p50"], 6), "p75": round(s["p75"], 6),
            "p95": round(s["p95"], 6),
            "frac_hit_ceiling": round(s["frac_hit_ceiling"], 6),
            "p_opt_in_sample": round(p_opt, 6),
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {args.out}")
    print("\nInterpretation: compare evolution at the same budget against the mean and the")
    print("p95 of this best-of-k distribution.")
    print("=" * 70)


if __name__ == "__main__":
    main()
