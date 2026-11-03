"""
Complete variance decomposition of the oracle (main + 2-way + 3-way + 4-way)
===========================================================================
Reads results/brute_force/brute_force_combined.csv.

The design is a complete 8^4 factorial with one deterministic observation per
cell, so there is no sampling error and no residual: the four orders account for
100% of the variance. With no replication the highest-order term absorbs whatever
is not captured below it.

For each subset S of the four genes, take the mean of every cell defined by S and
subtract the effects of all proper subsets of S, so nothing is counted twice.
SS(S) is the sum of squares of the remainder; its share is SS(S) / SS_total.

Outputs to results/analysis/:
  variance_decomposition.csv   one row per objective x interaction order
  variance_effects.csv         one row per effect (15 per objective)

Run:  python variance_decomposition.py
"""

import csv
import itertools
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
ORACLE = os.path.join(ROOT, "results", "brute_force", "brute_force_combined.csv")
OUT = os.path.join(ROOT, "results", "analysis")
os.makedirs(OUT, exist_ok=True)

GENES = ["intro_idx", "task_idx", "labels_idx", "output_idx"]
NAMES = {"intro_idx": "INTRO", "task_idx": "TASK",
         "labels_idx": "LABELS", "output_idx": "OUTPUT"}
OBJECTIVES = [("exact_accuracy", "exact"), ("ordinal_score", "ordinal")]
N_ALLELES = 8


def load_oracle():
    with open(ORACLE, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != N_ALLELES ** 4:
        raise SystemExit(f"expected {N_ALLELES ** 4} genomes, found {len(rows)}")
    keys = [tuple(int(r[g]) for g in GENES) for r in rows]
    if len(set(keys)) != len(rows):
        raise SystemExit("oracle contains duplicate genomes; decomposition assumes one per cell")
    return rows, keys


def decompose(values, keys):
    """Return (ss_by_order, ss_by_effect, ss_total, max_reconstruction_error)."""
    n = len(values)
    grand = sum(values) / n
    ss_total = sum((v - grand) ** 2 for v in values)

    # effect[S] is a list of length n: the part of each genome's score attributable
    # to subset S beyond everything explained by the proper subsets of S.
    effect = {(): [grand] * n}

    for order in range(1, len(GENES) + 1):
        for subset in itertools.combinations(range(len(GENES)), order):
            totals, counts = {}, {}
            for i, key in enumerate(keys):
                cell = tuple(key[g] for g in subset)
                totals[cell] = totals.get(cell, 0.0) + values[i]
                counts[cell] = counts.get(cell, 0) + 1
            lower = [
                sum(effect[sub][i]
                    for size in range(order)
                    for sub in itertools.combinations(subset, size))
                for i in range(n)
            ]
            effect[subset] = [
                totals[tuple(keys[i][g] for g in subset)] / counts[tuple(keys[i][g] for g in subset)]
                - lower[i]
                for i in range(n)
            ]

    ss_by_effect = {S: sum(x * x for x in effect[S]) for S in effect if S}
    ss_by_order = {}
    for S, ss in ss_by_effect.items():
        ss_by_order[len(S)] = ss_by_order.get(len(S), 0.0) + ss

    # verification: grand mean plus every effect must reconstruct the raw scores
    recon = [sum(effect[S][i] for S in effect) for i in range(n)]
    max_err = max(abs(recon[i] - values[i]) for i in range(n))
    return ss_by_order, ss_by_effect, ss_total, max_err


def n_parameters(order):
    """Free parameters at a given interaction order: C(4, order) * 7^order."""
    from math import comb
    return comb(len(GENES), order) * (N_ALLELES - 1) ** order


def main():
    rows, keys = load_oracle()
    order_rows, effect_rows = [], []

    for column, label in OBJECTIVES:
        values = [float(r[column]) for r in rows]
        by_order, by_effect, ss_total, max_err = decompose(values, keys)

        print(f"\n=== {label.upper()}   total SS = {ss_total:.4f}")
        total_share = 0.0
        for order in range(1, len(GENES) + 1):
            ss = by_order[order]
            share = ss / ss_total
            total_share += share
            df = n_parameters(order)
            print(f"   {order}-way  SS={ss:9.4f}  share={100 * share:5.1f}%  "
                  f"parameters={df:5d} ({100 * df / (N_ALLELES ** 4 - 1):4.1f}% of all)")
            order_rows.append({
                "objective": label, "order": order,
                "n_effects": len(list(itertools.combinations(GENES, order))),
                "n_parameters": df, "ss": round(ss, 6),
                "share": round(share, 6),
            })
        print(f"   shares sum to {100 * total_share:.1f}%  "
              f"(must be 100.0% for a complete factorial with one observation per cell)")
        print(f"   reconstruction check: max |rebuilt - observed| = {max_err:.2e}")

        for S, ss in sorted(by_effect.items(), key=lambda kv: (len(kv[0]), -kv[1])):
            effect_rows.append({
                "objective": label, "order": len(S),
                "effect": " x ".join(NAMES[GENES[g]] for g in S),
                "ss": round(ss, 6), "share": round(ss / ss_total, 6),
            })

    with open(os.path.join(OUT, "variance_decomposition.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["objective", "order", "n_effects",
                                           "n_parameters", "ss", "share"])
        w.writeheader()
        w.writerows(order_rows)

    with open(os.path.join(OUT, "variance_effects.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["objective", "order", "effect", "ss", "share"])
        w.writeheader()
        w.writerows(effect_rows)

    print(f"\nwrote {os.path.join(OUT, 'variance_decomposition.csv')}")
    print(f"wrote {os.path.join(OUT, 'variance_effects.csv')}")
    print("\nCross-check: the 1-way and 2-way shares should reproduce the eta^2 values "
          "already in gene_importance_*.csv and gene_interactions_*.csv.")


if __name__ == "__main__":
    main()
