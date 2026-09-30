"""
Paired uncertainty quantification: guided (EA) vs unguided (random) search
=========================================================================
Quantifies the uncertainty on the random-vs-EA comparison and tests it for
equivalence. Everything here is offline (CPU only, no GPU, no new model calls)
because the complete 4096-genome oracle already fixes the score of every genome
either search could return.

THE RANDOM ARM IS THE EXACT ORDER-STATISTIC EXPECTATION, NOT A SIMULATION.
Random search's expected best-of-K is computed in closed form to avoid Monte-Carlo
noise, matching the definition used elsewhere in the analysis. Each evolutionary
run is paired with the expected best-of-K at ITS OWN budget:

    d_s = EA_s - E[best-of-K_s]

where K_s is the number of DISTINCT genomes seed s actually scored (244-308) and
E[.] is exact. Pairing is on budget, which is the matching that matters. Because
the reference carries no sampling error, the only uncertainty in d_s is the
evolutionary run's own variability, which is the quantity the test should be about.

A fully simulated arm (draw K_s genomes once per seed, take the max) is also
computed and reported as a robustness check. It is strictly more conservative,
since it adds the random arm's sampling noise on top, and it reaches the same
verdict.

TESTS
  1. Paired t-test on d_s     -> is there a difference?
  2. TOST at margin delta     -> is any difference small enough to call the two
                                 equivalent?
  3. Wilcoxon signed-rank     -> nonparametric backup, because the scores are
                                 multiples of 1/250 and therefore lumpy
  4. Power / MDE              -> could this design have found a difference?

Reads : results/brute_force/brute_force_combined.csv   (the 4096-genome oracle)
        results/evolution/llama/{exact,ordinal}/evolution_convergence_seed*.csv
Writes: results/analysis/paired_ea_vs_random.csv       (per-seed pairs)
        results/analysis/paired_ea_vs_random_tests.csv (test statistics)
        results/analysis/outcome_distribution.csv      (how often each arm lands
                                                        within a band of the optimum)

Run:
    python paired_random_vs_ea.py
    python paired_random_vs_ea.py --delta 0.01
"""

import argparse
import ast
import csv
import glob
import math
import os
import random

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
ORACLE = os.path.join(ROOT, "results", "brute_force", "brute_force_combined.csv")
EVODIR = os.path.join(ROOT, "results", "evolution", "llama")
OUT = os.path.join(ROOT, "results", "analysis")
os.makedirs(OUT, exist_ok=True)

# Which oracle column each mode is judged on, and which logged genome it returns.
MODES = {
    "exact": {"metric": "exact", "genome_col": "best_genome_exact"},
    "ordinal": {"metric": "ordinal", "genome_col": "best_genome_ordinal"},
}
ACTIVE_MODES = ("exact", "ordinal")

# Bands below the optimum used for the outcome distribution. Scores are multiples
# of 1/250, so TOL only guards float comparison, not a real tolerance.
BANDS = (0.0, 0.005, 0.010, 0.020)
TOL = 1e-12


def load_oracle():
    """genome tuple -> (exact, ordinal) for all 4096 genomes."""
    table = {}
    with open(ORACLE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g = tuple(int(x) for x in ast.literal_eval(r["genome"]))
            table[g] = (float(r["exact_accuracy"]), float(r["ordinal_score"]))
    if len(table) != 4096:
        raise SystemExit(f"Expected 4096 oracle rows, found {len(table)}")
    return table


def load_runs(mode):
    """Per-seed (seed, returned genome, distinct-evaluation budget)."""
    col = MODES[mode]["genome_col"]
    runs = []
    pattern = os.path.join(EVODIR, mode, f"evolution_convergence_seed*_{mode}_*.csv")
    for p in sorted(glob.glob(pattern)):
        seed = int(os.path.basename(p).split("seed")[1].split("_")[0])
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        last = rows[-1]
        genome = tuple(int(x) for x in ast.literal_eval(last[col]))
        budget = int(last["n_unique_evals_total"])
        runs.append((seed, genome, budget))
    if not runs:
        raise SystemExit(f"No convergence logs found for mode={mode}")
    return sorted(runs)


def expected_best_of_k(sorted_desc, k, _cache={}):
    """EXACT E[best-of-k] under sampling without replacement, from the complete
    oracle. With scores ranked descending, the i-th ranked score is the maximum
    of the draw with probability [C(N-i+1,k) - C(N-i,k)] / C(N,k). Evaluated in
    log space for numerical stability. This is the same closed form used for the
    search-vs-random curve elsewhere in the analysis."""
    key = (id(sorted_desc), k)
    if key in _cache:
        return _cache[key]
    n = len(sorted_desc)
    k = min(k, n)
    lg = math.lgamma

    def log_comb(a, b):
        if b < 0 or b > a:
            return None
        return lg(a + 1) - lg(b + 1) - lg(a - b + 1)

    log_total = log_comb(n, k)
    exp_val = 0.0
    for i in range(1, n - k + 2):
        upper = log_comb(n - i + 1, k)
        lower = log_comb(n - i, k)
        if upper is None:
            continue
        p = math.exp(upper - log_total)
        if lower is not None:
            p -= math.exp(lower - log_total)
        exp_val += p * sorted_desc[i - 1]
    _cache[key] = exp_val
    return exp_val


def p_best_of_k_at_least(scores, threshold, k):
    """EXACT P(best of k draws >= threshold), sampling without replacement. The
    complement is the probability that all k draws miss the m qualifying genomes,
    C(N-m, k) / C(N, k). Same log-space evaluation as expected_best_of_k."""
    n = len(scores)
    m = sum(1 for s in scores if s >= threshold - TOL)
    if n - m < k:
        return 1.0
    lg = math.lgamma

    def log_comb(a, b):
        return lg(a + 1) - lg(b + 1) - lg(a - b + 1)

    return 1.0 - math.exp(log_comb(n - m, k) - log_comb(n, k))


def outcome_distribution(mode, scores, runs, ea_scores):
    """How often each arm lands within a band of the optimum. The mean of the two
    arms can agree while their spread does not, so the paired test is reported
    alongside this. The random arm is the exact probability at each run's own
    budget, averaged over runs, so both arms are summarised over the same budgets."""
    best = max(scores)
    rows = []
    for eps in BANDS:
        thr = best - eps
        ea_frac = sum(1 for s in ea_scores if s >= thr - TOL) / len(ea_scores)
        rnd_frac = sum(p_best_of_k_at_least(scores, thr, k) for _, _, k in runs) / len(runs)
        rows.append({
            "mode": mode, "band": eps, "threshold": round(thr, 6),
            "ea_fraction": round(ea_frac, 4),
            "random_fraction": round(rnd_frac, 4),
        })
    return rows


def paired_stats(d, delta, alpha=0.05):
    """Every statistic computed for one set of paired differences."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1

    t_stat = mean / se if se else float("nan")
    p_two = float(stats.ttest_1samp(d, 0.0).pvalue)
    c95 = float(stats.t.ppf(0.975, df)) * se
    c90 = float(stats.t.ppf(1 - alpha, df)) * se
    ci90 = (mean - c90, mean + c90)

    # TOST: reject "difference <= -delta" and "difference >= +delta"
    p_lo = float(stats.t.sf((mean + delta) / se, df)) if se else float("nan")
    p_hi = float(stats.t.cdf((mean - delta) / se, df)) if se else float("nan")
    p_tost = max(p_lo, p_hi)

    p_wil = (float("nan") if np.allclose(d, 0)
             else float(stats.wilcoxon(d).pvalue))
    # smallest true difference detectable at 80% power with this design
    mde = float((stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * sd / math.sqrt(n))

    return {
        "mean_difference": mean, "sd_difference": sd, "se_difference": se,
        "t_stat": t_stat, "df": df, "p_paired_t": p_two, "p_wilcoxon": p_wil,
        "cohen_dz": mean / sd if sd else float("nan"),
        "ci95_lo": mean - c95, "ci95_hi": mean + c95,
        "ci90_lo": ci90[0], "ci90_hi": ci90[1],
        "p_tost_lower": p_lo, "p_tost_upper": p_hi, "p_tost": p_tost,
        "equivalent_at_delta": bool(p_tost < alpha),
        "smallest_delta_for_equivalence": max(abs(ci90[0]), abs(ci90[1])),
        "mde_80pct_power": mde,
        "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
        "n_zero": int((d == 0).sum()),
    }


def analyse_mode(mode, oracle, delta, alpha):
    metric_idx = 0 if MODES[mode]["metric"] == "exact" else 1
    scores = [oracle[g][metric_idx] for g in sorted(oracle)]
    sorted_desc = sorted(scores, reverse=True)
    runs = load_runs(mode)

    pair_rows, ea_v, ref_v, sim_v = [], [], [], []
    for seed, genome, budget in runs:
        ea = oracle[genome][metric_idx]
        ref = expected_best_of_k(sorted_desc, budget)          # primary arm
        rng = random.Random(seed)
        sim = max(scores[i] for i in rng.sample(range(len(scores)), budget))
        ea_v.append(ea); ref_v.append(ref); sim_v.append(sim)
        pair_rows.append({
            "mode": mode, "metric": MODES[mode]["metric"], "seed": seed,
            "ea_genome": "".join(str(x) for x in genome),
            "budget_distinct_evals": budget,
            "ea_score": round(ea, 6),
            "expected_best_of_k": round(ref, 6),
            "difference": round(ea - ref, 6),
            "simulated_random_score": round(sim, 6),
            "difference_vs_simulated": round(ea - sim, 6),
        })

    ea = np.array(ea_v); ref = np.array(ref_v); sim = np.array(sim_v)
    primary = paired_stats(ea - ref, delta, alpha)
    robust = paired_stats(ea - sim, delta, alpha)

    res = {
        "mode": mode, "metric": MODES[mode]["metric"], "n_seeds": len(ea),
        "budget_min": int(min(r["budget_distinct_evals"] for r in pair_rows)),
        "budget_max": int(max(r["budget_distinct_evals"] for r in pair_rows)),
        "budget_mean": round(float(np.mean([r["budget_distinct_evals"] for r in pair_rows])), 1),
        "ea_mean": round(float(ea.mean()), 6), "ea_sd": round(float(ea.std(ddof=1)), 6),
        "expected_random_mean": round(float(ref.mean()), 6),
        "expected_random_min": round(float(ref.min()), 6),
        "expected_random_max": round(float(ref.max()), 6),
        "simulated_random_mean": round(float(sim.mean()), 6),
        "simulated_random_sd": round(float(sim.std(ddof=1)), 6),
        "delta": delta,
    }
    for k, v in primary.items():
        res[k] = round(v, 6) if isinstance(v, float) else v
    for k, v in robust.items():
        res["sim_" + k] = round(v, 6) if isinstance(v, float) else v

    outcomes = outcome_distribution(mode, scores, runs, ea_v)
    return pair_rows, res, outcomes


def report(res):
    d = res
    print("=" * 76)
    print(f"MODE {d['mode']}  (judged on {d['metric']})   n = {d['n_seeds']} seeds")
    print(f"  budget per run: {d['budget_min']}-{d['budget_max']} distinct genomes "
          f"(mean {d['budget_mean']})")
    print("=" * 76)
    print(f"  Guided search (EA)          : {d['ea_mean']:.4f} +/- {d['ea_sd']:.4f} (SD)")
    print(f"  Random, EXACT E[best-of-K]  : {d['expected_random_mean']:.4f} "
          f"(range {d['expected_random_min']:.4f}-{d['expected_random_max']:.4f} across budgets)")
    print()
    print(f"  PRIMARY  paired difference (EA - exact expectation)")
    print(f"    mean {d['mean_difference']:+.5f}   SD {d['sd_difference']:.5f}   "
          f"SE {d['se_difference']:.5f}")
    print(f"    t({d['df']}) = {d['t_stat']:+.3f},  p = {d['p_paired_t']:.4f}   "
          f"Wilcoxon p = {d['p_wilcoxon']:.4f}   dz = {d['cohen_dz']:+.3f}")
    print(f"    95% CI [{d['ci95_lo']:+.5f}, {d['ci95_hi']:+.5f}]")
    print(f"    90% CI [{d['ci90_lo']:+.5f}, {d['ci90_hi']:+.5f}]  (the TOST interval)")
    print(f"    TOST at delta={d['delta']}: p = {d['p_tost']:.5f}  -> "
          f"{'EQUIVALENT' if d['equivalent_at_delta'] else 'NOT equivalent'}")
    print(f"    equivalence holds for any margin > {d['smallest_delta_for_equivalence']:.5f}")
    print(f"    detectable at 80% power: {d['mde_80pct_power']:.5f}")
    print(f"    signs: {d['n_positive']} positive, {d['n_negative']} negative, "
          f"{d['n_zero']} zero")
    print()
    print(f"  ROBUSTNESS  same test against one SIMULATED random run per seed")
    print(f"    mean {d['sim_mean_difference']:+.5f}   p = {d['sim_p_paired_t']:.4f}   "
          f"p_TOST = {d['sim_p_tost']:.5f}  -> "
          f"{'EQUIVALENT' if d['sim_equivalent_at_delta'] else 'NOT equivalent'}")
    print(f"    90% CI [{d['sim_ci90_lo']:+.5f}, {d['sim_ci90_hi']:+.5f}]   "
          f"(wider: adds the random arm's own sampling noise)")
    print()


def report_outcomes(rows):
    print("  OUTCOME DISTRIBUTION  runs landing within a band of the optimum")
    print(f"    {'band':>8}  {'EA':>6}  {'random':>7}")
    for r in rows:
        print(f"    {r['band']:>8.3f}  {100 * r['ea_fraction']:>5.0f}%  "
              f"{100 * r['random_fraction']:>6.0f}%")
    print()


def main():
    p = argparse.ArgumentParser(
        description="Paired equivalence test of guided (EA) vs unguided (random) search.")
    p.add_argument("--delta", type=float, default=0.01,
                   help="Equivalence margin in accuracy points (default 0.01 = one "
                        "accuracy point, under a tenth of the 0.117 head-room).")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    oracle = load_oracle()
    all_pairs, all_res, all_outcomes = [], [], []
    print()
    for mode in ACTIVE_MODES:
        pairs, res, outcomes = analyse_mode(mode, oracle, args.delta, args.alpha)
        all_pairs.extend(pairs)
        all_res.append(res)
        all_outcomes.extend(outcomes)
        report(res)
        report_outcomes(outcomes)

    pair_path = os.path.join(OUT, "paired_ea_vs_random.csv")
    with open(pair_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_pairs[0].keys()))
        wtr.writeheader(); wtr.writerows(all_pairs)

    test_path = os.path.join(OUT, "paired_ea_vs_random_tests.csv")
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_res[0].keys()))
        wtr.writeheader(); wtr.writerows(all_res)

    outcome_path = os.path.join(OUT, "outcome_distribution.csv")
    with open(outcome_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_outcomes[0].keys()))
        wtr.writeheader(); wtr.writerows(all_outcomes)

    print(f"Saved: {pair_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
