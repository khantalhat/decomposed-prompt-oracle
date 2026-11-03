"""
Single-objective evolutionary search with oracle-lookup fitness (CPU, no GPU)
=============================================================================
Runs the same (mu+lambda) EA as evolve_prompt_llama.py, but scores each genome
by reading the completed 4,096-genome oracle instead of calling the model.

The fitness is the genome's score on the balanced validation set, which the oracle
already holds for every genome, computed once under the pinned float32 setting.
Reading it avoids re-scoring the same genome under greedy inference, and matches
the random-search control, which is also computed from the oracle.

The operators are DEAP's, identical to the GPU script: random genome initialisation,
single-gene mutation, (mu+lambda) selection with selBest, mu=20, lambda=20,
20 generations (0..19).

Reads : results/brute_force/brute_force_combined.csv
Writes (per seed, matching the GPU script's schema so analyze.py needs no change):
        evolution_convergence_seed{S}_{mode}_mu20_lam20_g20.csv
        evolution_eval_cache_seed{S}_{mode}_mu20_lam20_g20.csv
        evolution_llama_seed{S}_{mode}_mu20_lam20_g20.csv

Run:
    python evolve_oracle.py --fitness ordinal --outdir ../../results/evolution/llama/ordinal
    python evolve_oracle.py --fitness exact   --outdir ../../results/evolution/llama/exact_oracle
"""

import argparse
import csv
import os
import random
import time
from typing import Dict, List, Tuple

from deap import base, creator, tools

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "..", "..", "results", "brute_force", "brute_force_combined.csv")
GENE_BOUNDS = [(0, 7), (0, 7), (0, 7), (0, 7)]


def load_oracle() -> Dict[Tuple[int, int, int, int], Tuple[float, float, int]]:
    table = {}
    with open(ORACLE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g = (int(r["intro_idx"]), int(r["task_idx"]),
                 int(r["labels_idx"]), int(r["output_idx"]))
            table[g] = (float(r["exact_accuracy"]), float(r["ordinal_score"]),
                        int(r["n_unrecognized"]))
    if len(table) != 4096:
        raise SystemExit(f"Expected 4096 oracle rows, found {len(table)}")
    return table


def build_toolbox() -> base.Toolbox:
    """Single-objective toolbox, mirroring evolve_prompt_llama.py."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def random_genome() -> List[int]:
        return [random.randint(0, 7) for _ in range(4)]

    toolbox.register("individual", tools.initIterate, creator.Individual, random_genome)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def mutate_one_gene(individual: List[int]):
        gene = random.randrange(len(GENE_BOUNDS))
        lo, hi = GENE_BOUNDS[gene]
        current = individual[gene]
        choices = [v for v in range(lo, hi + 1) if v != current]
        if choices:
            individual[gene] = random.choice(choices)
        del individual.fitness.values
        return individual, gene

    toolbox.register("mutate", mutate_one_gene)
    toolbox.register("select", tools.selBest)
    return toolbox


def run_seed(seed: int, oracle, metric: str, mu: int, lambda_: int,
             n_generations: int, outdir: str, tag_suffix: str):
    random.seed(seed)
    toolbox = build_toolbox()
    idx = 0 if metric == "exact" else 1

    eval_cache: Dict[Tuple[int, ...], Tuple[float, float, int]] = {}

    def fitness_of(ind) -> Tuple[float]:
        key = tuple(ind)
        if key not in eval_cache:
            eval_cache[key] = oracle[key]
        return (eval_cache[key][idx],)

    best = {"exact": -1.0, "ordinal": -1.0, "g_exact": None, "g_ordinal": None}

    def update_best(ind):
        ex, orr, _ = eval_cache[tuple(ind)]
        if ex > best["exact"]:
            best["exact"], best["g_exact"] = ex, list(ind)
        if orr > best["ordinal"]:
            best["ordinal"], best["g_ordinal"] = orr, list(ind)

    conv_rows = []
    hist_rows = []

    t0 = time.time()
    population = toolbox.population(n=mu)
    for ind in population:
        ind.fitness.values = fitness_of(ind)
        update_best(ind)
    population.sort(key=lambda x: x.fitness.values[0], reverse=True)

    def log(gen, elapsed):
        conv_rows.append([gen, best["exact"], best["ordinal"],
                          str(best["g_exact"]), str(best["g_ordinal"]),
                          len(eval_cache), round(elapsed, 4)])
        hist_rows.append([gen, population[0].fitness.values[0],
                          eval_cache[tuple(population[0])][0],
                          eval_cache[tuple(population[0])][1],
                          str(list(population[0])),
                          str([round(i.fitness.values[0], 4) for i in population]),
                          str([list(i) for i in population]),
                          len(eval_cache)])

    log(0, time.time() - t0)

    for gen in range(1, n_generations):
        gs = time.time()
        offspring = []
        for _ in range(lambda_):
            parent = random.choice(population)
            child = toolbox.clone(parent)
            child, _ = toolbox.mutate(child)
            offspring.append(child)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = fitness_of(ind)
                update_best(ind)
        population = toolbox.select(population + offspring, k=mu)
        population.sort(key=lambda x: x.fitness.values[0], reverse=True)
        log(gen, time.time() - gs)

    os.makedirs(outdir, exist_ok=True)
    tag = f"seed{seed}_{tag_suffix}"

    with open(os.path.join(outdir, f"evolution_convergence_{tag}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["gen", "best_exact_so_far", "best_ordinal_so_far",
                    "best_genome_exact", "best_genome_ordinal",
                    "n_unique_evals_total", "gen_time_s"])
        w.writerows(conv_rows)

    with open(os.path.join(outdir, f"evolution_eval_cache_{tag}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["genome", "exact_accuracy", "ordinal_score", "n_unrecognized"])
        for g, (ex, orr, unrec) in eval_cache.items():
            w.writerow([str(list(g)), ex, orr, unrec])

    with open(os.path.join(outdir, f"evolution_llama_{tag}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["gen", "best_fitness", "best_exact", "best_ordinal",
                    "best_genome", "pop_fitness", "pop_genomes",
                    "n_unique_evals_total"])
        w.writerows(hist_rows)

    returned = best["g_exact"] if metric == "exact" else best["g_ordinal"]
    return returned, best[metric], len(eval_cache)


def main():
    p = argparse.ArgumentParser(description="Oracle-fitness (mu+lambda) EA, no GPU.")
    p.add_argument("--fitness", choices=["exact", "ordinal"], default="ordinal")
    p.add_argument("--seeds", default="42-61",
                   help="Inclusive range 'a-b' or a comma list.")
    p.add_argument("--mu", type=int, default=20)
    p.add_argument("--lambda_", type=int, default=20)
    p.add_argument("--generations", type=int, default=20,
                   help="Total INCLUDING gen 0, so 20 gives generations 0..19.")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]

    outdir = args.outdir or os.path.join(HERE, "..", "..", "results",
                                         "evolution", "llama", args.fitness)
    oracle = load_oracle()
    suffix = f"{args.fitness}_mu{args.mu}_lam{args.lambda_}_g{args.generations}"

    print(f"oracle fitness, metric={args.fitness}, mu={args.mu}, "
          f"lambda={args.lambda_}, generations={args.generations}")
    print(f"{'seed':>5} {'returned genome':>16} {'score':>9} {'distinct evals':>15}")
    results = []
    for s in seeds:
        g, v, k = run_seed(s, oracle, args.fitness, args.mu, args.lambda_,
                           args.generations, outdir, suffix)
        results.append((s, g, v, k))
        print(f"{s:>5} {str(g):>16} {v:>9.4f} {k:>15}")

    vals = [v for _, _, v, _ in results]
    ks = [k for _, _, _, k in results]
    mean = sum(vals) / len(vals)
    sd = (sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
    best = max(oracle.values(), key=lambda t: t[0 if args.fitness == "exact" else 1])
    ceiling = best[0 if args.fitness == "exact" else 1]
    print(f"\nmean {mean:.4f}  sd {sd:.4f}  "
          f"reached the optimum {ceiling:.4f} on "
          f"{sum(1 for v in vals if v >= ceiling - 1e-9)} of {len(vals)} seeds")
    print(f"distinct evaluations per run: {min(ks)}-{max(ks)} (mean {sum(ks)/len(ks):.0f})")
    print(f"\nwrote {len(seeds)*3} files to {os.path.normpath(outdir)}")


if __name__ == "__main__":
    main()
