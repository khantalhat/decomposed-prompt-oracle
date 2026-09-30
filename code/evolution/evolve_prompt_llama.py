# -*- coding: utf-8 -*-
"""
Prompt Evolution on Llama 3.2-3B-Instruct  —  matches brute_force_gpu1.py
========================================================================
Runs (mu + lambda) Evolutionary Strategy on the SST-5 sentiment classification
task using the SAME alleles, model, validation set, and prompt-assembly logic
as brute_force_gpu1.py. This guarantees that the evolution convergence curve
can be directly compared against the brute-force upper bound.

Brute-force upper bounds (from combined results):
    Best exact accuracy : 0.5440   genome [3, 0, 4, 7]
    Best ordinal score  : 0.8630   genome [3, 1, 4, 5]
    (Different genomes — real Pareto trade-off exists.)

Four fitness modes supported:
    --fitness exact      single-objective, fitness = exact_accuracy
    --fitness ordinal    single-objective, fitness = ordinal_score
    --fitness weighted   single-objective, fitness = w*exact + (1-w)*ordinal
                         (use --weight to set w; default 0.5)
    --fitness multi      multi-objective NSGA-II over (exact, ordinal)
                         (outputs Pareto front instead of single best)

Usage:
    python evolve_prompt_llama.py --seed 42
    python evolve_prompt_llama.py --seed 43 --fitness ordinal
    python evolve_prompt_llama.py --seed 44 --fitness multi
    python evolve_prompt_llama.py --seed 42 --fitness weighted --weight 0.5

Outputs per seed:
    evolution_llama_seed{seed}_{fitness}.csv             — per-generation log
    evolution_convergence_seed{seed}_{fitness}.csv       — best-so-far curve
    evolution_eval_cache_seed{seed}_{fitness}.csv        — all unique genome evaluations
    evolution_pareto_seed{seed}_multi.csv                — final Pareto front (multi mode only)
"""

import argparse
import csv
import os
import random
import time
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import pipeline
from deap import base, creator, tools

# ---------------------------------------------------------------------------
# Pin the model dtype EXPLICITLY so evolution is measured on the SAME scale as
# the brute-force baseline. The transformers "auto" default silently differs
# across versions (float32 on 4.49, bfloat16 on 4.57), which shifts batched
# predictions and made evolution appear to "beat" the brute-force ceiling.
# float32 is batch-invariant; standardize on it everywhere.
# ---------------------------------------------------------------------------
MODEL_DTYPE = torch.float32

# ============================================================================
# ALLELES — copied verbatim from brute_force_gpu1.py
# ============================================================================

INTRO_ALLELES = [
    "You are an expert sentiment analysis system.",
    "You are a specialist in natural language sentiment analysis with expertise in film criticism.",
    "You have extensive experience reading and classifying movie reviews by sentiment.",
    "Note: in film criticism, strong feelings are often expressed in understated or measured language.",
    "Sentiment in film reviews can be nuanced — a calm or qualified tone does not imply mild sentiment.",
    "Treat this as an ordinal classification task: five sentiment levels from very negative to very positive.",
    "Form your judgment only after reading the review in full — first impressions can be misleading.",
    "",
]

TASK_ALLELES = [
    "Classify the sentiment of the review.",
    "Identify both the direction (positive or negative) and the intensity (mild or strong) of the sentiment.",
    "Determine the reviewer's dominant overall sentiment, not the sentiment of individual phrases.",
    "Distinguish between mild and strong sentiment — both negative and positive have weak and strong forms.",
    "Weigh all evidence in the review and select the label that best represents the overall tone.",
    "Do not anchor on the first phrase — read the entire review and judge the overall picture.",
    "Identify what the reviewer ultimately concludes about the film, not just their reaction to specific scenes.",
    "",
]

LABELS_ALLELES = [
    "Possible labels: very negative, negative, neutral, positive, very positive.",
    ("Labels: very negative, negative, neutral, positive, very positive. "
     "positive means the reviewer genuinely recommends the film despite flaws. "
     "neutral means the reviewer is neither recommending nor discouraging — genuinely on the fence."),
    ("Labels: very negative, negative, neutral, positive, very positive. "
     "very negative=strong disappointment or contempt, even if expressed calmly; "
     "negative=mild or qualified criticism; "
     "neutral=genuinely balanced or ambiguous overall; "
     "positive=genuine approval despite some reservations; "
     "very positive=strong admiration, even if expressed briefly."),
    ("Labels: very negative, negative, neutral, positive, very positive. "
     "very negative means the reviewer strongly dislikes the film, even if expressed in measured language. "
     "negative means mild or partial criticism only."),
    ("Labels: very negative, negative, neutral, positive, very positive. "
     "very positive means strong admiration or delight, even if expressed briefly or with understatement. "
     "positive means genuine appreciation that includes reservations."),
    ("Labels: very negative, negative, neutral, positive, very positive. "
     "Use neutral only when the review is genuinely balanced overall — "
     "not just because it contains some negative phrases alongside a positive conclusion."),
    ("Five levels of sentiment: very negative < negative < neutral < positive < very positive. "
     "Use the extremes when the overall judgment is clearly strong, not just mildly so."),
    ("Labels: very negative, negative, neutral, positive, very positive. "
     "very negative requires strong overall dislike — not just some criticism. "
     "very positive requires strong overall admiration — not just some praise. "
     "neutral requires genuine balance — not a positive conclusion with some reservations."),
]

OUTPUT_ALLELES = [
    "Return only the label.",
    "Do not write any explanation, reasoning, or full sentences. Output the label only.",
    "Answer:",
    "Your entire response must be exactly one of the five labels. Nothing else.",
    "Do not reason aloud. Do not summarise the review. Output the label only.",
    "Sentiment:",
    "Any text beyond the label itself is an error. Output the label only.",
    "",
]

GENE_BOUNDS = [(0, 7)] * 4
GENE_NAMES  = ["INTRO", "TASK", "LABELS", "OUTPUT"]

LABEL_ID_TO_NAME = {
    0: "very negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive",
}

LABEL_PATTERNS = [
    ("VERY NEGATIVE", 0),
    ("VERY POSITIVE", 4),
    ("NEGATIVE",      1),
    ("POSITIVE",      3),
    ("NEUTRAL",       2),
]


# ============================================================================
# PROMPT ASSEMBLY — matches brute_force_gpu1.py exactly
# OUTPUT allele goes AFTER the review text, not inside the instruction
# ============================================================================

def assemble_instruction(genome: List[int]) -> str:
    """INTRO + TASK + LABELS only (OUTPUT placed after review)."""
    parts = [
        INTRO_ALLELES[genome[0]],
        TASK_ALLELES[genome[1]],
        LABELS_ALLELES[genome[2]],
    ]
    return " ".join(p for p in parts if p)


def build_messages(genome: List[int], review_text: str) -> List[Dict]:
    """Chat-format message for Llama 3.2 Instruct.  OUTPUT goes after the review."""
    instruction = assemble_instruction(genome)
    output_fmt  = OUTPUT_ALLELES[genome[3]]

    content = f"{instruction}\n\nReview: {review_text}"
    if output_fmt:
        content += f"\n{output_fmt}"

    return [{"role": "user", "content": content}]


def extract_label(output_text: str) -> int:
    answer = " ".join(output_text.strip().upper().split())
    if not answer:
        return -1
    for label_text, label_id in LABEL_PATTERNS:
        if answer == label_text:
            return label_id
    for label_text, label_id in LABEL_PATTERNS:
        if answer.startswith(label_text):
            return label_id
    for label_text, label_id in LABEL_PATTERNS:
        if label_text in answer:
            return label_id
    return -1


# ============================================================================
# FITNESS FUNCTIONS
# ============================================================================

def exact_accuracy(actuals: List[int], predictions: List[int]) -> float:
    return sum(1 for a, p in zip(actuals, predictions) if a == p) / len(actuals)


def ordinal_score(actuals: List[int], predictions: List[int]) -> float:
    """Linear distance penalty: score = 1 - |actual - predicted| / 4.  Unrecognized -> 0."""
    total = 0.0
    for a, p in zip(actuals, predictions):
        if p == -1:
            total += 0.0
        else:
            total += 1.0 - abs(a - p) / 4.0
    return total / len(actuals)


# ============================================================================
# DATASET — matches brute_force_gpu1.py exactly (no shuffle, sequential per class)
# ============================================================================

def load_balanced_validation(n_per_class: int = 50) -> Tuple[List[Dict], List[int]]:
    print("Loading SST-5 validation set...")
    sst5 = load_dataset("SetFit/sst5")
    val_data = sst5["validation"]

    balanced = []
    for label_id in range(5):
        samples = [r for r in val_data if r["label"] == label_id][:n_per_class]
        balanced.extend(samples)

    ground_truth = [r["label"] for r in balanced]
    print(f"Validation set: {len(balanced)} samples (50 per class, sequential — matches brute force)")
    return balanced, ground_truth


# ============================================================================
# LLM PIPELINE
# ============================================================================

def setup_pipeline(gpu=None):
    """Load the model.  gpu=None -> device_map='auto' (all visible GPUs,
    matches prior 42/43/44 runs).  gpu=N -> pin to a single card cuda:N
    (numerically identical; use to run one seed sweep per GPU)."""
    print("Loading Llama-3.2-3B-Instruct...")
    model_kwargs = dict(
        model="meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=MODEL_DTYPE,
    )
    if gpu is None:
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device"] = gpu
        print(f"  pinned to single GPU: cuda:{gpu}")
    pipe = pipeline("text-generation", **model_kwargs)
    pipe.tokenizer.padding_side = "left"
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    return pipe


def evaluate_genome(
    genome: Tuple[int, ...],
    dataset: List[Dict],
    ground_truth: List[int],
    pipe,
    batch_size: int = 8,
) -> Tuple[float, float, int, List[int], List[str]]:
    """Evaluate a genome on the validation set.

    Returns (exact, ordinal, n_unrecognized, predictions, raw_texts).
    predictions/raw_texts are returned so the caller can record per-sample
    failure cases for the ablation / error analysis without re-running the LLM.
    """
    all_messages = [build_messages(list(genome), r["text"]) for r in dataset]
    outputs = pipe(
        all_messages,
        max_new_tokens=16,
        do_sample=False,
        return_full_text=False,
        batch_size=batch_size,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    raw_texts   = [out[0]["generated_text"].strip() for out in outputs]
    predictions = [extract_label(t) for t in raw_texts]
    ea  = exact_accuracy(ground_truth, predictions)
    os_ = ordinal_score(ground_truth, predictions)
    n_unrec = sum(1 for p in predictions if p == -1)
    return ea, os_, n_unrec, predictions, raw_texts


def collect_failures(
    dataset: List[Dict],
    ground_truth: List[int],
    predictions: List[int],
    raw_texts: List[str],
) -> List[Dict]:
    """Build per-sample failure records for one genome (misclassified samples only)."""
    failures = []
    for i, (pred, raw) in enumerate(zip(predictions, raw_texts)):
        actual = ground_truth[i]
        if pred != actual:
            failures.append({
                "sample_index":    i,
                "text":            dataset[i]["text"],
                "actual_label":    actual,
                "actual_answer":   LABEL_ID_TO_NAME.get(actual, "unknown"),
                "predicted_label": pred,
                "predicted_answer": LABEL_ID_TO_NAME.get(pred, "unrecognized"),
                "model_output":    raw,
            })
    return failures


# ============================================================================
# EVOLUTION  ((mu + lambda) ES)
# ============================================================================

def build_toolbox(multi_objective: bool) -> base.Toolbox:
    """Build DEAP toolbox.  If multi_objective, use FitnessMulti + NSGA-II selection."""
    if multi_objective:
        # Two objectives: maximise exact AND maximise ordinal
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
        if not hasattr(creator, "IndividualMulti"):
            creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)
        individual_cls = creator.IndividualMulti
        select_fn = tools.selNSGA2
    else:
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        individual_cls = creator.Individual
        select_fn = tools.selBest

    toolbox = base.Toolbox()

    def random_genome() -> List[int]:
        return [random.randint(0, 7) for _ in range(4)]

    toolbox.register("individual", tools.initIterate, individual_cls, random_genome)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def mutate_one_gene(individual: List[int]) -> Tuple[List[int], int]:
        gene = random.randrange(len(GENE_BOUNDS))
        lo, hi = GENE_BOUNDS[gene]
        current = individual[gene]
        choices = [v for v in range(lo, hi + 1) if v != current]
        if choices:
            individual[gene] = random.choice(choices)
        del individual.fitness.values
        return individual, gene

    toolbox.register("mutate", mutate_one_gene)
    toolbox.register("select", select_fn)
    return toolbox


def run_evolution(
    toolbox: base.Toolbox,
    dataset: List[Dict],
    ground_truth: List[int],
    pipe,
    mu: int,
    lambda_: int,
    n_generations: int,
    seed: int,
    fitness_metric: str,        # "exact" | "ordinal" | "weighted" | "multi"
    weight: float,              # used only when fitness_metric == "weighted"
    gen_csv_path: str,
    conv_csv_path: str,
    cache_csv_path: str,
    pareto_csv_path: str,       # used only when fitness_metric == "multi"
    failure_csv_path: str,      # per-generation failure cases (for ablation / error analysis)
):
    """(mu + lambda) ES with evaluation caching and convergence tracking.

    Single-objective modes ("exact", "ordinal", "weighted") use selBest.
    Multi-objective mode ("multi") uses NSGA-II and returns a Pareto front.
    """
    random.seed(seed)
    is_multi = (fitness_metric == "multi")

    # Evaluation cache: genome tuple -> (exact, ordinal, n_unrec)
    eval_cache: Dict[Tuple[int, ...], Tuple[float, float, int]] = {}
    # Failure cache: genome tuple -> list of per-sample failure dicts (computed once)
    failure_cache: Dict[Tuple[int, ...], List[Dict]] = {}
    # Flat list of failure rows, tagged with generation + source, written at the end
    failure_records: List[Dict] = []

    def cached_evaluate(genome_list: List[int]) -> Tuple[float, float, int]:
        key = tuple(genome_list)
        if key in eval_cache:
            return eval_cache[key]
        ea, os_, n_unrec, preds, raws = evaluate_genome(key, dataset, ground_truth, pipe)
        eval_cache[key] = (ea, os_, n_unrec)
        failure_cache[key] = collect_failures(dataset, ground_truth, preds, raws)
        return ea, os_, n_unrec

    def log_failures(generation: int, source: str, genome: List[int]):
        """Append this genome's cached failure cases, tagged with generation + source."""
        for f in failure_cache.get(tuple(genome), []):
            failure_records.append({
                "generation": generation,
                "source":     source,
                "genome":     list(genome),
                **f,
            })

    def fitness_of(genome_list: List[int]):
        """Return DEAP-compatible fitness tuple — length 2 for multi, length 1 otherwise."""
        ea, os_, _ = cached_evaluate(genome_list)
        if fitness_metric == "exact":
            return (ea,)
        if fitness_metric == "ordinal":
            return (os_,)
        if fitness_metric == "weighted":
            return (weight * ea + (1.0 - weight) * os_,)
        # multi
        return (ea, os_)

    # --- CSV setup ---
    gen_file = open(gen_csv_path, "w", newline="", encoding="utf-8")
    gen_writer = csv.writer(gen_file, quoting=csv.QUOTE_ALL)
    gen_writer.writerow([
        "gen", "best_fitness", "best_exact", "best_ordinal", "best_genome",
        "pop_fitness", "pop_genomes",
        "offspring_fitness", "offspring_genomes",
        "n_evals_this_gen", "n_unique_evals_total", "gen_time_s",
    ])

    conv_file = open(conv_csv_path, "w", newline="", encoding="utf-8")
    conv_writer = csv.writer(conv_file, quoting=csv.QUOTE_ALL)
    conv_writer.writerow([
        "gen", "best_exact_so_far", "best_ordinal_so_far",
        "best_genome_exact", "best_genome_ordinal",
        "n_unique_evals_total", "gen_time_s",
    ])

    # --- Initial population ---
    print(f"\n[Evolution] seed={seed}  fitness={fitness_metric}  mu={mu}  lambda={lambda_}  gens={n_generations}")
    print("[Evolution] Initializing population...")
    population = toolbox.population(n=mu)

    best_exact_so_far = -1.0
    best_ordinal_so_far = -1.0
    best_genome_exact: List[int] = []
    best_genome_ordinal: List[int] = []

    def update_best(genome: List[int]):
        nonlocal best_exact_so_far, best_ordinal_so_far, best_genome_exact, best_genome_ordinal
        ea, os_, _ = eval_cache[tuple(genome)]
        if ea > best_exact_so_far:
            best_exact_so_far = ea
            best_genome_exact = list(genome)
        if os_ > best_ordinal_so_far:
            best_ordinal_so_far = os_
            best_genome_ordinal = list(genome)

    gen_start = time.time()
    for ind in population:
        ind.fitness.values = fitness_of(ind)
        update_best(ind)
        log_failures(0, "initial_population", ind)
    gen_elapsed = time.time() - gen_start

    if is_multi:
        # NSGA-II needs crowding distances assigned before the first selection
        population = tools.selNSGA2(population, mu)
    else:
        population.sort(key=lambda x: x.fitness.values[0], reverse=True)

    for ind in population:
        log_failures(0, "selected_population", ind)

    best = population[0]
    best_ea, best_os, _ = eval_cache[tuple(best)]
    best_fit_str = (f"({best.fitness.values[0]:.4f}, {best.fitness.values[1]:.4f})"
                    if is_multi else f"{best.fitness.values[0]:.4f}")
    print(f"Gen 0: best_fitness={best_fit_str}  "
          f"exact={best_ea:.4f}  ordinal={best_os:.4f}  "
          f"genome={list(best)}  evals={len(eval_cache)}  time={gen_elapsed:.1f}s")

    def fitness_summary(ind):
        return tuple(round(v, 4) for v in ind.fitness.values)

    gen_writer.writerow([
        0, fitness_summary(best), best_ea, best_os, list(best),
        [fitness_summary(ind) for ind in population],
        [list(ind) for ind in population],
        [], [],
        mu, len(eval_cache), round(gen_elapsed, 2),
    ])
    gen_file.flush()
    conv_writer.writerow([
        0, best_exact_so_far, best_ordinal_so_far,
        best_genome_exact, best_genome_ordinal,
        len(eval_cache), round(gen_elapsed, 2),
    ])
    conv_file.flush()

    # --- Evolution loop ---
    # gen 0 is the initial population (logged above). With --generations N we run
    # generations 0..N-1 (N total), so range(1, N) here. Default N=20 -> gens 0..19.
    for gen in range(1, n_generations):
        gen_start = time.time()

        # Generate offspring via single-gene mutation
        offspring = []
        for _ in range(lambda_):
            parent = random.choice(population)
            child = toolbox.clone(parent)
            child, _gene_idx = toolbox.mutate(child)
            offspring.append(child)

        # Evaluate offspring (cache-aware)
        n_unique_new = 0
        for ind in offspring:
            if not ind.fitness.valid:
                key = tuple(ind)
                cache_hit_before = key in eval_cache
                ind.fitness.values = fitness_of(ind)
                if not cache_hit_before:
                    n_unique_new += 1
                update_best(ind)
            log_failures(gen, "offspring", ind)

        # (mu + lambda) selection
        population = toolbox.select(population + offspring, k=mu)
        if not is_multi:
            population.sort(key=lambda x: x.fitness.values[0], reverse=True)

        for ind in population:
            log_failures(gen, "selected_population", ind)

        gen_elapsed = time.time() - gen_start
        best = population[0]
        best_ea, best_os, _ = eval_cache[tuple(best)]
        best_fit_str = (f"({best.fitness.values[0]:.4f}, {best.fitness.values[1]:.4f})"
                        if is_multi else f"{best.fitness.values[0]:.4f}")

        print(f"Gen {gen}: best_fitness={best_fit_str}  "
              f"exact={best_ea:.4f}  ordinal={best_os:.4f}  "
              f"genome={list(best)}  unique_evals={len(eval_cache)} (+{n_unique_new})  "
              f"time={gen_elapsed:.1f}s")

        gen_writer.writerow([
            gen, fitness_summary(best), best_ea, best_os, list(best),
            [fitness_summary(ind) for ind in population],
            [list(ind) for ind in population],
            [fitness_summary(ind) for ind in offspring],
            [list(ind) for ind in offspring],
            n_unique_new, len(eval_cache), round(gen_elapsed, 2),
        ])
        gen_file.flush()
        conv_writer.writerow([
            gen, best_exact_so_far, best_ordinal_so_far,
            best_genome_exact, best_genome_ordinal,
            len(eval_cache), round(gen_elapsed, 2),
        ])
        conv_file.flush()

    gen_file.close()
    conv_file.close()

    # --- Save evaluation cache (all unique genomes seen) ---
    with open(cache_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["genome", "exact_accuracy", "ordinal_score", "n_unrecognized"])
        for genome_key, (ea, os_, n_unrec) in eval_cache.items():
            w.writerow([list(genome_key), round(ea, 6), round(os_, 6), n_unrec])

    # --- Save failure cases (per generation + source) for ablation / error analysis ---
    # One row per misclassified validation sample, every time a genome appears in a
    # generation as initial_population / offspring / selected_population. Mirrors the
    # failure_cases CSV produced by the older evolve_prompt_alpaca.py.
    with open(failure_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "generation", "source", "genome",
            "sample_index", "text",
            "actual_label", "actual_answer",
            "predicted_label", "predicted_answer",
            "model_output",
        ]
        fw = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        fw.writeheader()
        for rec in failure_records:
            fw.writerow(rec)
    print(f"Failure cases logged: {len(failure_records)} rows -> {failure_csv_path}")

    # --- Save Pareto front (multi-objective only) ---
    # Computed over the entire evaluation cache, not just final population,
    # to give the most accurate Pareto front from everything the search saw.
    if is_multi:
        all_points = [(g, ea, os_) for g, (ea, os_, _) in eval_cache.items()]

        def dominated(p, others):
            ea_p, os_p = p[1], p[2]
            for q in others:
                if q is p:
                    continue
                ea_q, os_q = q[1], q[2]
                if (ea_q >= ea_p and os_q >= os_p) and (ea_q > ea_p or os_q > os_p):
                    return True
            return False

        pareto = [p for p in all_points if not dominated(p, all_points)]
        pareto.sort(key=lambda x: (-x[1], -x[2]))  # by exact desc, then ordinal desc

        with open(pareto_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["genome", "exact_accuracy", "ordinal_score"])
            for g, ea, os_ in pareto:
                w.writerow([list(g), round(ea, 6), round(os_, 6)])

        print(f"\nPareto front: {len(pareto)} non-dominated genomes")
        for g, ea, os_ in pareto[:10]:
            print(f"  genome={list(g)}  exact={ea:.4f}  ordinal={os_:.4f}")
        if len(pareto) > 10:
            print(f"  ... and {len(pareto) - 10} more (see {pareto_csv_path})")

    return population, eval_cache, (best_exact_so_far, best_genome_exact, best_ordinal_so_far, best_genome_ordinal)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int, default=42,
                        help="RNG seed. Any integer; paper uses 42..61 (20 seeds) per objective.")
    parser.add_argument("--fitness",     choices=["exact", "ordinal", "weighted", "multi"],
                        default="exact",
                        help="exact|ordinal|weighted|multi  (multi = NSGA-II Pareto over both)")
    parser.add_argument("--weight",      type=float, default=0.5,
                        help="When --fitness weighted: w*exact + (1-w)*ordinal.  Default 0.5.")
    parser.add_argument("--mu",          type=int, default=20,
                        help="Parent population size.  Default 20 (matches lambda — standard NSGA-II ratio).")
    parser.add_argument("--lambda_",     type=int, default=20,
                        help="Offspring per generation.  Default 20.")
    parser.add_argument("--generations", type=int, default=20,
                        help="Total generations INCLUDING the initial population (gen 0). "
                             "Default 20 -> generations 0..19 (i.e. gen 19 is the 20th generation).")
    parser.add_argument("--gpu",         type=int, default=None,
                        help="Pin the model to a single GPU index (e.g. 0 or 1). "
                             "Omit to use device_map=auto across all visible GPUs "
                             "(default; matches the seed 42/43/44 runs).")
    parser.add_argument("--outdir",      default=".",
                        help="Directory for the output CSVs. Default '.' (current dir). "
                             "The analysis scripts read results/evolution/llama/<mode>/.")
    args = parser.parse_args()

    if args.fitness == "weighted":
        if not (0.0 <= args.weight <= 1.0):
            parser.error("--weight must be in [0, 1]")
        fitness_tag = f"weighted{args.weight:.2f}".replace(".", "p")
    else:
        fitness_tag = args.fitness

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"seed{args.seed}_{fitness_tag}_mu{args.mu}_lam{args.lambda_}_g{args.generations}"
    gen_csv     = os.path.join(args.outdir, f"evolution_llama_{tag}.csv")
    conv_csv    = os.path.join(args.outdir, f"evolution_convergence_{tag}.csv")
    cache_csv   = os.path.join(args.outdir, f"evolution_eval_cache_{tag}.csv")
    pareto_csv  = os.path.join(args.outdir, f"evolution_pareto_{tag}.csv")
    failure_csv = os.path.join(args.outdir, f"evolution_failures_{tag}.csv")

    print("=" * 70)
    print("PROMPT EVOLUTION — Llama-3.2-3B-Instruct — matches brute_force_gpu1.py")
    print("=" * 70)
    print(f"Mode: {args.fitness}" + (f"  (w={args.weight})" if args.fitness == "weighted" else ""))

    dataset, ground_truth = load_balanced_validation(n_per_class=50)
    pipe = setup_pipeline(args.gpu)

    toolbox = build_toolbox(multi_objective=(args.fitness == "multi"))

    population, eval_cache, (best_ea, best_g_ea, best_os, best_g_os) = run_evolution(
        toolbox, dataset, ground_truth, pipe,
        mu=args.mu, lambda_=args.lambda_, n_generations=args.generations,
        seed=args.seed, fitness_metric=args.fitness, weight=args.weight,
        gen_csv_path=gen_csv, conv_csv_path=conv_csv,
        cache_csv_path=cache_csv, pareto_csv_path=pareto_csv,
        failure_csv_path=failure_csv,
    )

    total_evals = len(eval_cache)
    coverage = 100.0 * total_evals / 4096

    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETED")
    print("=" * 70)
    print(f"Seed              : {args.seed}")
    print(f"Fitness metric    : {args.fitness}" + (f" (w={args.weight})" if args.fitness == "weighted" else ""))
    print(f"Configuration     : mu={args.mu}, lambda={args.lambda_}, generations={args.generations}")
    print(f"Unique evaluations: {total_evals} / 4096 ({coverage:.2f}% of search space)")
    print()
    print(f"Best exact-accuracy: {best_ea:.4f}  genome={best_g_ea}")
    print(f"  (brute-force max : 0.5440  genome=[3, 0, 4, 7])")
    print(f"Best ordinal-score : {best_os:.4f}  genome={best_g_os}")
    print(f"  (brute-force max : 0.8630  genome=[3, 1, 4, 5])")
    print()
    print(f"Output files:")
    print(f"  {gen_csv}")
    print(f"  {conv_csv}")
    print(f"  {cache_csv}")
    print(f"  {failure_csv}")
    if args.fitness == "multi":
        print(f"  {pareto_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
