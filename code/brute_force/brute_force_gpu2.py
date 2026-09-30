"""
Brute Force Evaluation — GPU 2  (genomes 2048–4095, INTRO alleles 4–7)
=======================================================================
Evaluates the second half of all 4096 genome combinations on the SST-5
validation set using Llama-3.2-3B-Instruct.

Fitness functions:
  - exact_accuracy  : fraction of exactly correct predictions
  - ordinal_score   : 1 - |actual - predicted| / 4  (linear distance penalty)

Run on GPU 2:
    python brute_force_gpu2.py

Output: brute_force_results_part2.csv
"""

import csv
import itertools
import os
import time
import torch
from datasets import load_dataset
from transformers import pipeline

# ---------------------------------------------------------------------------
# Pin the model dtype EXPLICITLY. Do NOT rely on the transformers "auto"
# default: it silently loads float32 on transformers 4.49 but bfloat16 on
# 4.57, which changes batched-inference results across environments.
# float32 is batch-invariant (raising batch_size does not change predictions),
# so we standardize on it for reproducible, comparable results.
# ---------------------------------------------------------------------------
MODEL_DTYPE = torch.float32

# ---------------------------------------------------------------------------
# ALLELES  (redesigned for Llama 3.2 — orthogonal genes)
# ---------------------------------------------------------------------------

INTRO_ALLELES = [
    # 0 — generic expert role
    "You are an expert sentiment analysis system.",
    # 1 — specialist framing with domain
    "You are a specialist in natural language sentiment analysis with expertise in film criticism.",
    # 2 — experience-based framing
    "You have extensive experience reading and classifying movie reviews by sentiment.",
    # 3 — domain knowledge: understatement and irony
    "Note: in film criticism, strong feelings are often expressed in understated or measured language.",
    # 4 — domain knowledge: nuance of tone
    "Sentiment in film reviews can be nuanced — a calm or qualified tone does not imply mild sentiment.",
    # 5 — task-type framing
    "Treat this as an ordinal classification task: five sentiment levels from very negative to very positive.",
    # 6 — process framing
    "Form your judgment only after reading the review in full — first impressions can be misleading.",
    # 7 — empty
    "",
]

TASK_ALLELES = [
    # 0 — minimal anchor
    "Classify the sentiment of the review.",
    # 1 — direction and intensity
    "Identify both the direction (positive or negative) and the intensity (mild or strong) of the sentiment.",
    # 2 — dominant sentiment, not individual phrases
    "Determine the reviewer's dominant overall sentiment, not the sentiment of individual phrases.",
    # 3 — mild vs strong distinction
    "Distinguish between mild and strong sentiment — both negative and positive have weak and strong forms.",
    # 4 — evidence weighting
    "Weigh all evidence in the review and select the label that best represents the overall tone.",
    # 5 — anti-anchoring
    "Do not anchor on the first phrase — read the entire review and judge the overall picture.",
    # 6 — final judgment focus
    "Identify what the reviewer ultimately concludes about the film, not just their reaction to specific scenes.",
    # 7 — empty
    "",
]

LABELS_ALLELES = [
    # 0 — minimal list
    "Possible labels: very negative, negative, neutral, positive, very positive.",
    # 1 — positive-neutral boundary (fixes positive→neutral, 207 cases)
    ("Labels: very negative, negative, neutral, positive, very positive. "
     "positive means the reviewer genuinely recommends the film despite flaws. "
     "neutral means the reviewer is neither recommending nor discouraging — genuinely on the fence."),
    # 2 — full definitions for all five
    ("Labels: very negative, negative, neutral, positive, very positive. "
     "very negative=strong disappointment or contempt, even if expressed calmly; "
     "negative=mild or qualified criticism; "
     "neutral=genuinely balanced or ambiguous overall; "
     "positive=genuine approval despite some reservations; "
     "very positive=strong admiration, even if expressed briefly."),
    # 3 — negative-end boundary (fixes very negative→negative, 218 cases)
    ("Labels: very negative, negative, neutral, positive, very positive. "
     "very negative means the reviewer strongly dislikes the film, even if expressed in measured language. "
     "negative means mild or partial criticism only."),
    # 4 — positive-end boundary (fixes very positive→positive, 148 cases)
    ("Labels: very negative, negative, neutral, positive, very positive. "
     "very positive means strong admiration or delight, even if expressed briefly or with understatement. "
     "positive means genuine appreciation that includes reservations."),
    # 5 — neutral clarification (fixes neutral→negative, 168 cases)
    ("Labels: very negative, negative, neutral, positive, very positive. "
     "Use neutral only when the review is genuinely balanced overall — "
     "not just because it contains some negative phrases alongside a positive conclusion."),
    # 6 — ordinal scale with extremes guidance
    ("Five levels of sentiment: very negative < negative < neutral < positive < very positive. "
     "Use the extremes when the overall judgment is clearly strong, not just mildly so."),
    # 7 — both-boundary combined (covers very negative↔negative and very positive↔positive and neutral)
    ("Labels: very negative, negative, neutral, positive, very positive. "
     "very negative requires strong overall dislike — not just some criticism. "
     "very positive requires strong overall admiration — not just some praise. "
     "neutral requires genuine balance — not a positive conclusion with some reservations."),
]

OUTPUT_ALLELES = [
    # 0 — simple positive instruction
    "Return only the label.",
    # 1 — explicit prohibition
    "Do not write any explanation, reasoning, or full sentences. Output the label only.",
    # 2 — fill-in-the-blank stem (appears after review, model completes it)
    "Answer:",
    # 3 — strict constraint
    "Your entire response must be exactly one of the five labels. Nothing else.",
    # 4 — chain-of-thought suppression
    "Do not reason aloud. Do not summarise the review. Output the label only.",
    # 5 — sentiment stem
    "Sentiment:",
    # 6 — penalty framing
    "Any text beyond the label itself is an error. Output the label only.",
    # 7 — empty
    "",
]

GENE_NAMES   = ["INTRO", "TASK", "LABELS", "OUTPUT"]
ALL_ALLELES  = [INTRO_ALLELES, TASK_ALLELES, LABELS_ALLELES, OUTPUT_ALLELES]
LABEL_NAMES  = ["very negative", "negative", "neutral", "positive", "very positive"]
LABEL_PATTERNS = [
    ("VERY NEGATIVE", 0),
    ("VERY POSITIVE",  4),
    ("NEGATIVE",       1),
    ("POSITIVE",       3),
    ("NEUTRAL",        2),
]

# ---------------------------------------------------------------------------
# SPLIT CONFIG
# ---------------------------------------------------------------------------

ALL_GENOMES   = list(itertools.product(range(8), range(8), range(8), range(8)))
START_IDX     = 2048
END_IDX       = 4096        # GPU 2 handles second half
# Write straight into the paper results tree: code/brute_force/ -> ../../results/brute_force/
_RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "brute_force")
os.makedirs(_RESULTS_DIR, exist_ok=True)
OUTPUT_CSV    = os.path.join(_RESULTS_DIR, "brute_force_results_part2.csv")
BATCH_SIZE    = 8

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def assemble_instruction(genome):
    """Concatenate INTRO + TASK + LABELS (not OUTPUT — placed after review)."""
    parts = [
        INTRO_ALLELES[genome[0]],
        TASK_ALLELES[genome[1]],
        LABELS_ALLELES[genome[2]],
    ]
    return " ".join(p for p in parts if p)


def build_messages(genome, review_text):
    """Chat-format message for Llama 3.2 Instruct. OUTPUT allele goes after review."""
    instruction = assemble_instruction(genome)
    output_fmt  = OUTPUT_ALLELES[genome[3]]

    content = f"{instruction}\n\nReview: {review_text}"
    if output_fmt:
        content += f"\n{output_fmt}"

    return [{"role": "user", "content": content}]


def extract_label(output_text):
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


def exact_accuracy(actuals, predictions):
    correct = sum(1 for a, p in zip(actuals, predictions) if a == p)
    return correct / len(actuals)


def ordinal_score(actuals, predictions):
    """Linear distance penalty: score = 1 - |actual - predicted| / 4."""
    total = 0.0
    for a, p in zip(actuals, predictions):
        if p == -1:
            total += 0.0  # unrecognized = maximum penalty
        else:
            total += 1.0 - abs(a - p) / 4.0
    return total / len(actuals)


def per_class_correct(actuals, predictions):
    return [
        sum(1 for a, p in zip(actuals, predictions) if a == label_id and p == label_id)
        for label_id in range(5)
    ]

# ---------------------------------------------------------------------------
# RESUME SUPPORT
# ---------------------------------------------------------------------------

def load_done_genomes(csv_path):
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                genome = tuple(int(x) for x in row["genome"].strip("()[] ").split(","))
                done.add(genome)
            except Exception:
                pass
    return done

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"BRUTE FORCE — GPU 2  (genomes {START_IDX}–{END_IDX - 1})")
    print("=" * 70)

    # Load dataset
    print("\nLoading SST-5 validation set...")
    sst5 = load_dataset("SetFit/sst5")
    val_data = sst5["validation"]

    # Balanced 50 per class = 250 total (same as evolution experiments)
    balanced = []
    for label_id in range(5):
        samples = [r for r in val_data if r["label"] == label_id][:50]
        balanced.extend(samples)

    ground_truth = [r["label"] for r in balanced]
    print(f"Validation set: {len(balanced)} samples (50 per class)")

    # Load model
    print("\nLoading Llama-3.2-3B-Instruct...")
    pipe = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct",
        device_map="auto",
        torch_dtype=MODEL_DTYPE,
    )
    pipe.tokenizer.padding_side = "left"
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

    # Resume support
    done_genomes = load_done_genomes(OUTPUT_CSV)
    print(f"Already evaluated: {len(done_genomes)} genomes — resuming.")

    # Open CSV
    file_exists = os.path.exists(OUTPUT_CSV) and len(done_genomes) > 0
    csv_file = open(OUTPUT_CSV, "a" if file_exists else "w", newline="", encoding="utf-8")
    fieldnames = [
        "genome", "intro_idx", "task_idx", "labels_idx", "output_idx",
        "exact_accuracy", "ordinal_score",
        "n_correct", "n_unrecognized",
        "vn_correct", "neg_correct", "neu_correct", "pos_correct", "vp_correct",
        "prompt_preview",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    if not file_exists:
        writer.writeheader()

    genomes_to_run = [
        g for g in ALL_GENOMES[START_IDX:END_IDX]
        if g not in done_genomes
    ]
    total = len(genomes_to_run)
    print(f"Genomes to evaluate this session: {total}")

    start_time = time.time()

    for i, genome in enumerate(genomes_to_run):
        genome_start = time.time()

        all_messages = [build_messages(genome, r["text"]) for r in balanced]

        outputs = pipe(
            all_messages,
            max_new_tokens=16,
            do_sample=False,
            return_full_text=False,
            batch_size=BATCH_SIZE,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )

        predictions = [extract_label(out[0]["generated_text"]) for out in outputs]

        ea  = exact_accuracy(ground_truth, predictions)
        os_ = ordinal_score(ground_truth, predictions)
        n_correct  = sum(1 for a, p in zip(ground_truth, predictions) if a == p)
        n_unrec    = sum(1 for p in predictions if p == -1)
        pcc        = per_class_correct(ground_truth, predictions)

        instruction = assemble_instruction(genome)
        output_fmt  = OUTPUT_ALLELES[genome[3]]
        full_prompt = instruction + (" " + output_fmt if output_fmt else "")

        writer.writerow({
            "genome":          list(genome),
            "intro_idx":       genome[0],
            "task_idx":        genome[1],
            "labels_idx":      genome[2],
            "output_idx":      genome[3],
            "exact_accuracy":  round(ea,  6),
            "ordinal_score":   round(os_, 6),
            "n_correct":       n_correct,
            "n_unrecognized":  n_unrec,
            "vn_correct":      pcc[0],
            "neg_correct":     pcc[1],
            "neu_correct":     pcc[2],
            "pos_correct":     pcc[3],
            "vp_correct":      pcc[4],
            "prompt_preview":  full_prompt[:120],
        })
        csv_file.flush()

        elapsed   = time.time() - start_time
        per_genome = elapsed / (i + 1)
        remaining  = per_genome * (total - i - 1)

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"[{i+1:>4}/{total}] genome={list(genome)}  "
                f"exact={ea:.3f}  ordinal={os_:.3f}  "
                f"ETA {remaining/60:.0f} min"
            )

    csv_file.close()

    # Final summary
    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)
    best_exact   = df.loc[df["exact_accuracy"].idxmax()]
    best_ordinal = df.loc[df["ordinal_score"].idxmax()]

    print("\n" + "=" * 70)
    print("PART 2 COMPLETE")
    print("=" * 70)
    print(f"Genomes evaluated : {len(df)}")
    print(f"Best exact        : {best_exact['exact_accuracy']:.4f}  genome={best_exact['genome']}")
    print(f"Best ordinal      : {best_ordinal['ordinal_score']:.4f}  genome={best_ordinal['genome']}")
    print(f"Results saved to  : {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
