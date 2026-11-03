"""
Evaluate a genome on the FULL SST-5 test set (2210 samples)
===========================================================
After evolution selects a best genome on the 250-sample balanced validation
set, this script confirms it on the held-out full test set. It reuses
brute_force_gpu1.py's evaluation logic VERBATIM (build_messages, extract_label,
float32 pipeline), so the full-test number is on the SAME measurement scale as
the brute-force oracle and the evolution runs.

Two ways to pick the genome:
  1. Explicit:   --genome "3,0,4,7"
  2. From evolution outputs (pick the best across seeds in a mode directory):
                 --from-dir ../../results/evolution/llama/exact --select exact
     (globs evolution_eval_cache_*.csv, picks the genome with the highest
      --select metric across all seeds in that directory)
     or a single cache file:  --from-cache <evolution_eval_cache_*.csv> --select exact

Outputs (into --outdir):
  full_test_{tag}.csv          one row per test sample (text, labels, output, correct, ...)
  full_test_{tag}_summary.csv  headline metrics + per-class breakdown + confusion matrix

Examples:
  # SST-5 (the model used for evolution), best EXACT genome found across seeds:
  python evaluate_full_test.py --from-dir ../../results/evolution/llama/exact \
      --select exact --tag llama_exact_best --outdir ../../results/full_test

  # A specific genome (e.g. the brute-force ceiling) on full test:
  python evaluate_full_test.py --genome "3,0,4,7" --tag bf_ceiling \
      --outdir ../../results/full_test

PROMPT FORMATS (--prompt-format):
  chat   (default) : Llama chat template via build_messages (single user turn).
                     Use for the Llama-3.2-3B headline (same scale as oracle/evolution).
  alpaca           : CROSS-MODEL PROMPT-TEXT TRANSFER. The genome is resolved to its
                     prompt TEXT using the (Llama) alleles, then wrapped in the standard
                     Alpaca '### Instruction / ### Input / ### Response' template — the
                     same template EvoPrompt used for Alpaca-7b. The genome indices are
                     NOT re-mapped into any Alpaca-specific allele table (those differ);
                     we transfer the evolved PROMPT, not the index. Use with a non-chat
                     Alpaca model, e.g. --model chavinlo/alpaca-native --prompt-format alpaca.

Example (Llama-evolved prompt transferred onto Alpaca-7b):
  python evaluate_full_test.py --genome "4,1,7,5" --prompt-format alpaca \
      --model chavinlo/alpaca-native --tag g4175_on_alpaca --outdir ../../results/full_test

FEW-SHOT (--fewshot): deploy-time few-shot for the SECONDARY table that compares
  to EvoPrompt's reported (few-shot) SST-5 numbers. Prepends a fixed 5-shot set
  (exactly 1 demo per class, from the validation split -> no test leakage). The
  SAME demos are used for chat and alpaca so the comparison isolates the prompt.
  This evaluates the SAME evolved genome, just with demos added at inference; the
  zero-shot headline (and its oracle / random-search control) is untouched.
  Use a DISTINCT --tag so the few-shot output does not overwrite the zero-shot run.
  Example (Alpaca, the EvoPrompt head-to-head):
    python evaluate_full_test.py --genome "4,1,7,5" --prompt-format alpaca --fewshot \
        --tag alpaca_g4175_fs --outdir ../../results/full_test
"""

import argparse
import csv
import glob
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import pipeline

# Reuse brute-force eval logic so full-test, brute-force, evolution, and random
# search all score genomes identically.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "brute_force"))
from brute_force_gpu1 import (   # noqa: E402
    MODEL_DTYPE,
    build_messages,
    extract_label,
    assemble_instruction,
    OUTPUT_ALLELES,
    LABEL_NAMES,
)

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_ALPACA_MODEL = "chavinlo/alpaca-native"

# Standard Alpaca instruction template (matches EvoPrompt's Alpaca-7b setup):
# the resolved evolved prompt goes in ### Instruction, the review in ### Input.
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

# Re-usable Alpaca preamble (one per stacked block in few-shot mode).
ALPACA_PREAMBLE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
)

# Fixed 5-shot demo set: exactly 1 example per SST-5 class, drawn from the
# VALIDATION split (NOT test -> no leakage). The SAME demos are used for chat
# (Llama) and alpaca (transfer) so the few-shot comparison isolates the PROMPT,
# not the examples.
FEWSHOT_EXAMPLES = [
    ("so unremittingly awful that labeling it a dog probably constitutes cruelty to canines .", "very negative"),
    ("it offers little beyond the momentary joys of pretty and weightless intellectual entertainment .", "negative"),
    ("a perplexing example of promise unfulfilled , despite many charming moments .", "neutral"),
    ("although laced with humor and a few fanciful touches , the film is a refreshingly serious look at young women .", "positive"),
    ("a very well-made , funny and entertaining picture .", "very positive"),
]

# Yelp-5 few-shot demo set: exactly 1 example per class, drawn from the Yelp
# Review Full TRAIN split (the test sub-sample is from the TEST split -> no
# leakage). Hand-picked to be compact, domain-matched (business/restaurant
# reviews), and clearly graded — the neutral demo shows genuine balance (praise
# AND complaint) and the positive demo shows appreciation-with-reservation, so
# the demos reinforce the evolved prompt's own definitions. Ordered 0..4.
YELP_FEWSHOT_EXAMPLES = [
    ("The worst dental office I ever been. No one can beat it!!! You should avoid it at any time.", "very negative"),
    ("Papa J's is expensive and inconsistent.  The ambiance is nice but it doesn't justify the prices.  There are better restaurants in Carnegie.", "negative"),
    ("Good beer selection. Understaffed for a light Monday night crowd, it wasn't her fault she was the only server. But it took about an hour to get our sandwiches. Mine was one of the best reubens I've ever had.", "neutral"),
    ("I really miss the Northstar Amber but the specialty beers are always good.  Food is excellent and the service staff is friendly.", "positive"),
    ("I love this place! The food is always so fresh and delicious. The staff is always friendly, as well.", "very positive"),
]


# ---------------------------------------------------------------------------
# CROSS-DATASET DOMAIN ADAPTATION
# ---------------------------------------------------------------------------
# The SST-5 alleles (imported from brute_force_gpu1, left BYTE-IDENTICAL) contain
# movie-specific nouns. To TRANSFER an SST-5-evolved prompt onto another ordinal
# sentiment dataset, we re-lexicalise ONLY those nouns — never the review text —
# so the reasoning structure (the evolved component choices) is unchanged and the
# experiment isolates cross-dataset generalisability. Longer phrases first so the
# replacements don't collide ("the film" before bare "film").
YELP_DOMAIN_SUBS = [
    ("film criticism", "customer feedback"),
    ("movie reviews",  "customer reviews"),
    ("film reviews",   "customer reviews"),
    ("the film",       "the business"),
    ("scenes",         "details"),
    ("movie",          "business"),
    ("film",           "business"),
]


def apply_subs(text, subs):
    """Apply ordered (find, replace) domain substitutions to a SCAFFOLD string."""
    for a, b in subs:
        text = text.replace(a, b)
    return text


def resolve_prompt_text(genome, subs=()):
    """Full evolved prompt TEXT for a genome (INTRO+TASK+LABELS + OUTPUT format),
    using the canonical (Llama) alleles. This is what gets transferred cross-model.
    `subs` re-lexicalises domain nouns for cross-dataset transfer (empty = SST-5)."""
    instruction = assemble_instruction(genome)
    out_fmt = OUTPUT_ALLELES[genome[3]]
    txt = instruction + (" " + out_fmt if out_fmt else "")
    return apply_subs(txt, subs) if subs else txt


def build_chat(genome, text, subs=()):
    """Domain-aware chat builder. With empty subs it delegates to the canonical
    build_messages (SST-5 path stays byte-identical). With subs it re-lexicalises
    ONLY the instruction + output scaffold, leaving the review `text` untouched."""
    if not subs:
        return build_messages(genome, text)
    instruction = apply_subs(assemble_instruction(genome), subs)
    output_fmt = apply_subs(OUTPUT_ALLELES[genome[3]], subs)
    content = f"{instruction}\n\nReview: {text}"
    if output_fmt:
        content += f"\n{output_fmt}"
    return [{"role": "user", "content": content}]


def build_chat_fewshot(genome, text, demos=FEWSHOT_EXAMPLES, subs=()):
    """Multi-turn chat: 5 (user=demo review, assistant=label) pairs, then the
    test review as the final user turn. Demo user turns reuse build_chat so
    their instruction/review formatting is IDENTICAL to the zero-shot path
    (subs=() delegates to build_messages -> SST-5 path byte-identical). For a
    cross-dataset transfer the demos AND the instruction are domain-adapted."""
    msgs = []
    for demo_text, demo_label in demos:
        msgs.append({"role": "user", "content": build_chat(genome, demo_text, subs)[0]["content"]})
        msgs.append({"role": "assistant", "content": demo_label})
    msgs.append(build_chat(genome, text, subs)[0])
    return msgs


def build_alpaca_fewshot(full_prompt, text, demos=FEWSHOT_EXAMPLES):
    """Stacked Alpaca blocks: 5 filled demo blocks + 1 final block whose
    ### Response is left open for the model to complete (matches the repo's
    testAlpaca_fewshot.py layout: preamble repeated per block). `full_prompt`
    is already domain-adapted; `demos` carries the per-dataset demo set."""
    blocks = []
    for demo_text, demo_label in demos:
        blocks.append(
            ALPACA_PREAMBLE
            + f"### Instruction:\n{full_prompt}\n\n"
            + f"### Input:\n{demo_text}\n\n"
            + f"### Response:\n{demo_label}"
        )
    blocks.append(
        ALPACA_PREAMBLE
        + f"### Instruction:\n{full_prompt}\n\n"
        + f"### Input:\n{text}\n\n"
        + "### Response:\n"
    )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# PUBLISHED BASELINE PROMPTS
# ---------------------------------------------------------------------------
# Prompts from prior work, run through this pipeline so the comparison is on one
# measurement scale. EvoPrompt's SST-5 prompt grades on terrible/bad/okay/good/great,
# so it needs a vocabulary mapping before scoring.
BASELINE_PROMPTS = {
    "evoprompt_ga": {
        "text": ("Have your friend evaluate the movie they had just seen and provide a summary "
                 "opinion (e.g. terrible, bad, okay, good, or great) to determine the sentiment "
                 "of the movie review."),
        "vocab": "evoprompt",
        "source": "Guo et al., EvoPrompt (ICLR 2024), GA variant, SST-5",
    },
}

# Alternate output vocabularies, mapped onto the five canonical classes. Ordered
# most specific first: a two-word label must be tested before its one-word suffix,
# and OKAY before OK.
ALT_LABEL_PATTERNS = {
    "evoprompt": [
        ("VERY NEGATIVE", 0), ("VERY POSITIVE", 4),
        ("TERRIBLE", 0), ("GREAT", 4), ("OKAY", 2),
        ("NEGATIVE", 1), ("POSITIVE", 3), ("NEUTRAL", 2),
        ("BAD", 1), ("GOOD", 3), ("OK", 2),
    ],
}


def extract_label_alt(output_text, patterns):
    """extract_label with an alternate vocabulary: same three-pass precedence
    (exact match, then prefix, then substring)."""
    answer = " ".join(output_text.strip().upper().split())
    if not answer:
        return -1
    for probe in (lambda a, t: a == t,
                  lambda a, t: a.startswith(t),
                  lambda a, t: t in a):
        for label_text, label_id in patterns:
            if probe(answer, label_text):
                return label_id
    return -1


def build_inputs(genome, texts, prompt_format, fewshot=False, subs=(), demos=FEWSHOT_EXAMPLES,
                 prompt_text=None):
    """Return per-sample model inputs. chat -> list of message-lists (Llama);
    alpaca -> list of Alpaca-template strings (prompt-text transfer).
    fewshot=True prepends the fixed 5-shot (1-per-class) demo set (deploy-time).
    `subs` re-lexicalises domain nouns for cross-dataset transfer (empty = SST-5);
    `demos` is the per-dataset few-shot demo set (defaults keep SST-5 identical)."""
    if prompt_format == "chat":
        if fewshot:
            return [build_chat_fewshot(genome, t, demos, subs) for t in texts]
        return [build_chat(genome, t, subs) for t in texts]
    # prompt_text replaces the genome's resolved text, leaving the demo set,
    # template, decoding and label extraction unchanged.
    full_prompt = prompt_text if prompt_text is not None else resolve_prompt_text(genome, subs)
    if fewshot:
        return [build_alpaca_fewshot(full_prompt, t, demos) for t in texts]
    return [ALPACA_TEMPLATE.format(instruction=full_prompt, input=t) for t in texts]


# ---------------------------------------------------------------------------
# GENOME SELECTION
# ---------------------------------------------------------------------------

def parse_genome_str(s):
    return tuple(int(x) for x in s.strip().strip("()[]").split(","))


def best_genome_from_caches(paths, select):
    """Across one or more evolution_eval_cache_*.csv files, return the genome
    with the highest `select` metric ('exact' or 'ordinal')."""
    metric_col = "exact_accuracy" if select == "exact" else "ordinal_score"
    best_val, best_genome, best_src = -1.0, None, None
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                val = float(row[metric_col])
                if val > best_val:
                    best_val = val
                    best_genome = parse_genome_str(row["genome"])
                    best_src = path
    if best_genome is None:
        raise SystemExit(f"No genomes found in: {paths}")
    print(f"Best {select} genome across {len(paths)} cache file(s): "
          f"{list(best_genome)}  ({metric_col}={best_val:.4f})  from {os.path.basename(best_src)}")
    return best_genome


# ---------------------------------------------------------------------------
# DATA / MODEL
# ---------------------------------------------------------------------------

def load_sst5_test(**_):
    """SST-5 full test (2210). Labels already 0..4 = very negative..very positive."""
    test = load_dataset("SetFit/sst5")["test"]
    return [r["text"] for r in test], [r["label"] for r in test]


def load_yelp5_test(n_per_class=500, sample_seed=0, **_):
    """Yelp Review Full test (5-star). HF labels 0..4 map DIRECTLY to
    very negative..very positive (1 star -> 0, 5 stars -> 4). The test split is
    class-balanced at 10k/class; we draw a fixed balanced sub-sample so the run
    is affordable and the per-class / ordinal analysis mirrors SST-5."""
    import random
    ds = load_dataset("yelp_review_full")["test"]
    buckets = {c: [] for c in range(5)}
    for r in ds:
        buckets[r["label"]].append(r["text"])
    rng = random.Random(sample_seed)
    texts, labels = [], []
    for c in range(5):
        picks = rng.sample(buckets[c], min(n_per_class, len(buckets[c])))
        texts.extend(picks)
        labels.extend([c] * len(picks))
    return texts, labels


# Dataset registry. SST-5 is the default (empty subs -> existing behaviour and
# every existing command stays byte-identical). New datasets carry their own
# loader + domain-adaptation substitutions for cross-dataset prompt transfer.
DATASETS = {
    "sst5":  {"loader": load_sst5_test,  "subs": [],               "fewshot": FEWSHOT_EXAMPLES,      "label": "SST-5"},
    "yelp5": {"loader": load_yelp5_test, "subs": YELP_DOMAIN_SUBS, "fewshot": YELP_FEWSHOT_EXAMPLES, "label": "Yelp-5"},
}


def setup_pipeline(model_name):
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype=MODEL_DTYPE,
    )
    pipe.tokenizer.padding_side = "left"
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    # Some tokenizers (e.g. chavinlo/alpaca-native) report a tiny model_max_length
    # (512), which fires a spurious "indexing errors" warning and risks truncating
    # the longer few-shot prompts. Pin it to the model's REAL context window so the
    # full (zero- or few-shot) prompt is never silently cut. LLaMA uses RoPE, so the
    # real limit is max_position_embeddings (alpaca-native=2048; Llama-3.2-3B=131072).
    real_ctx = getattr(pipe.model.config, "max_position_embeddings", None)
    if isinstance(real_ctx, int) and real_ctx > 0:
        pipe.tokenizer.model_max_length = real_ctx
        print(f"  tokenizer.model_max_length pinned to model context = {real_ctx}")
    return pipe


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate(all_inputs, pipe, batch_size, alt_patterns=None, max_new_tokens=16):
    preds, raws = [], []
    start = time.time()
    # Process in chunks so we can print progress on the large test set.
    chunk = 200
    for i in range(0, len(all_inputs), chunk):
        batch = all_inputs[i:i + chunk]
        outputs = pipe(
            batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
            batch_size=batch_size,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        for out in outputs:
            raw = out[0]["generated_text"]
            raws.append(raw)
            preds.append(extract_label_alt(raw, alt_patterns) if alt_patterns
                         else extract_label(raw))
        done = min(i + chunk, len(all_inputs))
        elapsed = time.time() - start
        eta = (elapsed / done) * (len(all_inputs) - done)
        print(f"  [{done}/{len(all_inputs)}]  ETA {eta/60:.0f} min", flush=True)
    return preds, raws


def main():
    parser = argparse.ArgumentParser(description="Evaluate a genome on full SST-5 test (2210).")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--genome", help='Explicit genome, e.g. "3,0,4,7"')
    g.add_argument("--baseline", choices=sorted(BASELINE_PROMPTS),
                   help="Evaluate a published baseline prompt verbatim instead of a genome, "
                        "through the identical pipeline (--prompt-format alpaca only).")
    parser.add_argument("--max-new-tokens", type=int, default=16,
                        help="Generation budget per item (default 16, the setting all "
                             "reported results use). Raise it to test whether an "
                             "unparseable output is a truncated answer rather than a "
                             "format failure.")
    parser.add_argument("--no-vocab-map", action="store_true",
                        help="Diagnostic: score a --baseline run with our canonical label "
                             "vocabulary only, so its own wording counts as unparseable. "
                             "Quantifies how much of a baseline's score is extraction artefact.")
    g.add_argument("--from-cache", help="Single evolution_eval_cache_*.csv to pick the best genome from")
    g.add_argument("--from-dir", help="Mode dir (globs evolution_eval_cache_*.csv across seeds)")
    parser.add_argument("--select", choices=["exact", "ordinal"], default="exact",
                        help="Metric used to pick the best genome (with --from-cache/--from-dir).")
    parser.add_argument("--model", default=None,
                        help="Model name. Defaults to Llama-3.2-3B-Instruct for chat, "
                             "chavinlo/alpaca-native for --prompt-format alpaca.")
    parser.add_argument("--prompt-format", choices=["chat", "alpaca"], default="chat",
                        help="chat = Llama chat template (headline). "
                             "alpaca = prompt-text transfer into the Alpaca template (cross-model).")
    parser.add_argument("--fewshot", action="store_true",
                        help="Deploy-time few-shot: prepend the fixed 5-shot (1-per-class) demo set. "
                             "Use for the secondary few-shot table vs EvoPrompt's few-shot numbers. "
                             "Use a DISTINCT --tag (e.g. *_fs) so it doesn't overwrite the zero-shot run.")
    parser.add_argument("--dataset", choices=list(DATASETS), default="sst5",
                        help="sst5 (default, alleles byte-identical) or yelp5 (CROSS-DATASET TRANSFER: "
                             "the SST-5-evolved prompt is re-lexicalised for the Yelp domain and the "
                             "same genome is scored on Yelp-5 test). Headline/oracle SST-5 path unchanged.")
    parser.add_argument("--n-per-class", type=int, default=500,
                        help="Balanced test sub-sample per class (yelp5 only; sst5 uses its full 2210 test).")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Seed for the balanced yelp5 sub-sample (fixed -> reproducible).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tag", default="run", help="Label used in the output filenames.")
    parser.add_argument("--outdir", default=".", help="Directory for output CSVs.")
    args = parser.parse_args()

    subs = DATASETS[args.dataset]["subs"]
    demos = DATASETS[args.dataset]["fewshot"]
    ds_label = DATASETS[args.dataset]["label"]
    if args.fewshot and not demos:
        parser.error(f"--fewshot requested but no demo set is registered for {args.dataset}.")

    # Pick a sensible default model for the chosen prompt format.
    if args.model is None:
        args.model = DEFAULT_ALPACA_MODEL if args.prompt_format == "alpaca" else DEFAULT_MODEL
    # Guard against an obvious format/model mismatch.
    if args.prompt_format == "alpaca" and "Instruct" in args.model:
        print(f"WARNING: --prompt-format alpaca with a chat/instruct model ({args.model}). "
              f"Did you mean a non-chat Alpaca model?")
    if args.prompt_format == "chat" and "alpaca" in args.model.lower():
        print(f"WARNING: --prompt-format chat with an Alpaca model ({args.model}). "
              f"Alpaca is not a chat model; did you mean --prompt-format alpaca?")

    # Resolve genome
    external_prompt, alt_patterns = None, None
    if args.baseline:
        if args.prompt_format != "alpaca":
            raise SystemExit("--baseline requires --prompt-format alpaca")
        spec = BASELINE_PROMPTS[args.baseline]
        external_prompt = spec["text"]
        alt_patterns = None if args.no_vocab_map else ALT_LABEL_PATTERNS.get(spec.get("vocab"))
        genome = (0, 0, 0, 0)  # unused; the baseline prompt text overrides it
        print(f"Baseline: {args.baseline}  ({spec['source']})")
        print(f"  prompt : {external_prompt}")
        print(f"  vocab  : {spec.get('vocab', 'canonical')}")
    elif args.genome:
        genome = parse_genome_str(args.genome)
        print(f"Genome (explicit): {list(genome)}")
    elif args.from_cache:
        genome = best_genome_from_caches([args.from_cache], args.select)
    else:
        paths = sorted(glob.glob(os.path.join(args.from_dir, "evolution_eval_cache_*.csv")))
        if not paths:
            raise SystemExit(f"No evolution_eval_cache_*.csv in {args.from_dir}")
        genome = best_genome_from_caches(paths, args.select)

    os.makedirs(args.outdir, exist_ok=True)
    detail_csv  = os.path.join(args.outdir, f"full_test_{args.tag}.csv")
    summary_csv = os.path.join(args.outdir, f"full_test_{args.tag}_summary.csv")

    full_prompt = external_prompt if external_prompt is not None else resolve_prompt_text(genome, subs)

    print("=" * 70)
    print(f"FULL TEST EVALUATION  tag={args.tag}")
    print("=" * 70)
    shots = 5 if args.fewshot else 0
    print(f"Dataset : {ds_label}" + ("  (CROSS-DATASET TRANSFER — SST-5-evolved prompt, domain-adapted)" if args.dataset != "sst5" else ""))
    print(f"Model   : {args.model}  (dtype={MODEL_DTYPE})")
    print(f"Format  : {args.prompt_format}" + ("  (cross-model prompt-text transfer)" if args.prompt_format == "alpaca" else ""))
    fs_src = "train" if args.dataset == "yelp5" else "val"
    print(f"Shots   : {shots}" + (f"  (deploy-time few-shot, 1 demo/class from {fs_src} split)" if args.fewshot else "  (zero-shot)"))
    print(f"Genome  : {list(genome)}")
    print(f"Prompt  : {full_prompt[:160]}")

    print(f"\nLoading {ds_label} test set...")
    texts, labels = DATASETS[args.dataset]["loader"](
        n_per_class=args.n_per_class, sample_seed=args.sample_seed)
    print(f"Test samples: {len(texts)}")

    print(f"\nLoading {args.model} (float32)...")
    pipe = setup_pipeline(args.model)

    print("\nEvaluating...")
    all_inputs = build_inputs(genome, texts, args.prompt_format, fewshot=args.fewshot, subs=subs, demos=demos,
                              prompt_text=external_prompt)
    preds, raws = evaluate(all_inputs, pipe, args.batch_size, alt_patterns=alt_patterns,
                           max_new_tokens=args.max_new_tokens)

    # --- Metrics ---
    n = len(labels)
    n_correct = sum(1 for a, p in zip(labels, preds) if a == p)
    n_unrec   = sum(1 for p in preds if p == -1)
    exact_acc = n_correct / n
    ordinal   = sum(0.0 if p == -1 else 1.0 - abs(a - p) / 4.0
                    for a, p in zip(labels, preds)) / n

    # per-class support / correct / recall
    per_class = []
    for cls in range(5):
        support = sum(1 for a in labels if a == cls)
        correct = sum(1 for a, p in zip(labels, preds) if a == cls and p == cls)
        recall  = correct / support if support else 0.0
        per_class.append((cls, LABEL_NAMES[cls], support, correct, recall))

    # 5x5 confusion (rows = actual, cols = predicted; -1 -> 'unrec' bucket)
    confusion = [[0] * 6 for _ in range(5)]   # col 5 = unrecognized
    for a, p in zip(labels, preds):
        confusion[a][p if p != -1 else 5] += 1

    # --- Write per-sample detail ---
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["text", "true_label_id", "true_label",
                    "predicted_label_id", "predicted_label",
                    "model_output", "correct", "ordinal_score", "distance"])
        for t, a, p, raw in zip(texts, labels, preds, raws):
            pred_name = LABEL_NAMES[p] if p != -1 else "unrecognized"
            dist = abs(a - p) if p != -1 else 4
            osc  = 0.0 if p == -1 else 1.0 - abs(a - p) / 4.0
            w.writerow([t, a, LABEL_NAMES[a], p, pred_name, raw,
                        int(a == p), round(osc, 4), dist])

    # --- Write summary ---
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["tag", "dataset", "model", "prompt_format", "shots", "genome", "prompt",
                    "n_samples", "exact_accuracy", "ordinal_score",
                    "n_correct", "n_unrecognized"])
        # A baseline run has no genome.
        genome_field = f"baseline:{args.baseline}" if args.baseline else list(genome)
        w.writerow([args.tag, args.dataset, args.model, args.prompt_format, shots, genome_field, full_prompt,
                    n, round(exact_acc, 6), round(ordinal, 6), n_correct, n_unrec])
        w.writerow([])
        w.writerow(["per_class:", "label_id", "label", "support", "correct", "recall"])
        for cls, name, support, correct, recall in per_class:
            w.writerow(["", cls, name, support, correct, round(recall, 4)])
        w.writerow([])
        w.writerow(["confusion (row=actual, col=predicted)",
                    "very negative", "negative", "neutral", "positive", "very positive", "unrecognized"])
        for cls in range(5):
            w.writerow([LABEL_NAMES[cls]] + confusion[cls])

    print("\n" + "=" * 70)
    print(f"RESULT  tag={args.tag}")
    print("=" * 70)
    print(f"Exact accuracy : {exact_acc:.4f}  ({n_correct}/{n})")
    print(f"Ordinal score  : {ordinal:.4f}")
    print(f"Unrecognized   : {n_unrec}")
    print("Per-class recall:")
    for cls, name, support, correct, recall in per_class:
        print(f"  {name:<14} {correct:>4}/{support:<4}  recall={recall:.3f}")
    print(f"\nSaved: {detail_csv}")
    print(f"Saved: {summary_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
