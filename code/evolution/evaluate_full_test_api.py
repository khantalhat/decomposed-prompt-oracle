"""
Evaluate a genome on an OpenAI chat model (API sibling of evaluate_full_test.py)
================================================================================
This is the CLOSED-API arm of the cross-model comparison. It reuses
evaluate_full_test.py's prompt builders VERBATIM (build_chat / build_chat_fewshot,
resolve_prompt_text, the dataset registry, extract_label, LABEL_NAMES) so the
prompt the API model sees is logically identical to the local chat path -- only
the inference backend differs (OpenAI chat completions instead of a local HF
pipeline). It writes the SAME summary/detail CSV schema as evaluate_full_test.py,
so the offline analysis ingests it with no special-casing.

Ship prompt for the paper's cross-model table: genome [3,1,4,5] (validation-
ordinal-selected). The band (Zhang et al., NAACL 2024) reports ChatGPT zero-shot
at 48.0 (SST-5) / 52.4 (Yelp-5) with THEIR prompt; this measures the SAME model
family with OUR prompt -> a controlled, same-model comparison.

DETERMINISM CAVEAT (disclose in-paper): the OpenAI API is not guaranteed
deterministic even at temperature=0, and the served snapshot can drift over time.
We pin an explicit dated snapshot (default gpt-3.5-turbo-0125) and set
temperature=0 + a fixed seed to maximise reproducibility, but this arm is
inherently less reproducible than the open-weight runs.

Requires:  pip install openai   and   export OPENAI_API_KEY=sk-...
Run from code/evolution/ (it imports evaluate_full_test, which lives here).

Examples (the 4 runs that fill the API cells; all default --genome "3,1,4,5"):
  python evaluate_full_test_api.py --dataset sst5            --tag gpt35_g3145_sst5    --outdir ../../results/full_test
  python evaluate_full_test_api.py --dataset sst5  --fewshot --tag gpt35_g3145_sst5_fs --outdir ../../results/full_test
  python evaluate_full_test_api.py --dataset yelp5           --tag gpt35_g3145_yelp    --outdir ../../results/full_test
  python evaluate_full_test_api.py --dataset yelp5 --fewshot --tag gpt35_g3145_yelp_fs --outdir ../../results/full_test

Smoke test first (cheap, confirms wiring/format before the full spend):
  python evaluate_full_test_api.py --dataset sst5 --limit 25 --tag smoke --outdir /tmp
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse the EXACT builders/parsers/loaders from the local evaluator so the API
# model sees the same prompt and is scored on the same scale. (This pulls in the
# same environment evaluate_full_test.py uses: datasets, and transitively torch/
# transformers -- run it where that import already works.)
from evaluate_full_test import (   # noqa: E402
    build_chat,
    build_chat_fewshot,
    resolve_prompt_text,
    parse_genome_str,
    DATASETS,
    extract_label,
    LABEL_NAMES,
)

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai SDK not installed.  Run: pip install openai")


def build_messages_for(genome, text, fewshot, subs, demos):
    """Logical prompt as a list of chat-completions messages (role/content).
    Identical structure to the local chat path: zero-shot -> single user turn;
    few-shot -> 5 (user=demo, assistant=label) pairs then the test review."""
    if fewshot:
        return build_chat_fewshot(genome, text, demos, subs)
    return build_chat(genome, text, subs)


def call_one(client, model, messages, max_tokens, seed, max_retries=6):
    """One chat completion with greedy decoding + retry/backoff. Retries on BOTH
    transient exceptions AND empty/whitespace responses -- an empty 200 is almost
    always a dropped/rate-limited call, not a genuine prediction, and must not be
    silently scored as wrong. Returns "" only after exhausting retries ->
    extract_label maps it to -1 (unrecognized), so a truly dead call degrades
    gracefully instead of aborting the whole run."""
    delay = 2.0
    for attempt in range(max_retries):
        err = None
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                seed=seed,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
            # empty-but-successful response: treat as transient and retry
        except Exception as e:  # rate limit / transient / network
            err = e
        if attempt == max_retries - 1:
            reason = f"error: {err}" if err is not None else "empty response"
            print(f"  ! giving up after {max_retries} attempts ({reason})", flush=True)
            return ""
        time.sleep(delay)
        delay = min(delay * 2, 60)
    return ""


def main():
    parser = argparse.ArgumentParser(description="Evaluate a genome on an OpenAI chat model (API arm).")
    parser.add_argument("--genome", default="3,1,4,5", help='Genome, e.g. "3,1,4,5" (ship prompt).')
    parser.add_argument("--model", default="gpt-3.5-turbo-0125",
                        help="Pinned OpenAI chat snapshot (default matches the band's 'ChatGPT').")
    parser.add_argument("--dataset", choices=list(DATASETS), default="sst5")
    parser.add_argument("--fewshot", action="store_true",
                        help="Prepend the fixed 5-shot (1/class) demo set for this dataset.")
    parser.add_argument("--n-per-class", type=int, default=500, help="Yelp test sub-sample size/class.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Yelp sub-sample seed.")
    parser.add_argument("--max-workers", type=int, default=8, help="Concurrent API requests.")
    parser.add_argument("--max-tokens", type=int, default=16, help="Mirror of local max_new_tokens.")
    parser.add_argument("--seed", type=int, default=0, help="OpenAI sampling seed (best-effort determinism).")
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: only first N samples (0 = all).")
    parser.add_argument("--tag", default="api_run", help="Label used in the output filenames.")
    parser.add_argument("--outdir", default=".", help="Directory for output CSVs.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set.  export OPENAI_API_KEY=sk-...")

    genome   = parse_genome_str(args.genome)
    subs     = DATASETS[args.dataset]["subs"]
    demos    = DATASETS[args.dataset]["fewshot"]
    ds_label = DATASETS[args.dataset]["label"]
    if args.fewshot and not demos:
        parser.error(f"--fewshot requested but no demo set is registered for {args.dataset}.")
    shots = 5 if args.fewshot else 0
    full_prompt = resolve_prompt_text(genome, subs)

    os.makedirs(args.outdir, exist_ok=True)
    detail_csv  = os.path.join(args.outdir, f"full_test_{args.tag}.csv")
    summary_csv = os.path.join(args.outdir, f"full_test_{args.tag}_summary.csv")

    print("=" * 70)
    print(f"API EVALUATION  tag={args.tag}")
    print("=" * 70)
    print(f"Dataset : {ds_label}" + ("  (CROSS-DATASET TRANSFER -- SST-5-evolved prompt, domain-adapted)" if args.dataset != "sst5" else ""))
    print(f"Model   : {args.model}  (OpenAI API, temperature=0, seed={args.seed})")
    print(f"Shots   : {shots}" + ("  (deploy-time few-shot, 1 demo/class)" if args.fewshot else "  (zero-shot)"))
    print(f"Genome  : {list(genome)}")
    print(f"Prompt  : {full_prompt[:160]}")

    print(f"\nLoading {ds_label} test set...")
    texts, labels = DATASETS[args.dataset]["loader"](
        n_per_class=args.n_per_class, sample_seed=args.sample_seed)
    if args.limit and args.limit < len(texts):
        texts, labels = texts[:args.limit], labels[:args.limit]
        print(f"  LIMIT active: first {args.limit} samples (smoke test)")
    n = len(texts)
    print(f"Test samples: {n}")

    messages_list = [build_messages_for(genome, t, args.fewshot, subs, demos) for t in texts]

    client = OpenAI()
    raws = [None] * n
    print(f"\nQuerying {args.model} ({args.max_workers} workers)...")
    start = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        future_to_i = {
            ex.submit(call_one, client, args.model, messages_list[i], args.max_tokens, args.seed): i
            for i in range(n)
        }
        for fut in as_completed(future_to_i):
            i = future_to_i[fut]
            raws[i] = fut.result()
            done += 1
            if done % 100 == 0 or done == n:
                elapsed = time.time() - start
                eta = (elapsed / done) * (n - done)
                print(f"  [{done}/{n}]  ETA {eta/60:.0f} min", flush=True)

    preds = [extract_label(r) for r in raws]

    # --- Metrics (identical to evaluate_full_test.py) ---
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

    # --- Write summary (same schema; prompt_format='chat', model=the API snapshot) ---
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["tag", "dataset", "model", "prompt_format", "shots", "genome", "prompt",
                    "n_samples", "exact_accuracy", "ordinal_score",
                    "n_correct", "n_unrecognized"])
        w.writerow([args.tag, args.dataset, args.model, "chat", shots, list(genome), full_prompt,
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
