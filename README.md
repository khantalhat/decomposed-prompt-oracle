# Decomposed and Enumerable Prompt Optimisation — release

Artefacts for *Decomposed and Enumerable Prompt Optimisation for Five-Class Ordinal
Sentiment Classification*.

The study decomposes a prompt into four functional components (INTRO, TASK, LABELS,
OUTPUT), gives each eight frozen phrasings, and scores all `8^4 = 4,096` resulting prompts
on a fixed 250-item SST-5 probe. That exhaustive lookup table is the **oracle**, and every
claim in the paper is checkable against it.

Total size is under 2 MB. Nothing here needs a GPU to re-analyse.

## What is here

| path | what it is |
|---|---|
| `allele_library.json` | the 32 frozen allele strings, verbatim (Appendix A of the paper) |
| `LICENCE` | MIT for `code/`, CC BY 4.0 for the derived artefacts, original terms for the dataset excerpts |
| `validation_items.csv` | the 250-item probe the oracle is built on |
| `results/brute_force/` | the oracle: all 4,096 genomes with exact accuracy, ordinal score, per-class recalls, unrecognised count |
| `results/evolution/llama/` | per-seed convergence traces and evaluation caches, 20 seeds per objective |
| `results/analysis/` | the derived tables behind the paper's figures and tables |
| `results/full_test/` | held-out test summaries for every reported condition, including the 64-token generation-budget diagnostic, plus per-item outcomes for the two prompts in the paired EvoPrompt comparison |
| `code/` | the scripts that produced and analyse the above |

The layout mirrors the working tree, so the scripts in `code/` run unmodified: they resolve
paths relative to their own location.

## Reproducing the paper's numbers

Everything downstream of the oracle is deterministic and runs on a laptop:

```bash
cd code/analysis
python analyze.py                 # component analysis, transfer, error analysis
python paired_random_vs_ea.py     # equivalence test, guided vs unguided search
python variance_decomposition.py  # main / 2-way / 3-way / 4-way variance split

python mcnemar_vs_evoprompt.py   ../../results/full_test/full_test_alpaca_g3145_fs.csv   ../../results/full_test/full_test_evoprompt_ga_fs.csv
```

The last one is the paired comparison against EvoPrompt's released GA prompt on Alpaca-7B.
Both systems are scored on the same 2,210 SST-5 test items, so they are not independent
samples; McNemar conditions on the items where they disagree, and equivalence is tested at
the same margin used for the search comparison.

Each rewrites `results/analysis/`, so a clean checkout regenerates those CSVs from the
oracle and the convergence traces alone. `make_paper_figures.py` additionally redraws the
figures.

Rebuilding the oracle itself needs a GPU (`code/brute_force/`), roughly 4,096 x 250
generations on Llama-3.2-3B. It was run as two shards on two GPUs:
`brute_force_gpu1.py` covers genomes 0-2047 and `brute_force_gpu2.py` covers 2048-4095.
The two files are identical apart from that range and their output filename, and
`combine_brute_force.py` merges the halves into `brute_force_combined.csv`. To run it in
one pass on a single GPU, set `START_IDX = 0` and `END_IDX = 4096` in either file.
`brute_force_gpu1.py` is also the module the other scripts import their evaluation logic
from, so the whole pipeline scores genomes identically.

## Reading the allele library

Index 7 of INTRO, TASK and OUTPUT is the **empty string**. That is deliberate, not a missing
entry: the empty allele is how a component is switched off, which is what makes the add-one-in
ablation possible from inside the same enumerated space. LABELS has no empty variant, because
every prompt must name the five class names for the output to be scoreable at all; its index 7
is a content-carrying allele.

## The probe

`validation_items.csv` is the first 50 items of each class from the `SetFit/sst5` validation
split, in dataset order. No shuffle and no random seed, so the selection is reproducible from
that rule alone; the file is provided so the probe survives any future change to the dataset's
row order.

The oracle is exact **for this probe**. It gives every prompt's score under this experimental
setup, not its true performance across all of SST-5.

## Determinism

Open-weight models run through Hugging Face `transformers` with float32 weights and greedy
decoding, so a genome re-scored on the same items returns the same value. The GPT-3.5 arm is
queried at temperature 0 against a pinned dated snapshot and is inherently less reproducible.

The `evolution/` traces come from a `(mu+lambda)` evolutionary algorithm with mutation only:
a single gene is reset per offspring, and there is no crossover.

## Data provenance

The review text in `validation_items.csv`, in the five-shot demonstration set inside
`code/evolution/evaluate_full_test.py`, and behind every score in `results/` comes from two
public datasets, redistributed here only to the extent needed to make the probe and the
demonstrations reproducible:

- **SST-5** via the `SetFit/sst5` Hugging Face mirror of the Stanford Sentiment Treebank
  (Socher et al., EMNLP 2013) — the 250 probe items and the SST-5 demonstrations.
- **Yelp-5** via the Yelp Open Dataset, under the Yelp Dataset Terms of Use — the five Yelp
  demonstrations only. All Yelp-5 results here are aggregate scores, with no per-item text.

Neither dataset is relicensed by this release; see `LICENCE` section 3. Model weights are not
redistributed either — the scripts download them from Hugging Face, and each model keeps its
own licence.

## Citation

Add the IEEE Access reference once assigned.
