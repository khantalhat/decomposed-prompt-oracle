"""
Offline analysis engine for the decomposed-prompt study
=======================================================
Reads ONLY pre-computed results (zero GPU):
  - results/brute_force/brute_force_combined.csv   (the 4096-genome ORACLE, 250-sample
    balanced validation; full 8^4 factorial, one observation per cell -> exact ANOVA)
  - results/full_test/*_summary.csv                (the held-out test evaluations)

Produces tidy CSV artifacts in results/analysis/ for the figure scripts to draw from,
and prints a findings report. The blocks are:

  PILLAR 1  INSTRUMENT     : the enumerated component space (described, not re-derived)
  PILLAR 2  COMPONENT SCI. : main effects, gene importance (ANOVA eta^2), interactions,
                             per-class component effects, ablation
  PILLAR 3  CONSEQUENCE    : search vs random (the plateau), the cheap additive rule
  RESULTS / ERROR ANALYSIS : test tables, cross-model, few-shot lift, ranking reversal,
                             per-class recall, format-transfer fragility
"""

import ast
import csv
import glob
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
ORACLE = os.path.join(ROOT, "results", "brute_force", "brute_force_combined.csv")
TESTDIR = os.path.join(ROOT, "results", "full_test")
OUT = os.path.join(ROOT, "results", "analysis")
EVODIR = os.path.join(ROOT, "results", "evolution", "llama")
os.makedirs(OUT, exist_ok=True)

GENE_NAMES = ["INTRO", "TASK", "LABELS", "OUTPUT"]
CLASS_NAMES = ["very negative", "negative", "neutral", "positive", "very positive"]
VAL_SUPPORT = 50  # 250 balanced / 5 classes

# Short human labels for charts (full text lives in the oracle source).
SHORT = {
    "INTRO":  ["expert role", "film specialist", "experienced reviewer", "understatement note",
               "nuance note", "ordinal framing", "read-in-full", "(empty)"],
    "TASK":   ["classify (minimal)", "direction+intensity", "dominant sentiment", "mild vs strong",
               "weigh evidence", "anti-anchor", "final conclusion", "(empty)"],
    "LABELS": ["bare list", "pos/neutral bound", "full definitions", "neg-end bound",
               "pos-end bound", "neutral clarify", "ordinal scale", "both-boundary"],
    "OUTPUT": ["return label", "no explanation", "Answer:", "exactly-one-label",
               "no chain-of-thought", "Sentiment:", "penalty framing", "(empty)"],
}
NAIVE = (7, 0, 0, 0)
DEPLOYED = {(7, 2, 0, 5): "g7205", (4, 1, 7, 5): "g4175", (4, 4, 4, 3): "g4443", (3, 1, 4, 5): "g3145"}


def w(name, header, rows):
    p = os.path.join(OUT, name)
    with open(p, "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(header)
        cw.writerows(rows)
    return p


# ---------------------------------------------------------------------------
# LOAD ORACLE
# ---------------------------------------------------------------------------
def load_oracle():
    rows = []
    with open(ORACLE, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            g = (int(d["intro_idx"]), int(d["task_idx"]), int(d["labels_idx"]), int(d["output_idx"]))
            rows.append({
                "g": g,
                "exact": float(d["exact_accuracy"]),
                "ordinal": float(d["ordinal_score"]),
                "unrec": int(d["n_unrecognized"]),
                "cls": [int(d["vn_correct"]), int(d["neg_correct"]), int(d["neu_correct"]),
                        int(d["pos_correct"]), int(d["vp_correct"])],
            })
    assert len(rows) == 4096, f"expected 4096 oracle rows, got {len(rows)}"
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# PILLAR 2a: PER-ALLELE MAIN EFFECTS  (+ per-class recall)
# ---------------------------------------------------------------------------
def main_effects(oracle):
    rows = []
    by_gene_allele_exact = {}  # (gene_i, allele) -> mean exact, for importance + rule
    for gi, gname in enumerate(GENE_NAMES):
        for a in range(8):
            sub = [r for r in oracle if r["g"][gi] == a]
            me = mean([r["exact"] for r in sub])
            mo = mean([r["ordinal"] for r in sub])
            recalls = [mean([r["cls"][c] / VAL_SUPPORT for r in sub]) for c in range(5)]
            by_gene_allele_exact[(gi, a)] = (me, mo)
            rows.append([gname, a, SHORT[gname][a], len(sub),
                         round(me, 4), round(mo, 4)] + [round(x, 4) for x in recalls])
    w("allele_main_effects.csv",
      ["gene", "allele", "label", "n", "mean_exact", "mean_ordinal"] +
      [f"recall_{c.replace(' ', '_')}" for c in CLASS_NAMES],
      rows)
    return by_gene_allele_exact


# ---------------------------------------------------------------------------
# PILLAR 2b: GENE IMPORTANCE via balanced-factorial ANOVA (eta^2)
# ---------------------------------------------------------------------------
def gene_importance(oracle, metric="exact"):
    y = [r[metric] for r in oracle]
    grand = mean(y)
    ss_total = sum((v - grand) ** 2 for v in y)

    # main-effect SS per gene
    ss_main = {}
    allele_means = {}
    for gi in range(4):
        means = {}
        for a in range(8):
            sub = [r[metric] for r in oracle if r["g"][gi] == a]
            means[a] = mean(sub)
        allele_means[gi] = means
        ss_main[gi] = 512 * sum((means[a] - grand) ** 2 for a in range(8))

    # 2-way interaction SS per pair
    ss_int = {}
    for gi in range(4):
        for gj in range(gi + 1, 4):
            cell = defaultdict(list)
            for r in oracle:
                cell[(r["g"][gi], r["g"][gj])].append(r[metric])
            ss_cells = 64 * sum((mean(v) - grand) ** 2 for v in cell.values())
            ss_int[(gi, gj)] = ss_cells - ss_main[gi] - ss_main[gj]

    rows_imp = []
    for gi in range(4):
        amn = allele_means[gi]
        rng = max(amn.values()) - min(amn.values())
        rows_imp.append([GENE_NAMES[gi], round(ss_main[gi] / ss_total, 4),
                         round(rng, 4),
                         SHORT[GENE_NAMES[gi]][max(amn, key=amn.get)],
                         SHORT[GENE_NAMES[gi]][min(amn, key=amn.get)]])
    w(f"gene_importance_{metric}.csv",
      ["gene", "eta2", "allele_range", "best_allele", "worst_allele"], rows_imp)

    rows_int = [[GENE_NAMES[gi], GENE_NAMES[gj], round(ss_int[(gi, gj)] / ss_total, 4)]
                for gi, gj in ss_int]
    rows_int.sort(key=lambda r: -r[2])
    w(f"gene_interactions_{metric}.csv", ["gene_a", "gene_b", "eta2"], rows_int)

    explained_main = sum(ss_main.values()) / ss_total
    explained_int = sum(ss_int.values()) / ss_total
    return rows_imp, rows_int, explained_main, explained_int, ss_int


# ---------------------------------------------------------------------------
# PILLAR 2b': interaction heatmap for the strongest pair
# ---------------------------------------------------------------------------
def interaction_heatmap(oracle, gi, gj, metric="exact"):
    cell = defaultdict(list)
    for r in oracle:
        cell[(r["g"][gi], r["g"][gj])].append(r[metric])
    rows = []
    for a in range(8):
        for b in range(8):
            rows.append([SHORT[GENE_NAMES[gi]][a], SHORT[GENE_NAMES[gj]][b],
                         a, b, round(mean(cell[(a, b)]), 4)])
    w(f"interaction_heatmap_{GENE_NAMES[gi]}_{GENE_NAMES[gj]}_{metric}.csv",
      ["a_label", "b_label", "a_idx", "b_idx", metric], rows)


# ---------------------------------------------------------------------------
# PILLAR 2c: ABLATION  (add-one-in from naive; leave-one-out from a strong genome)
# ---------------------------------------------------------------------------
def score_of(oracle, g):
    for r in oracle:
        if r["g"] == tuple(g):
            return r["exact"], r["ordinal"]
    return None


def rank_of(oracle_sorted, g):
    for i, r in enumerate(oracle_sorted, 1):
        if r["g"] == tuple(g):
            return i
    return None


def ablation(oracle, by_ord_desc):
    NULL = {0: 7, 1: 7, 3: 7}  # INTRO/TASK/OUTPUT have an empty allele; LABELS does not
    rows = []

    # add-one-in: start from NAIVE, switch ONE gene to its best-ordinal allele
    base_e, base_o = score_of(oracle, NAIVE)
    rows.append(["add-in baseline", "naive [7,0,0,0]", round(base_e, 4), round(base_o, 4),
                 0.0, 0.0, rank_of(by_ord_desc, NAIVE)])
    # best allele per gene by ordinal main effect
    best_allele = {}
    for gi in range(4):
        best_allele[gi] = max(range(8), key=lambda a: mean([r["ordinal"] for r in oracle if r["g"][gi] == a]))
    for gi in range(4):
        g = list(NAIVE)
        g[gi] = best_allele[gi]
        e, o = score_of(oracle, g)
        rows.append([f"add-in {GENE_NAMES[gi]}", f"{GENE_NAMES[gi]}->{SHORT[GENE_NAMES[gi]][best_allele[gi]]}",
                     round(e, 4), round(o, 4), round(e - base_e, 4), round(o - base_o, 4),
                     rank_of(by_ord_desc, g)])

    # leave-one-out: start from the SHIP genome g3145, ablate each ablatable gene to null
    SHIP = (3, 1, 4, 5)
    se, so = score_of(oracle, SHIP)
    rows.append(["LOO baseline", "ship g3145 [3,1,4,5]", round(se, 4), round(so, 4),
                 0.0, 0.0, rank_of(by_ord_desc, SHIP)])
    for gi, nun in NULL.items():
        g = list(SHIP)
        g[gi] = nun
        e, o = score_of(oracle, g)
        rows.append([f"LOO -{GENE_NAMES[gi]}", f"{GENE_NAMES[gi]}->(empty)",
                     round(e, 4), round(o, 4), round(e - se, 4), round(o - so, 4),
                     rank_of(by_ord_desc, g)])
    w("ablation.csv",
      ["study", "change", "exact", "ordinal", "delta_exact", "delta_ordinal", "oracle_rank_ordinal"], rows)
    return rows


# ---------------------------------------------------------------------------
# PILLAR 2c: LOCAL one-at-a-time sensitivity around the SHIP genome
#   Hold three genes at their ship values and sweep the fourth over all 8
#   alleles. This is the *local* analogue of the global main effects: it shows
#   how much each gene still moves accuracy at the actual deployed prompt (with
#   interactions baked in), and it generalises leave-one-out (the empty allele
#   is just one point on each curve). All values are free oracle look-ups.
# ---------------------------------------------------------------------------
def local_sensitivity(oracle, base=(3, 1, 4, 5)):
    base_e, base_o = score_of(oracle, base)
    rows = []
    for gi, gname in enumerate(GENE_NAMES):
        for a in range(8):
            g = list(base)
            g[gi] = a
            e, o = score_of(oracle, tuple(g))
            rows.append([gname, a, SHORT[gname][a],
                         round(e, 4), round(o, 4),
                         1 if a == base[gi] else 0,
                         round(e - base_e, 4), round(o - base_o, 4)])
    w("local_sensitivity_g3145.csv",
      ["gene", "allele", "label", "exact", "ordinal", "is_incumbent",
       "delta_exact", "delta_ordinal"], rows)
    return rows


# ---------------------------------------------------------------------------
# PILLAR 3: the cheap ADDITIVE RULE vs the oracle optimum
# ---------------------------------------------------------------------------
def additive_rule(oracle, by_e, by_ord_desc, by_exact_desc):
    rows = []
    for metric, idx in [("ordinal", 1), ("exact", 0)]:
        g = tuple(max(range(8), key=lambda a: by_e[(gi, a)][idx]) for gi in range(4))
        e, o = score_of(oracle, g)
        rows.append([f"additive-rule ({metric})", str(list(g)),
                     round(e, 4), round(o, 4),
                     rank_of(by_exact_desc, g), rank_of(by_ord_desc, g)])
    # references
    best_e = by_exact_desc[0]; best_o = by_ord_desc[0]
    rows.append(["oracle-best exact", str(list(best_e["g"])), round(best_e["exact"], 4),
                 round(best_e["ordinal"], 4), 1, rank_of(by_ord_desc, best_e["g"])])
    rows.append(["oracle-best ordinal", str(list(best_o["g"])), round(best_o["exact"], 4),
                 round(best_o["ordinal"], 4), rank_of(by_exact_desc, best_o["g"]), 1])
    for g, name in DEPLOYED.items():
        e, o = score_of(oracle, g)
        rows.append([f"evolved {name}", str(list(g)), round(e, 4), round(o, 4),
                     rank_of(by_exact_desc, g), rank_of(by_ord_desc, g)])
    e, o = score_of(oracle, NAIVE)
    rows.append(["naive", str(list(NAIVE)), round(e, 4), round(o, 4),
                 rank_of(by_exact_desc, NAIVE), rank_of(by_ord_desc, NAIVE)])
    w("additive_rule.csv",
      ["which", "genome", "exact", "ordinal", "rank_exact", "rank_ordinal"], rows)
    return rows


# ---------------------------------------------------------------------------
# PILLAR 3: SEARCH vs RANDOM  (exact expected-max of a K-sample, no replacement)
# ---------------------------------------------------------------------------
def expected_max(values_desc, K, N):
    # E[max] = sum_i v_(i) * C(N-i, K-1)/C(N, K)   (i = sorted position, 1-indexed)
    logden = math.lgamma(N + 1) - math.lgamma(K + 1) - math.lgamma(N - K + 1)
    s = 0.0
    for i in range(1, N - K + 2):
        ni = N - i
        lognum = math.lgamma(ni + 1) - math.lgamma(K) - math.lgamma(ni - K + 2)
        s += values_desc[i - 1] * math.exp(lognum - logden)
    return s


def search_vs_random(oracle):
    N = 4096
    ex_desc = sorted((r["exact"] for r in oracle), reverse=True)
    or_desc = sorted((r["ordinal"] for r in oracle), reverse=True)
    Ks = [1, 2, 5, 10, 20, 40, 80, 120, 200, 300, 400]
    rows = []
    for K in Ks:
        rows.append([K, round(expected_max(ex_desc, K, N), 4), round(expected_max(or_desc, K, N), 4)])
    w("search_vs_random.csv", ["K", "E_max_exact", "E_max_ordinal"], rows)

    # distribution (binned) for a clean histogram
    def hist(vals, lo, hi, nb=40):
        binw = (hi - lo) / nb
        counts = [0] * nb
        for v in vals:
            k = min(nb - 1, int((v - lo) / binw))
            counts[k] += 1
        return [[round(lo + (k + 0.5) * binw, 4), counts[k]] for k in range(nb)]
    w("oracle_hist_exact.csv", ["bin_center", "count"],
      hist([r["exact"] for r in oracle], min(r["exact"] for r in oracle), max(r["exact"] for r in oracle)))
    w("oracle_hist_ordinal.csv", ["bin_center", "count"],
      hist([r["ordinal"] for r in oracle], min(r["ordinal"] for r in oracle), max(r["ordinal"] for r in oracle)))

    # plateau width: how many genomes within epsilon of the max
    maxo = or_desc[0]; maxe = ex_desc[0]
    plateau = []
    for eps in [0.002, 0.005, 0.01, 0.02]:
        plateau.append([eps,
                        sum(1 for v in or_desc if v >= maxo - eps),
                        sum(1 for v in ex_desc if v >= maxe - eps)])
    w("plateau.csv", ["epsilon", "n_within_ordinal", "n_within_exact"], plateau)
    return rows, ex_desc, or_desc


# ---------------------------------------------------------------------------
# RESULTS + ERROR ANALYSIS  (from the test summaries)
# ---------------------------------------------------------------------------
MODEL_SHORT = {
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "chavinlo/alpaca-native": "Alpaca-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "gpt-3.5-turbo-0125": "GPT-3.5-turbo",
}


def parse_summary(fp):
    with open(fp, newline="", encoding="utf-8") as f:
        r = list(csv.reader(f))
    head, data = r[0], r[1]
    rec = dict(zip(head, data))
    per_class = {}
    mode = None
    for row in r[2:]:
        if not row or all(c == "" for c in row):
            mode = None
            continue
        if row[0].startswith("per_class"):
            mode = "pc"; continue
        if row[0].startswith("confusion"):
            mode = "cf"; continue
        if mode == "pc" and len(row) >= 6:
            per_class[row[2]] = float(row[5])  # label -> recall
    return rec, per_class


def results_tables():
    long_rows = []
    pc_rows = []
    for fp in sorted(glob.glob(os.path.join(TESTDIR, "*_summary.csv"))):
        rec, pc = parse_summary(fp)
        ds = rec.get("dataset", "") or "sst5"
        ds = {"sst5": "SST-5", "yelp5": "Yelp-5", "": "SST-5"}.get(ds, ds)
        model = MODEL_SHORT.get(rec.get("model", ""), rec.get("model", ""))
        shots = int(rec.get("shots") or 0)
        genome = rec.get("genome", "")
        n = int(rec.get("n_samples", 0))
        ex = float(rec.get("exact_accuracy", 0))
        od = float(rec.get("ordinal_score", 0))
        un = int(rec.get("n_unrecognized", 0))
        long_rows.append([ds, model, shots, genome, n, round(ex, 4), round(od, 4),
                          un, round(100 * un / n, 2)])
        for cls in CLASS_NAMES:
            if cls in pc:
                pc_rows.append([ds, model, shots, genome, cls, pc[cls]])
    long_rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    w("test_results_long.csv",
      ["dataset", "model", "shots", "genome", "n", "exact", "ordinal", "unrec", "unrec_pct"], long_rows)
    w("test_per_class_recall.csv",
      ["dataset", "model", "shots", "genome", "class", "recall"], pc_rows)
    return long_rows, pc_rows


def cross_model_and_lift(long_rows):
    # cross-model: ship genome g3145 only
    SHIP = "[3, 1, 4, 5]"
    cm = [r for r in long_rows if r[3] == SHIP]
    w("cross_model_g3145.csv",
      ["dataset", "model", "shots", "exact", "ordinal", "unrec_pct"],
      sorted([[r[0], r[1], r[2], r[5], r[6], r[8]] for r in cm]))

    # few-shot lift per (dataset, model, genome)
    idx = {(r[0], r[1], r[3], r[2]): r[5] for r in long_rows}
    lift = []
    for (ds, model, genome, shots), ex in idx.items():
        if shots == 0 and (ds, model, genome, 5) in idx:
            lift.append([ds, model, genome, round(ex, 4), round(idx[(ds, model, genome, 5)], 4),
                         round(idx[(ds, model, genome, 5)] - ex, 4)])
    lift.sort(key=lambda r: (r[0], r[1], r[2]))
    w("fewshot_lift.csv", ["dataset", "model", "genome", "zero_shot", "few_shot", "lift"], lift)

    # ranking reversal: Llama-3.2-3B 0-shot, SST-5 vs Yelp, per genome
    base = [r for r in long_rows if r[1] == "Llama-3.2-3B" and r[2] == 0]
    sst = {r[3]: r[5] for r in base if r[0] == "SST-5"}
    yelp = {r[3]: r[5] for r in base if r[0] == "Yelp-5"}
    rev = [[g, round(sst[g], 4), round(yelp[g], 4)] for g in sst if g in yelp]
    w("ranking_reversal.csv", ["genome", "sst5_exact", "yelp5_exact"], rev)
    return cm, lift, rev


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CONVERGENCE  (evolutionary best-so-far trajectories vs generation)
# ---------------------------------------------------------------------------
def convergence(oracle):
    """Consolidate the per-seed evolutionary convergence logs into one tidy CSV
    (best-so-far vs generation + the running count of DISTINCT genomes scored),
    plus a per-run summary: final best scores, the generation at which the best
    was first reached, and the total number of distinct genomes evaluated. The
    400-call budget includes elite cache-hits, so the distinct count is lower.
    Reads only the released convergence logs; no GPU.

    Each best-so-far is emitted twice. The *_so_far columns are the run's own
    fitness, which is what the search saw. The *_oracle columns are the fixed
    oracle score of the same genome, which is the comparable quantity: it is
    measured once, on the same items, for every genome in the space. The two
    differ slightly because greedy inference is not bit-reproducible across
    runs."""
    metric_key = {"exact": "best_exact_so_far", "ordinal": "best_ordinal_so_far"}
    genome_key = {"exact": "best_genome_exact", "ordinal": "best_genome_ordinal"}
    lookup = {r["g"]: r for r in oracle}
    long_rows, summ_rows = [], []
    for mode in ("exact", "ordinal"):
        pattern = os.path.join(EVODIR, mode, f"evolution_convergence_seed*_{mode}_*.csv")
        for p in sorted(glob.glob(pattern)):
            seed = os.path.basename(p).split("seed")[1].split("_")[0]
            with open(p, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                ge = lookup[tuple(ast.literal_eval(r["best_genome_exact"]))]
                go = lookup[tuple(ast.literal_eval(r["best_genome_ordinal"]))]
                long_rows.append([mode, seed, int(r["gen"]),
                                  float(r["best_exact_so_far"]),
                                  float(r["best_ordinal_so_far"]),
                                  ge["exact"], go["ordinal"],
                                  int(r["n_unique_evals_total"])])
            last = rows[-1]
            final_ex = float(last["best_exact_so_far"])
            final_or = float(last["best_ordinal_so_far"])
            final_ex_or = lookup[tuple(ast.literal_eval(last["best_genome_exact"]))]["exact"]
            final_or_or = lookup[tuple(ast.literal_eval(last["best_genome_ordinal"]))]["ordinal"]
            n_distinct = int(last["n_unique_evals_total"])
            # first generation at which the objective-relevant best was first reached
            target = float(last[metric_key[mode]])
            gen_best = next(int(r["gen"]) for r in rows
                            if float(r[metric_key[mode]]) >= target - 1e-9)
            summ_rows.append([mode, seed, final_ex, final_or,
                              final_ex_or, final_or_or,
                              last[genome_key[mode]].replace(" ", ""),
                              gen_best, n_distinct])
    w("convergence.csv",
      ["mode", "seed", "gen", "best_exact_so_far", "best_ordinal_so_far",
       "best_exact_oracle", "best_ordinal_oracle", "n_unique_evals_total"], long_rows)
    w("convergence_summary.csv",
      ["mode", "seed", "final_exact", "final_ordinal",
       "final_exact_oracle", "final_ordinal_oracle", "returned_genome",
       "gen_best_reached", "n_distinct_evals"], summ_rows)
    return summ_rows


def main():
    oracle = load_oracle()
    by_ord_desc = sorted(oracle, key=lambda r: -r["ordinal"])
    by_exact_desc = sorted(oracle, key=lambda r: -r["exact"])

    by_e = main_effects(oracle)
    imp_e, int_e, exp_main_e, exp_int_e, ss_int_e = gene_importance(oracle, "exact")
    imp_o, int_o, exp_main_o, exp_int_o, ss_int_o = gene_importance(oracle, "ordinal")
    # export heatmaps for the three strongest 2-way interactions
    # (TASK x OUTPUT, INTRO x TASK, LABELS x OUTPUT for the exact objective)
    for pa, pb, _eta in int_e[:3]:
        interaction_heatmap(oracle, GENE_NAMES.index(pa), GENE_NAMES.index(pb), "exact")

    abl = ablation(oracle, by_ord_desc)
    sens = local_sensitivity(oracle)
    rule = additive_rule(oracle, by_e, by_ord_desc, by_exact_desc)
    svr, ex_desc, or_desc = search_vs_random(oracle)
    conv = convergence(oracle)

    long_rows, pc_rows = results_tables()
    cm, lift, rev = cross_model_and_lift(long_rows)

    # ---- findings report ----
    print("\n" + "#" * 72)
    print("# FINDINGS REPORT")
    print("#" * 72)

    print("\n[PILLAR 2] GENE IMPORTANCE (eta^2, balanced-factorial ANOVA)")
    print(f"  {'gene':8} {'eta2_exact':>11} {'eta2_ord':>9}  {'range_ex':>8}  best -> worst allele")
    impo = {r[0]: r for r in imp_o}
    for r in imp_e:
        g = r[0]
        print(f"  {g:8} {r[1]:>11} {impo[g][1]:>9}  {r[2]:>8}  {r[3]} -> {r[4]}")
    print(f"  main effects explain: exact {exp_main_e:.1%}, ordinal {exp_main_o:.1%}  |  "
          f"2-way interactions: exact {exp_int_e:.1%}, ordinal {exp_int_o:.1%}")

    print("\n[PILLAR 2] STRONGEST 2-WAY INTERACTIONS (eta^2, exact)")
    for r in int_e[:3]:
        print(f"  {r[0]:7} x {r[1]:7}  eta2={r[2]}")

    print("\n[PILLAR 2] ABLATION")
    for r in abl:
        print(f"  {r[0]:16} {r[1]:32} exact={r[2]} ord={r[3]} dEx={r[4]:+} dOrd={r[5]:+} rank={r[6]}")

    print("\n[PILLAR 2] LOCAL SENSITIVITY around ship g3145 (sweep one gene, hold three)")
    print(f"  {'gene':8} {'exact range':>11} {'ordinal range':>13}")
    for gname in GENE_NAMES:
        ex = [r[3] for r in sens if r[0] == gname]
        od = [r[4] for r in sens if r[0] == gname]
        print(f"  {gname:8} {max(ex) - min(ex):>11.3f} {max(od) - min(od):>13.3f}")

    print("\n[PILLAR 3] CHEAP ADDITIVE RULE vs ORACLE")
    for r in rule:
        print(f"  {r[0]:24} {r[1]:14} exact={r[2]} ord={r[3]} rank_ex={r[4]} rank_ord={r[5]}")

    print("\n[PILLAR 3] SEARCH vs RANDOM (expected best-of-K, no replacement)")
    print(f"  {'K':>4} {'E_max_exact':>12} {'E_max_ordinal':>14}")
    for K, ee, oo in svr:
        print(f"  {K:>4} {ee:>12} {oo:>14}")
    with open(os.path.join(OUT, "plateau.csv")) as f:
        print("  plateau (n genomes within eps of max):")
        for line in list(csv.reader(f))[1:]:
            print(f"    eps={line[0]}: ordinal {line[1]}, exact {line[2]}")

    print("\n[PILLAR 3] EVOLUTION CONVERGENCE (best-so-far; distinct evals < 400 budget)")
    print(f"  {'mode':6} {'seed':>4} {'final_ex':>8} {'final_ord':>9} {'gen*':>5} {'distinct':>8}")
    for r in conv:
        print(f"  {r[0]:6} {r[1]:>4} {r[2]:>8} {r[3]:>9} {r[4]:>5} {r[5]:>8}")

    print("\n[RESULTS] CROSS-MODEL (ship g3145), exact accuracy")
    print(f"  {'dataset':7} {'model':14} {'shots':>5} {'exact':>7} {'unrec%':>7}")
    for r in sorted(cm, key=lambda x: (x[0], x[2], -x[5])):
        print(f"  {r[0]:7} {r[1]:14} {r[2]:>5} {r[5]:>7} {r[8]:>7}")

    print("\n[ERROR] RANKING REVERSAL (Llama-3.2-3B 0-shot, exact)")
    print(f"  {'genome':14} {'SST-5':>7} {'Yelp-5':>7}")
    for r in rev:
        print(f"  {r[0]:14} {r[1]:>7} {r[2]:>7}")

    print(f"\nArtifacts written to: {OUT}")
    print("#" * 72)


if __name__ == "__main__":
    main()
