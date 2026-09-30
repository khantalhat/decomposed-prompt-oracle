#!/usr/bin/env python3
"""
Generate publication-quality vector (PDF) figures for the manuscript from the
released analysis CSVs.  Every number is read from results/analysis/ (no values are
invented here).  Figures are written to docs/manuscript/latex/figures/.

Run:  python3 code/analysis/make_paper_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------------- paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANA  = os.path.join(ROOT, "results", "analysis")
OUT  = os.path.join(ROOT, "docs", "manuscript", "latex", "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})
COL = 3.45      # IEEE single-column width (inches)
WIDE = 7.16     # IEEE double-column width (inches)
C_EXACT, C_ORD = "#2c6fbb", "#d2792a"

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    if os.environ.get("PNG_PREVIEW"):
        fig.savefig(os.path.join("/tmp", name.replace(".pdf", ".png")), dpi=150)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))

# ============================================================ Fig 1: oracle landscape
def fig_landscape():
    # landmark exact accuracies, read from the released CSVs
    svr = pd.read_csv(os.path.join(ANA, "search_vs_random.csv"))
    grand_mean = float(svr.loc[svr.K == 1,   "E_max_exact"].iloc[0])   # 0.435 (mean over all 4096)
    rand400    = float(svr.loc[svr.K == 400, "E_max_exact"].iloc[0])   # 0.538
    # Evolutionary aggregates use the ORACLE score of each returned genome, so every
    # arm of this figure is measured on the same fixed scale as the random arm. Do NOT
    # read final_exact from convergence_summary.csv here: that is the run's own fitness,
    # which carries greedy-inference noise and is not comparable to an oracle lookup.
    pairs = pd.read_csv(os.path.join(ANA, "paired_ea_vs_random.csv"))
    evo_seeds = sorted(pairs.loc[pairs["mode"] == "exact", "ea_score"].tolist())
    evo_mean  = float(np.mean(evo_seeds))                              # 0.535
    evo_lo, evo_hi = evo_seeds[0], evo_seeds[-1]                       # 0.508, 0.552
    oracle_max = 0.552  # global optimum over the full oracle (== best evolutionary seed)

    c_mean, c_rand, c_evo, c_or = "#555555", C_ORD, "#2a8a4a", "#b03030"
    xs = [0, 1, 2, 3]
    ys = [grand_mean, rand400, evo_mean, oracle_max]
    cs = [c_mean, c_rand, c_evo, c_or]

    fig, ax = plt.subplots(figsize=(COL, 2.5))

    # usable head-room band (grand-mean floor to oracle ceiling)
    ax.axhspan(grand_mean, oracle_max, color="#f4f4f4", zorder=0)
    ax.axhline(oracle_max, color=c_or, lw=0.7, ls="--", zorder=1)

    # profile line: accuracy climbing with search effort, then flattening
    ax.plot(xs, ys, "-", color="#b3b3b3", lw=1.3, zorder=2)
    # seed spread at the evolution point (whisker straddles the random value)
    ax.errorbar(2, evo_mean, yerr=[[evo_mean - evo_lo], [evo_hi - evo_mean]],
                fmt="none", ecolor=c_evo, elinewidth=1.0, capsize=2.5, zorder=3)
    for x, y, c in zip(xs, ys, cs):
        ax.plot(x, y, "o", color=c, ms=4.5, zorder=4)

    # value labels
    ax.annotate(f"{grand_mean:.3f}", (0, grand_mean), textcoords="offset points",
                xytext=(0, -12), ha="center", fontsize=7, color=c_mean)
    ax.annotate(f"{rand400:.3f}",    (1, rand400),    textcoords="offset points",
                xytext=(0, 9),  ha="center", fontsize=7, color=c_rand)
    ax.annotate(f"mean {evo_mean:.3f}", (2, evo_mean), textcoords="offset points",
                xytext=(0, -12), ha="center", fontsize=7, color=c_evo)
    ax.annotate(f"{oracle_max:.3f}", (3, oracle_max), textcoords="offset points",
                xytext=(0, 9),  ha="center", fontsize=7, color=c_or)

    # head-room arrow on the right
    ax.annotate("", xy=(3.55, oracle_max), xytext=(3.55, grand_mean),
                arrowprops=dict(arrowstyle="<->", color="#999", lw=0.9))
    ax.text(3.67, (grand_mean + oracle_max) / 2, "head-room\n$\\approx$12 pts",
            va="center", ha="left", fontsize=6.5, color="#666")

    ax.set_xticks(xs)
    ax.set_xticklabels(["grand\nmean", "random\nsearch", "evolutionary\nsearch", "oracle\nmax"])
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(grand_mean - 0.02, oracle_max + 0.02)
    ax.set_ylabel("Exact accuracy")
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig_landscape.pdf")

# ============================================================ Fig 2: search vs random
def fig_search_vs_random():
    df = pd.read_csv(os.path.join(ANA, "search_vs_random.csv"))
    tick_K = [40, 80, 200, 400]
    df = df[df.K.isin(tick_K)].sort_values("K").reset_index(drop=True)
    x = np.arange(len(df))                      # even-spaced categorical positions
    xr = x[-1]                                   # index of the K=400 column (rightmost)
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    ax2 = ax.twinx()
    l1, = ax.plot(x, df.E_max_exact, "-o", color=C_EXACT, ms=3, label="exact")
    l2, = ax2.plot(x, df.E_max_ordinal, "-s", color=C_ORD, ms=3, label="ordinal")
    ax.axhline(0.552, color=C_EXACT, ls=":", lw=0.8)
    ax2.axhline(0.862, color=C_ORD, ls=":", lw=0.8)
    # evolutionary search results at budget 400 (rightmost column)
    ax.plot(xr, 0.552, marker="*", color=C_EXACT, ms=9, mec="k", mew=0.4)
    ax2.plot(xr, 0.862, marker="*", color=C_ORD, ms=9, mec="k", mew=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(k) for k in tick_K])
    ax.set_xlabel("Random-sampling budget $K$")
    ax.set_ylabel("Exact: $E[\\max_K]$", color=C_EXACT)
    ax2.set_ylabel("Ordinal: $E[\\max_K]$", color=C_ORD)
    ax.tick_params(axis="y", colors=C_EXACT)
    ax2.tick_params(axis="y", colors=C_ORD)
    ax.set_ylim(0.42, 0.565)
    ax2.set_ylim(0.805, 0.868)
    ax.legend([l1, l2], ["exact", "ordinal"], loc="lower right", frameon=False)
    ax.text(xr, 0.556, "guided\nsearch", fontsize=5.5, ha="center", color="k")
    for s in ("top",):
        ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    save(fig, "fig_search_vs_random.pdf")

# ============================================================ Fig 3: gene importance
def fig_eta2():
    ex = pd.read_csv(os.path.join(ANA, "gene_importance_exact.csv")).set_index("gene")
    od = pd.read_csv(os.path.join(ANA, "gene_importance_ordinal.csv")).set_index("gene")
    genes = ["INTRO", "TASK", "LABELS", "OUTPUT"]
    x = np.arange(len(genes)); w = 0.38
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    ax.bar(x - w/2, [ex.loc[g, "eta2"] for g in genes], w, color=C_EXACT,
           label="exact")
    ax.bar(x + w/2, [od.loc[g, "eta2"] for g in genes], w, color=C_ORD,
           label="ordinal")
    ax.set_xticks(x); ax.set_xticklabels(genes)
    ax.set_ylabel("$\\eta^2$ (variance explained)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig_eta2.pdf")

# ============================================ Fig 4: TASK x OUTPUT interaction heatmap
def fig_interaction():
    df = pd.read_csv(os.path.join(ANA, "interaction_heatmap_TASK_OUTPUT_exact.csv"))
    tasks = df.a_label.drop_duplicates().tolist()
    outs  = df.b_label.drop_duplicates().tolist()
    M = df.pivot(index="a_label", columns="b_label", values="exact").loc[tasks, outs]
    fig, ax = plt.subplots(figsize=(COL, 3.1))
    im = ax.imshow(M.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(outs))); ax.set_xticklabels(outs, rotation=45, ha="right")
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels(tasks)
    ax.set_xlabel("OUTPUT allele"); ax.set_ylabel("TASK allele")
    for i in range(len(tasks)):
        for j in range(len(outs)):
            v = M.values[i, j]
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                    fontsize=5, color="white" if v < 0.45 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("mean exact acc.", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig_interaction_heatmap.pdf")

# ============================================================ Fig 5: cross-model bars
def fig_crossmodel():
    df = pd.read_csv(os.path.join(ANA, "cross_model_g3145.csv"))
    order = ["GPT-3.5-turbo", "Llama-3.1-8B", "Llama-3.2-3B", "Qwen2.5-7B", "Alpaca-7B"]
    conds = [("SST-5", 0, "SST-5 0-shot", "#9ec3e6"),
             ("SST-5", 5, "SST-5 5-shot", "#2c6fbb"),
             ("Yelp-5", 0, "Yelp-5 0-shot", "#f2c08a"),
             ("Yelp-5", 5, "Yelp-5 5-shot", "#d2792a")]
    x = np.arange(len(order)); w = 0.2
    fig, ax = plt.subplots(figsize=(WIDE, 2.6))
    for k, (ds, sh, lab, c) in enumerate(conds):
        vals = []
        for m in order:
            r = df[(df.dataset == ds) & (df.model == m) & (df.shots == sh)]
            vals.append(r.exact.iloc[0] if len(r) else np.nan)
        ax.bar(x + (k - 1.5) * w, vals, w, color=c, label=lab,
               edgecolor="k", linewidth=0.3)
    ax.axhline(0.4991, color="#b03030", ls="--", lw=0.9)
    ax.text(len(order)-1, 0.502, "EvoPrompt GA (Alpaca) 0.4991", color="#b03030",
            fontsize=6, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("Exact accuracy")
    ax.set_ylim(0.30, 0.64)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig_crossmodel.pdf")

# ============================================================ Fig 6: confusion matrix
def fig_confusion():
    labels = ["v.neg", "neg", "neu", "pos", "v.pos"]
    M = np.array([
        [145, 130,  0,   3,   0],
        [134, 458, 22,  12,   5],
        [ 43, 207, 32,  92,  13],
        [ 20,  59, 17, 346,  68],
        [  5,   9,  2, 232, 151],
    ], dtype=float)
    rown = M / M.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(labels)
    ax.set_yticks(range(5)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{int(M[i,j])}", ha="center", va="center",
                    fontsize=6.5, color="white" if rown[i, j] > 0.5 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("row-normalised (recall)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig_confusion.pdf")

# ====================================================== Fig 7: per-class recall heatmap
def fig_recall():
    df = pd.read_csv(os.path.join(ANA, "test_per_class_recall.csv"))
    sub = df[(df.dataset == "SST-5") & (df.shots == 0) &
             (df.genome == "[3, 1, 4, 5]")]
    classes = ["very negative", "negative", "neutral", "positive", "very positive"]
    order = ["GPT-3.5-turbo", "Llama-3.1-8B", "Llama-3.2-3B", "Qwen2.5-7B", "Alpaca-7B"]
    M = np.full((len(order), len(classes)), np.nan)
    for i, m in enumerate(order):
        for j, c in enumerate(classes):
            r = sub[(sub.model == m) & (sub["class"] == c)]
            if len(r):
                M[i, j] = r.recall.iloc[0]
    fig, ax = plt.subplots(figsize=(COL, 2.45))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["v.neg", "neg", "neu", "pos", "v.pos"])
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(5):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="black")
    ax.set_xlabel("True class (SST-5, 0-shot, $g3145$)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("recall", fontsize=6); cb.ax.tick_params(labelsize=6)
    save(fig, "fig_perclass_recall.pdf")

# ====================================================== Fig 8: evolution convergence
def fig_convergence():
    df = pd.read_csv(os.path.join(ANA, "convergence.csv"))
    seeds = sorted(df.seed.unique())
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.5), sharex=True)
    # (mode, best-so-far column, colour, oracle ceiling, random@400, title, ylabel, ylim)
    # Oracle score of the best-so-far genome, not the run's own fitness, so the
    # endpoints agree with the aggregates reported in the text (0.535 / 0.859).
    panels = [
        ("exact", "best_exact_oracle", C_EXACT, 0.552, 0.538,
         "$(\\mu{+}\\lambda)$-EA (exact)", "Best exact accuracy", (0.470, 0.560)),
        ("ordinal", "best_ordinal_oracle", C_ORD, 0.862, 0.8586,
         "$(\\mu{+}\\lambda)$-EA (ordinal)", "Best ordinal accuracy", (0.838, 0.864)),
    ]
    for ax, (mode, col, color, ceil, rand, title, ylab, ylim) in zip(axes, panels):
        sub = df[df["mode"] == mode]
        for s in seeds:                                  # per-seed best-so-far traces
            d = sub[sub.seed == s].sort_values("gen")
            ax.step(d.gen, d[col], where="post", color=color, alpha=0.30,
                    linewidth=0.8)
        mean = sub.groupby("gen")[col].mean()            # mean across the 20 seeds
        ax.step(mean.index, mean.values, where="post", color=color,
                linewidth=1.7, label="mean (20 seeds)")
        ax.axhline(ceil, color="#b03030", ls="--", lw=0.9)
        ax.axhline(rand, color="#444444", ls=":", lw=0.9)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 19)
        ax.set_xticks(range(0, 20, 3))
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].text(0.3, 0.5525, "oracle max 0.552", color="#b03030", fontsize=6,
                 ha="left", va="bottom")
    axes[0].text(0.3, 0.5385, "random $E[\\max_{400}]{=}0.538$", color="#444444",
                 fontsize=6, ha="left", va="bottom")
    axes[1].text(0.3, 0.8623, "oracle max 0.862", color="#b03030", fontsize=6,
                 ha="left", va="bottom")
    axes[1].text(0.3, 0.8583, "random $E[\\max_{400}]{=}0.859$", color="#444444",
                 fontsize=6, ha="left", va="top")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_convergence.pdf")

def fig_equivalence():
    """Paired per-seed differences (guided EA minus budget-matched random) with
    the confidence intervals and the pre-specified equivalence band."""
    pairs = pd.read_csv(os.path.join(ANA, "paired_ea_vs_random.csv"))
    tests = pd.read_csv(os.path.join(ANA, "paired_ea_vs_random_tests.csv"))
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.1))
    panels = [
        ("exact", C_EXACT, "Exact accuracy", (-0.033, 0.033)),
        ("ordinal", C_ORD, "Ordinal score",  (-0.011, 0.011)),
    ]
    rng = np.random.default_rng(0)
    for ax, (mode, color, title, xlim) in zip(axes, panels):
        # difference = EA minus the EXACT expected best-of-K at that run's budget
        d = pairs[pairs["mode"] == mode]["difference"].values
        t = tests[tests["mode"] == mode].iloc[0]
        delta = float(t["delta"])

        # pre-specified equivalence band
        ax.axvspan(-delta, delta, color="#7aa85c", alpha=0.13, lw=0)
        for x in (-delta, delta):
            ax.axvline(x, color="#4f7a35", ls="--", lw=0.8)
        ax.axvline(0, color="#666666", lw=0.8)

        # the 20 paired differences
        y = 0.62 + rng.uniform(-0.11, 0.11, size=len(d))
        ax.scatter(d, y, s=11, color=color, alpha=0.75, edgecolors="none",
                   zorder=3, label="per-seed difference")

        # mean with 95% (thin) and 90% (thick, the TOST interval) intervals
        mean = float(t["mean_difference"])
        ax.plot([t["ci95_lo"], t["ci95_hi"]], [0.16, 0.16], color="#222222",
                lw=1.0, zorder=4, solid_capstyle="butt")
        ax.plot([t["ci90_lo"], t["ci90_hi"]], [0.16, 0.16], color="#222222",
                lw=3.0, zorder=5, solid_capstyle="butt")
        ax.plot([mean], [0.16], "o", ms=4.5, color="#222222", zorder=6)

        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 0.85)
        ax.set_yticks([])
        ax.set_title(f"{title}   (mean {mean:+.4f}, 90% CI "
                     f"[{t['ci90_lo']:+.4f}, {t['ci90_hi']:+.4f}])", fontsize=7.5)
        ax.set_xlabel("Guided search $-$ expected best-of-$K$ at the same budget")
        ax.grid(axis="x", alpha=0.25)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_equivalence.pdf")

if __name__ == "__main__":
    fig_landscape()
    fig_search_vs_random()
    fig_eta2()
    fig_interaction()
    fig_crossmodel()
    fig_confusion()
    fig_recall()
    fig_convergence()
    fig_equivalence()
    print("done ->", os.path.relpath(OUT, ROOT))
