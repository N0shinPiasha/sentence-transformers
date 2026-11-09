"""
Build the figures used in the final report (report/figures/*.pdf).

    python report/make_figures.py

Needs results/sbert_predictions.csv (from sbert_replicate.py) and
results/ulo_scores.csv (from ulo_dataset/evaluate_ulo.py). No model or internet needed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from matplotlib.patches import Rectangle            # noqa: E402
from scipy.stats import spearmanr                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# validated categorical slots 1-3 (light mode) + text / grid tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
    "ytick.color": TEXT2, "xtick.major.size": 0, "ytick.major.size": 0, "legend.fontsize": 7,
    "text.color": TEXT, "pdf.fonttype": 42,
})


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)


sts = pd.read_csv(ROOT / "results" / "sbert_predictions.csv")
ulo = pd.read_csv(ROOT / "results" / "ulo_scores.csv")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.28})

# ---------- (a) label distributions
sts_pct = pd.Series(np.clip(np.rint(sts["gold_0to5"]), 0, 5).astype(int)).value_counts(normalize=True)
ulo_pct = ulo["llm_score"].value_counts(normalize=True)
ks = np.arange(6)
a1.bar(ks - 0.175, [sts_pct.get(k, 0) * 100 for k in ks], width=0.33, color=BLUE, linewidth=0, zorder=3)
a1.bar(ks + 0.175, [ulo_pct.get(k, 0) * 100 for k in ks], width=0.33, color=ORANGE, linewidth=0, zorder=3)
a1.set_xlim(-0.6, 5.6)
a1.set_ylim(0, 46)
a1.set_xticks(range(6))
a1.set_xlabel("Similarity score (0 = unrelated, 5 = equivalent)")
a1.set_ylabel("Share of pairs (%)")
a1.set_title("(a) Label distribution", loc="left", color=TEXT)
n_top = int((ulo["llm_score"] >= 4).sum())
a1.text(4.75, 25.0, f"ULO: {n_top} pairs\nat 4 or 5", fontsize=6.5, color=TEXT2, ha="center", va="bottom")
a1.legend(handles=[Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color=ORANGE)],
          labels=[f"STS-B test, human ({len(sts):,})", f"ULO, LLM judge ({len(ulo)})"],
          frameon=False, loc="upper right", handlelength=1, borderaxespad=0)
style(a1)

# ---------- (b) SBERT cosine against the LLM label, by pair type
rng = np.random.default_rng(42)
types = [("same_unit", "Same unit", BLUE, -0.22), ("same_discipline", "Same discipline", ORANGE, 0.0),
         ("cross_discipline", "Cross-discipline", AQUA, 0.22)]
for key, label, color, off in types:
    g = ulo[ulo["pair_type"] == key]
    x = g["llm_score"] + off + rng.uniform(-0.07, 0.07, len(g))
    a2.scatter(x, g["sbert_cosine"], s=14, color=color, edgecolor="white", linewidth=0.6, label=label, zorder=3)
for k, med in ulo.groupby("llm_score")["sbert_cosine"].median().items():
    a2.plot([k - 0.36, k + 0.36], [med, med], color=TEXT2, linewidth=1.0, zorder=2)
rho = spearmanr(ulo["sbert_cosine"], ulo["llm_score"]).correlation * 100
a2.set_xticks(range(6))
a2.set_xlim(-0.5, 5.5)
a2.set_ylim(0.15, 1.0)
a2.set_xlabel("LLM judge score")
a2.set_ylabel("SBERT cosine similarity")
a2.set_title(f"(b) SBERT vs LLM labels on ULO pairs (Spearman {rho:.1f}; line = median)", loc="left", color=TEXT)
a2.legend(frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0, markerscale=1.2)
style(a2)

fig.savefig(OUT / "ulo_labels_and_scores.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "ulo_labels_and_scores.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print(f"Saved {OUT / 'ulo_labels_and_scores.pdf'}")
