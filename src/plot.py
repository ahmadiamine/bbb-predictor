"""
Make the figures for the README and for sharing.

  python src/plot.py

results/tpsa_vs_logp.png   every molecule placed by polar surface area and
                           greasiness, coloured by whether it reaches the brain.
                           The TPSA = 90 line is the textbook rule of thumb; the
                           blocked molecules pile up to the right of it.
results/split_gap.png      the same model scored two ways, showing how much
                           a random split flatters it.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import features as F

ROOT = Path(__file__).resolve().parents[1]

YES = "#1d4ed8"      # penetrates - blue
NO = "#b45309"       # blocked - amber
INK = "#1f2937"


def scatter(data=None):
    data = data or ROOT / "data" / "bbbp.csv"
    df = pd.read_csv(data)

    rows = []
    for _, r in df.iterrows():
        mol = F.parse(r["smiles"])
        if mol is None:
            continue
        f = F.featurize_mol(mol)
        f["label"] = int(r["label"])
        rows.append(f)
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=160)
    for label, color, name, marker in [
        (1, YES, "Reaches the brain", "o"),
        (0, NO, "Blocked", "^"),
    ]:
        sub = d[d.label == label]
        ax.scatter(sub.tpsa, sub.logp, c=color, label=name, marker=marker,
                   s=26, alpha=0.55, edgecolors="white", linewidths=0.4)

    ax.axvline(90, color=INK, lw=1.1, ls="--", zorder=5)
    ax.annotate("TPSA = 90\nthe textbook cutoff", xy=(94, -5.6), fontsize=9,
                color=INK, va="bottom")

    ax.set_xlim(-5, 260)
    ax.set_ylim(-7, 10)
    ax.set_xlabel("Polar surface area (A$^2$)  →  more polar", fontsize=11)
    ax.set_ylabel("logP  →  more greasy", fontsize=11)
    ax.set_title("Polar molecules do not reach the brain",
                 fontsize=13, pad=12, weight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.13, lw=0.6)

    fig.tight_layout()
    out = ROOT / "results" / "tpsa_vs_logp.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def split_gap():
    m = json.loads((ROOT / "results" / "model.json").read_text())["metrics"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)

    bars = ["Random split\n(flattering)", "Scaffold split\n(honest)"]
    vals = [m["random_auc"], m["auc"]]
    colors = ["#9ca3af", YES]

    b = ax.barh(bars, vals, color=colors, height=0.5)
    for rect, v in zip(b, vals):
        ax.text(v + 0.012, rect.get_y() + rect.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=11, weight="bold", color=INK)

    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color=INK, lw=0.8)
    ax.set_xlabel("ROC-AUC on held-out molecules", fontsize=10)
    ax.set_title(f"The same model, scored two ways  "
                 f"({m['gap']:+.3f} of optimism)",
                 fontsize=12, pad=10, weight="bold")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.13, lw=0.6)

    fig.tight_layout()
    out = ROOT / "results" / "split_gap.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    scatter()
    split_gap()
