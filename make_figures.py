"""Draw the manuscript figures from analysis_results.json and supplementary_results.json.

Usage: python make_figures.py OUTPUT_DIR
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
OUT.mkdir(parents=True, exist_ok=True)

A = json.loads((HERE / "analysis_results.json").read_text(encoding="utf-8"))["statistics"]
S = json.loads((HERE / "supplementary_results.json").read_text(encoding="utf-8"))
SEEDS = ["20260921", "20260922", "20260923"]
PARENT = {s: f"Parent {i + 1}" for i, s in enumerate(SEEDS)}

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "Latin Modern Roman", "mathtext.fontset": "cm", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 3,
})


def panel(name):
    v = A[name]
    return v[2], v[3], v[4]


# ---------------------------------------------------------------- Figure 2: forest plot
rows = []  # (label, est, lo, hi, color, marker, is_pooled)
rows.append(("header", "(a) Same table: unseen vs. three replay copies"))
for s in SEEDS:
    rows.append((PARENT[s], *panel(f"replication/{s}/direct"), BLUE, "o", False))
p = S["direct_pooled"]
rows.append(("Pooled", p["estimate"], *p["ci95"], INK, "D", True))

rows.append(("header", "(b) Continuation vs. three copies of its own parent"))
for s in SEEDS:
    rows.append((f"{PARENT[s]}, unseen", *panel(f"supplement/{s}/new"), BLUE, "o", False))
    rows.append((f"{PARENT[s]}, replay", *panel(f"supplement/{s}/old"), ORANGE, "s", False))
for key, lab, col, mk in (("unseen_vs_parent_pooled", "Pooled, unseen", BLUE, "D"),
                          ("replay_vs_parent_pooled", "Pooled, replay", ORANGE, "D")):
    rows.append((lab, S[key]["estimate"], *S[key]["ci95"], col, mk, True))

rows.append(("header", "(c) Continuation vs. three copies of Mortal"))
for s in SEEDS:
    rows.append((f"{PARENT[s]}, unseen", *panel(f"replication/{s}/mortal/new"), BLUE, "o", False))
    rows.append((f"{PARENT[s]}, replay", *panel(f"replication/{s}/mortal/old"), ORANGE, "s", False))
m = S["vs_mortal"]
rows.append(("Pooled, all six", m["pooled_gap"], *m["pooled_gap_ci95"], INK, "D", True))

fig, ax = plt.subplots(figsize=(6.0, 5.4))
y = 0
yt, yl = [], []
for r in rows:
    if r[0] == "header":
        y -= 0.4
        ax.text(-0.52, y, r[1], fontsize=8.5, fontweight="bold", color=INK, va="center",
                transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
        y -= 1
        continue
    lab, est, lo, hi, col, mk, pooled = r
    ax.plot([lo, hi], [y, y], color=col, lw=2.2 if pooled else 1.4, solid_capstyle="round")
    ax.plot(est, y, marker=mk, color=col, ms=6.5 if pooled else 5, mec="white", mew=0.8, zorder=3)
    yt.append(y)
    yl.append(lab)
    y -= 1
ax.axvline(0, color=MUTED, lw=0.8, zorder=0)
ax.set_yticks(yt)
ax.set_yticklabels(yl)
for t, r in zip(ax.get_yticklabels(), [r for r in rows if r[0] != "header"]):
    if r[6]:
        t.set_fontweight("bold")
ax.set_xlim(-0.08, 0.15)
ax.set_ylim(y + 0.3, 0.4)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.set_xlabel(r"Rank gap $\Delta$ (lower favours the challenger)")
ax.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_forest.pdf")
fig.savefig(OUT / "fig_forest.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: CE vs play
fig, ax = plt.subplots(figsize=(3.6, 2.5))
for r in S["ce_vs_play"]["rows"]:
    s, b = r["parent"], r["branch"]
    est, lo, hi = panel(f"supplement/{s}/{'new' if b == 'unseen' else 'old'}")
    x = 100 * r["relative_delta_ce"]
    col, mk = (BLUE, "o") if b == "unseen" else (ORANGE, "s")
    ax.plot([x, x], [lo, hi], color=col, lw=1.2, alpha=0.8)
    ax.plot(x, est, marker=mk, color=col, ms=6, mec="white", mew=0.8, zorder=3)
    ax.annotate(PARENT[s].replace("Parent ", "P"), (x, est), xytext=(4, -3), textcoords="offset points",
                fontsize=7, color=MUTED)
ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
ax.plot([], [], "o", color=BLUE, label="Unseen-data continuation")
ax.plot([], [], "s", color=ORANGE, label="Replay continuation")
ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3)
ax.set_xlabel("Change in dev. cross-entropy vs. parent (%)")
ax.set_ylabel(r"Rank gap $\Delta$ vs. parent")
ax.set_xlim(-1.95, -1.45)
ax.set_ylim(-0.085, 0.085)
ax.grid(color=GRID, lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_ce_vs_play.pdf")
fig.savefig(OUT / "fig_ce_vs_play.png", dpi=200)
plt.close(fig)
print("figures written to", OUT)
