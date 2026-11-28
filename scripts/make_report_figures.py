"""Figures 1 and 2 of the report (report/before-the-wait.md).

    .venv/bin/python scripts/make_report_figures.py . report/figures

Figure 1 is a schematic of the setup; its band widths are not to scale.
Figure 2 is drawn from results/verdicts.csv and results/stronger_rival.csv; no number is typed by hand.
The committed PNGs were rendered with the Inter typeface; without it matplotlib falls back to DejaVu Sans
and the layout shifts slightly. Reads result files only: no probe, rival, activation or example is opened.
"""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle

repo, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
for f in fm.findSystemFonts():
    if "Inter-" in f:
        fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 11, "savefig.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, INK2, GRID = "#191919", "#666666", "#e3e3e3"
BLUE, ORANGE = "#257de0", "#f47230"
DELTA = 0.05

# ---------------------------------------------------------------- figure 2: verdicts
V = {r["band"]: r for r in csv.DictReader(open(repo / "results/verdicts.csv"))}
S = next(csv.DictReader(open(repo / "results/stronger_rival.csv")))
rows = [(f"{b.replace('-', '–')} tokens", float(V[b]["lift"]), float(V[b]["interval_low"]),
         float(V[b]["interval_high"]), V[b]["verdict_0.05"], BLUE) for b in ["2-5", "6-10", "11-20", "21-40"]]
rows.append(("6–10 tokens", float(S["lift"]), float(S["interval_low"]), float(S["interval_high"]),
             S["verdict_0.05"], ORANGE))

fig, ax = plt.subplots(figsize=(9.2, 5.5))
ys = [5.2, 4.2, 3.2, 2.2, 0.6]
ax.plot([0, 0], [0, 5.85], color=INK2, lw=1)
ax.plot([DELTA, DELTA], [0, 5.85], color=INK2, lw=1, ls=(0, (4, 3)))
ax.text(0, 6.05, "0: no lift", ha="center", va="bottom", color=INK2, fontsize=10)
ax.text(DELTA, 6.05, "δ = 0.05: the bar", ha="center", va="bottom", color=INK2, fontsize=10)
for (name, lift, lo, hi, verdict, col), y in zip(rows, ys):
    ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
    ax.plot([lift], [y], "o", color=col, ms=9, mec="white", mew=2)
    ax.text(0.094, y, f"{lift:+.3f}".replace("-", "−"), va="center", ha="right", color=INK, fontsize=10.5)
    ax.text(0.098, y, verdict, va="center", ha="left", color=INK, fontsize=10.5)
ax.text(0.094, 6.05, "lift", ha="right", va="bottom", color=INK2, fontsize=10)
ax.text(0.098, 6.05, "verdict", ha="left", va="bottom", color=INK2, fontsize=10)
tr = ax.get_yaxis_transform()
ax.text(-0.3, 5.85, "Against the combined outside reader", color=INK, fontsize=10.5, fontweight="bold", va="center", transform=tr, clip_on=False)
ax.text(-0.3, 1.25, "Against the stronger reader (7B)", color=INK, fontsize=10.5, fontweight="bold", va="center", transform=tr, clip_on=False)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], color=INK)
ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors=INK2)
ax.set_ylim(0, 6.5); ax.set_xlim(-0.03, 0.125)
ax.set_xticks([-0.025, 0, 0.025, 0.05, 0.075]); ax.set_xticklabels(["−0.025", "0", "+0.025", "+0.05", "+0.075"])
ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color(GRID)
ax.set_xlabel("Lift: probe AUROC minus reader AUROC, on test (point and 98.75% interval)", color=INK2, labelpad=8)
fig.subplots_adjust(left=0.27, right=0.98, top=0.975, bottom=0.30)
key = [("positive", "the interval is entirely above 0 and reaches the bar"),
       ("null", "the interval is entirely below the bar, so the lift is smaller than δ"),
       ("inconclusive", "the interval includes 0 and reaches the bar: the data cannot tell")]
fig.text(0.03, 0.155, "The verdict is read from the interval, not from the point:", fontsize=10, color=INK2)
for i, (k, t) in enumerate(key):
    fig.text(0.03, 0.108 - 0.043 * i, k, fontsize=10, color=INK, fontweight="bold")
    fig.text(0.155, 0.108 - 0.043 * i, t, fontsize=10, color=INK2)
fig.savefig(out / "fig2_verdicts.png", facecolor="white"); plt.close(fig)

# ---------------------------------------------------------------- figure 1: setup
fig, ax = plt.subplots(figsize=(10, 5.0)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 50)

def box(x, y, w, h, title, lines, edge, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fill, ec=edge, lw=1.5))
    ax.text(x + 1.6, y + h - 2.3, title, fontsize=10.5, fontweight="bold", color=INK, va="center")
    for i, t in enumerate(lines):
        ax.text(x + 1.6, y + h - 5.3 - 2.9 * i, t, fontsize=9.5, color=INK2, va="center")

# the trace strip
y0, h = 25, 5
ax.text(4, y0 + h + 9.2, "One reasoning trace, read left to right", fontsize=10, color=INK2)
bands = [("21–40", 6, 34, "#eef3fa"), ("11–20", 34, 55, "#dde8f6"), ("6–10", 55, 70, "#c6d9f1"), ("2–5", 70, 82, "#dde8f6")]
ax.text(4.2, y0 + h / 2, "…", fontsize=12, color=INK2, va="center")
for name, a, b, col in bands:
    ax.add_patch(Rectangle((a, y0), b - a, h, fc=col, ec="white", lw=2))
    ax.text((a + b) / 2, y0 + h + 1.6, name, ha="center", fontsize=9.5, color=INK2)
ax.text(44, y0 + h + 4.6, "tokens before the marker: four bands, each tested on its own", ha="center", fontsize=9.5, color=INK2)
ax.add_patch(Rectangle((82, y0), 3, h, fc="#f5f5f5", ec="white", lw=2))
ax.add_patch(FancyBboxPatch((85.3, y0 - 0.3), 9.4, h + 0.6, boxstyle="round,pad=0.2,rounding_size=1", fc=INK, ec=INK))
ax.text(90, y0 + h / 2, "“Wait”", color="white", ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(90, y0 + h + 1.6, "marker", ha="center", fontsize=9.5, color=INK2)
# position t
tx = 62
ax.add_patch(Rectangle((tx - 0.9, y0 - 0.6), 1.8, h + 1.2, fc=BLUE, ec="white", lw=1.5))
ax.text(tx, y0 - 2.6, "position t", ha="center", fontsize=9.5, color=INK, fontweight="bold")
# question
ax.text(50, 46.5, "At position t, is a marker coming in this band?  Three readers answer.",
        ha="center", fontsize=11, color=INK, fontweight="bold")
# readers
box(3, 1, 29, 17, "Inside reader (the probe)", ["A linear readout of the 1.5B's", "own state at t, at one middle", "layer (chosen from layers 1–27)"], BLUE, "#f3f8fe")
box(35.5, 1, 29, 17, "Outside reader", ["Everything visible up to t:", "the text, output uncertainty,", "the output-layer state, position"], INK2)
box(68, 1, 29, 17, "Stronger reader", ["The same kind of readout on a", "larger sibling model (7B) that", "reads the same tokens.", "Band 6–10 only"], ORANGE)
for bx, col in ((17.5, BLUE), (50, INK2), (82.5, ORANGE)):
    ax.annotate("", xy=(bx, 18.4), xytext=(tx, y0 - 4.2), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, shrinkA=0, shrinkB=0))
fig.savefig(out / "fig1_setup.png", facecolor="white", bbox_inches="tight", pad_inches=0.25); plt.close(fig)
print("ok")
