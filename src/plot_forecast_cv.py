import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "FreeSans", "savefig.dpi": 300,
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#333333",
                     "axes.linewidth": 0.8, "xtick.color": "#333333",
                     "ytick.color": "#333333"})

INK, MUTED, HAIR = "#1a1a1a", "#6b6b6b", "#c8c8c8"
COL = {"Ridge": "#24435f", "RandomForest": "#3d6b5e", "XGBoost": "#a8432a"}
LBL = {"Ridge": "Ridge", "RandomForest": "Random Forest", "XGBoost": "XGBoost"}
MODELS, HORIZONS = ["Ridge", "RandomForest", "XGBoost"], [5, 10, 20]

test = pd.read_csv("forecast_summary.csv").pivot_table(
    index=["horizon", "model"], columns="specification", values="test_rmse")
cv = pd.read_csv("cv_results.csv").set_index(["horizon", "model"])

# shared row layout
ypos, y = {}, 0.0
for h in HORIZONS:
    for m in MODELS:
        ypos[(h, m)] = y
        y += 1
    y += 0.9
centers = {h: np.mean([ypos[(h, m)] for m in MODELS]) for h in HORIZONS}

fig = plt.figure(figsize=(15, 6.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.1, 0.85], wspace=0.22,
                      left=0.115, right=0.97, top=0.78, bottom=0.17)
axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))


def frame(ax, title, sub):
    ax.set_ylim(max(ypos.values()) + 0.7, -0.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color="#e3e3e3", lw=0.6)
    ax.set_axisbelow(True)
    for i, h in enumerate(HORIZONS):
        if i % 2 == 0:
            ax.axhspan(centers[h] - 1.45, centers[h] + 1.45, color="#f4f4f4",
                       zorder=0, lw=0)
    ax.text(0, 1.10, title, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=INK)
    ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=8.6, color=MUTED)
    ax.tick_params(axis="x", labelsize=8.5)


# row labels on the far left
for (h, m), yy in ypos.items():
    fig.text(0.108, axA.transData.transform((0, 0))[1], "", ha="right")
for h in HORIZONS:
    pass

# ---------- A: cross-validation ----------
frame(axA, "A.  Cross-validation, training period",
      "Mean RMSE ± 1 SD across 5 time-series folds")
for (h, m), yy in ypos.items():
    mu, sd = cv.loc[(h, m), "cv_mean"], cv.loc[(h, m), "cv_std"]
    folds = [float(v) for v in cv.loc[(h, m), "folds"].split(";")]
    c = COL[m]
    axA.plot([mu - sd, mu + sd], [yy, yy], color=c, lw=5, alpha=0.18,
             solid_capstyle="butt", zorder=1)
    axA.scatter(folds, [yy] * 5, s=9, color=c, alpha=0.45, lw=0, zorder=2)
    axA.scatter(mu, yy, s=48, color=c, edgecolor="white", lw=0.8, zorder=3)
    axA.text(0.99, yy, f"{mu:.4f} ± {sd:.4f}", fontsize=7.8,
             color="#333333", va="center", ha="right",
             transform=axA.get_yaxis_transform(), family="DejaVu Sans Mono")
for h in HORIZONS:
    best = cv.loc[h]["cv_mean"].idxmin()
    axA.scatter(cv.loc[(h, best), "cv_mean"], ypos[(h, best)] - 0.42,
                marker="v", s=26, color=INK, zorder=4)
axA.set_xlim(0.025, 0.20)
axA.set_xticks([0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
axA.set_xlabel("RMSE (annualized realized volatility)", fontsize=9)

# labels for rows and horizons, drawn once at left of panel A
for (h, m), yy in ypos.items():
    axA.text(-0.02, yy, LBL[m], transform=axA.get_yaxis_transform(),
             ha="right", va="center", fontsize=9, color=INK)
for h in HORIZONS:
    axA.text(-0.36, centers[h], f"{h}-day", transform=axA.get_yaxis_transform(),
             ha="left", va="center", fontsize=9.5, fontweight="bold",
             color=INK)

# ---------- B: test, market vs macro ----------
frame(axB, "B.  Held-out test, 2022 to Sep 2026",
      "Market-only vs. market + macro features")
for (h, m), yy in ypos.items():
    a, b = test.loc[(h, m), "market"], test.loc[(h, m), "market_macro"]
    c = COL[m]
    axB.plot([a, b], [yy, yy], color=c, lw=2, alpha=0.35, zorder=1)
    axB.scatter(a, yy, s=48, color=c, edgecolor="white", lw=0.8, zorder=3)
    axB.scatter(b, yy, s=48, facecolor="white", edgecolor=c, lw=1.5, zorder=3)
axB.set_xlim(0.0525, 0.0735)
axB.set_xlabel("Test RMSE", fontsize=9)

# ---------- C: macro penalty heatmap ----------
mat = np.array([[(test.loc[(h, m), "market_macro"] / test.loc[(h, m), "market"] - 1) * 100
                 for h in HORIZONS] for m in MODELS])
axC.imshow(mat, cmap="Reds", vmin=0, vmax=30, aspect="auto")
for i in range(3):
    for j in range(3):
        v = mat[i, j]
        axC.text(j, i, f"+{v:.1f}%", ha="center", va="center", fontsize=11,
                 fontweight="bold", color="white" if v > 15 else INK)
axC.set_xticks(range(3))
axC.set_xticklabels([f"{h}-day" for h in HORIZONS], fontsize=8.5)
axC.set_yticks(range(3))
axC.set_yticklabels([LBL[m] for m in MODELS], fontsize=9)
axC.tick_params(length=0)
for s in axC.spines.values():
    s.set_visible(False)
axC.set_xticks(np.arange(-0.5, 3), minor=True)
axC.set_yticks(np.arange(-0.5, 3), minor=True)
axC.grid(which="minor", color="white", lw=3)
axC.tick_params(which="minor", length=0)
axC.text(0, 1.10, "C.  Cost of adding macro", transform=axC.transAxes,
         fontsize=11, fontweight="bold", color=INK)
axC.text(0, 1.045, "% change in test RMSE vs. market-only",
         transform=axC.transAxes, fontsize=8.6, color=MUTED)

# ---------- header, legend, footer ----------
fig.add_artist(Rectangle((0.03, 0.965), 0.94, 0.004, fc=INK, ec="none",
                         transform=fig.transFigure))
fig.text(0.03, 0.945, "Ridge has the lowest cross-validated error, and macro "
         "features raise test error for every model", fontsize=15,
         fontweight="bold", color=INK, va="top")
fig.text(0.03, 0.895, "XGBoost's small test-period edge (0.001 to 0.002) is far "
         "inside one cross-validation standard deviation (about 0.023).",
         fontsize=10, color="#3a3a3a", va="top")

handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555",
           markeredgecolor="white", markersize=8, label="CV mean / market-only test"),
    Line2D([0], [0], color="#555555", lw=5, alpha=0.25, label="± 1 SD"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555",
           alpha=0.5, markersize=4, label="Individual folds"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
           markeredgecolor="#555555", markeredgewidth=1.5, markersize=8,
           label="Market + macro test"),
    Line2D([0], [0], marker="v", color="none", markerfacecolor=INK,
           markersize=7, label="Lowest CV RMSE"),
]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.03, 0.045),
           ncol=5, frameon=False, fontsize=8.8, handletextpad=0.4,
           columnspacing=1.6)

fig.add_artist(Rectangle((0.03, 0.04), 0.94, 0.0008, fc=HAIR, ec="none",
                         transform=fig.transFigure))
fig.text(0.03, 0.028, "Note: CV uses 5 expanding-window folds on training data "
         "through Dec 2018, with a gap equal to the forecast horizon to prevent "
         "overlapping targets leaking across folds. Market-only features in "
         "panel A.", fontsize=7.8, color=MUTED, va="top")

fig.savefig("forecast_cv_test.png", bbox_inches="tight", facecolor="white")
fig.savefig("forecast_cv_test.pdf", bbox_inches="tight", facecolor="white")