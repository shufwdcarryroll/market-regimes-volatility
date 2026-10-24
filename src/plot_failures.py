import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "FreeSans", "savefig.dpi": 300,
                     "pdf.fonttype": 42})

INK, MUTED, HAIR = "#1a1a1a", "#6b6b6b", "#c8c8c8"
PRED, ACT = "#9aa5b1", "#24435f"

cases = [
    dict(tag="SCHEDULED MACRO SURPRISE", color="#584a7a",
         date="Nov 7, 2022", h="5-day", actual=0.4229, pred=0.1786,
         verdict="Underpredicted 2.4x",
         happened="Three days later October CPI came in below forecast and the "
                  "S&P 500 jumped 5.5%, its best day since 2020. Midterm "
                  "elections and the FTX collapse landed in the same week.",
         why="The volatility came from a rally, not a selloff. Inputs on Nov 7 "
             "were elevated but stable, and nothing in the features says a "
             "CPI release falls inside the forecast window.",
         fix="Calendar flags for CPI, FOMC and payrolls dates inside the "
             "horizon."),
    dict(tag="SPIKE THAT FADES FAST", color="#3d6b5e",
         date="Aug 5, 2024", h="20-day", actual=0.1479, pred=0.3337,
         verdict="Overpredicted 2.3x",
         happened="The yen carry trade unwound and the VIX closed near 39 "
                  "after touching 65 intraday. Within two weeks the S&P 500 "
                  "had its best week of the year and the VIX was back near 15.",
         why="Trailing RV and the VIX were at extremes, so the model "
             "extrapolated stress forward. The shock was a positioning "
             "flush, not a change in fundamentals, and reverted quickly.",
         fix="VIX term structure and vol-of-vol features to separate panic "
             "spikes from persistent stress."),
    dict(tag="UNSCHEDULED POLICY SHOCK", color="#a8432a",
         date="Apr 2, 2025", h="5-day", actual=0.8629, pred=0.1670,
         verdict="Underpredicted 5.2x",
         happened="Liberation Day tariffs were announced after the close. "
                  "The S&P 500 fell 4.8% and 6.0% on the next two days, then "
                  "rose 9.5% on April 9 when a 90-day pause was announced.",
         why="Forecast made the day before the shock with a calm market. "
             "No market or macro input at time t carried information about "
             "the announcement.",
         fix="Policy uncertainty or news indices, plus quantile forecasts "
             "that report tail risk, not just the mean."),
]

fig = plt.figure(figsize=(10.5, 7.0))
fig.add_artist(Rectangle((0.03, 0.965), 0.94, 0.004, fc=INK, ec="none",
                         transform=fig.transFigure))
fig.text(0.03, 0.945, "Three ways the Ridge forecast failed in the test period",
         fontsize=15, fontweight="bold", color=INK, va="top")
fig.text(0.03, 0.902, "Forward realized volatility, annualized. Bars share one "
         "scale so the size of each miss is comparable.",
         fontsize=9.8, color="#3a3a3a", va="top")

W, GAP, X0, TOP = 0.30, 0.02, 0.03, 0.86
SCALE = 0.95  # max RV on bar scale

for k, c in enumerate(cases):
    x = X0 + k * (W + GAP)
    ax = fig.add_axes([x, 0.07, W, TOP - 0.07])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # header
    ax.add_patch(Rectangle((0, 0.935), 1, 0.065, fc=c["color"], ec="none"))
    ax.text(0.04, 0.967, c["tag"], fontsize=8.6, fontweight="bold",
            color="white", va="center")
    ax.text(0, 0.885, c["date"], fontsize=13, fontweight="bold", color=INK,
            va="center")
    ax.text(1, 0.885, f"{c['h']} horizon", fontsize=8.8, color=MUTED,
            ha="right", va="center")

    # bars
    for i, (lbl, v, col) in enumerate([("Predicted", c["pred"], PRED),
                                       ("Actual", c["actual"], ACT)]):
        yb = 0.80 - i * 0.07
        ax.text(0, yb, lbl, fontsize=8.6, color="#333333", va="center")
        bx, bw = 0.27, 0.58
        ax.add_patch(Rectangle((bx, yb - 0.022), bw, 0.044, fc="#f0f0f0",
                               ec="none"))
        ax.add_patch(Rectangle((bx, yb - 0.022), bw * v / SCALE, 0.044,
                               fc=col, ec="none"))
        ax.text(bx + bw * v / SCALE + 0.02, yb, f"{v:.2f}", fontsize=8.8,
                color=INK, va="center", family="DejaVu Sans Mono")
    ax.text(0, 0.635, c["verdict"], fontsize=11, fontweight="bold",
            color=c["color"], va="center")
    ax.text(1, 0.635, f"error {abs(c['actual'] - c['pred']):.2f}",
            fontsize=8.8, color=MUTED, ha="right", va="center",
            family="DejaVu Sans Mono")
    ax.plot([0, 1], [0.6, 0.6], color=HAIR, lw=0.7)

    # narrative
    y = 0.565
    for head, body in [("What happened", c["happened"]),
                       ("Why the model missed", c["why"]),
                       ("Possible fix", c["fix"])]:
        ax.text(0, y, head.upper(), fontsize=7.8, fontweight="bold",
                color=MUTED, va="top")
        wrapped = textwrap.fill(body, 46)
        ax.text(0, y - 0.035, wrapped, fontsize=8.9, color=INK, va="top",
                linespacing=1.35)
        y -= 0.05 + 0.031 * (wrapped.count("\n") + 1) + 0.03

fig.add_artist(Rectangle((0.03, 0.06), 0.94, 0.0008, fc=HAIR, ec="none",
                         transform=fig.transFigure))
fig.text(0.03, 0.045, "Note: Ridge, market-only features. Actual RV uses returns "
         "from t+1 to t+h. Market moves are S&P 500 daily closes.",
         fontsize=7.8, color=MUTED, va="top")

fig.savefig("failure_analysis.png", bbox_inches="tight", facecolor="white")
fig.savefig("failure_analysis.pdf", bbox_inches="tight", facecolor="white")