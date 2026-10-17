"""3D feature-space figure for the README, built from the real per-question
feature table (question_features.csv, 500 questions, run-2).

Axes: mean entropy, mean Fisher-Rao candidate distance, mean NLL.
Color: final answer wrong (red) vs correct (blue).

The figure makes the study's core fact visible without any statistics:
the two classes overlap in feature space. No rotation of this cloud makes
the red and blue points separate cleanly - which is exactly why even the
best monitor stops around AUROC 0.77 and why the geometry features added
nothing beyond entropy/NLL/length.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

df = pd.read_csv("analysis/outputs2/question_features.csv")

x = df["mean_H"].to_numpy()
y = df["mean_altdist"].to_numpy()
z = df["mean_nll"].to_numpy()
wrong = df["wrong"].to_numpy().astype(bool)

C_RIGHT = "#1E88E5"  # correct answers: blue
C_WRONG = "#E53935"  # wrong answers: red

fig = plt.figure(figsize=(10, 7.5))
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=14, azim=-58)

ax.scatter(
    x[~wrong], y[~wrong], z[~wrong],
    s=26, c=C_RIGHT, alpha=0.55, edgecolors="white", linewidths=0.3,
    depthshade=False, label="correct answer",
)
# wrong answers drawn last, slightly larger, so they stay visible
ax.scatter(
    x[wrong], y[wrong], z[wrong],
    s=42, c=C_WRONG, alpha=0.85, edgecolors="white", linewidths=0.5,
    depthshade=False, label="wrong answer",
)

ax.set_xlabel("mean next-token entropy", labelpad=10)
ax.set_ylabel("mean Fisher–Rao candidate distance", labelpad=10)
ax.set_zlabel("mean NLL (surprise)", labelpad=6)
ax.set_title("Every question, all three feature axes at once — 500 GSM8K questions (real data)", pad=14)

ax.xaxis.pane.set_alpha(0.04)
ax.yaxis.pane.set_alpha(0.04)
ax.zaxis.pane.set_alpha(0.04)
ax.tick_params(labelsize=8)
ax.dist = 10

handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_RIGHT,
           markersize=9, label=f"correct (n={int((~wrong).sum())})"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_WRONG,
           markersize=9, label=f"wrong (n={int(wrong.sum())})"),
    Line2D([0], [0], color="#444444", lw=1.5, ls="--",
           label="no plane separates the colors\n(this is the null result, drawn)"),
]
ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9)

fig.tight_layout()
fig.savefig("analysis/figures/fig_3d_feature_space.png", dpi=170, bbox_inches="tight")
print("saved analysis/figures/fig_3d_feature_space.png")
print("points:", df.shape[0], "| wrong:", int(wrong.sum()),
      "| mean_H corr with wrong:",
      round(float(np.corrcoef(x, df['wrong'])[0, 1]), 3),
      "| mean_altdist corr with wrong:",
      round(float(np.corrcoef(y, df['wrong'])[0, 1]), 3))
