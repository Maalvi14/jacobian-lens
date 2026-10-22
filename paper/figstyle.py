"""Shared plotting style for the paper figures (modelled on the workspace paper: small multi-panel line plots, a light shaded band for the layer
range of interest, gray panel titles, direct annotations)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"orange": "#E8762B", "blue": "#2F6DA6", "gray": "#8A8A8A", "green": "#4C9F70", "red": "#C0504D", "purple": "#7B5EA7", "teal": "#2A9D9D",
           "lightgray": "#D9D9D9", "band": "#FBF3DC"}
MODEL_MARKERS = {"Qwen3.5-4B": "o", "Gemma 3 12B": "s", "Llama 3.1 8B": "^", "Qwen3.6-27B": "D"}
MODEL_COLORS = {"Qwen3.5-4B": "#E8762B", "Gemma 3 12B": "#2F6DA6", "Llama 3.1 8B": "#4C9F70", "Qwen3.6-27B": "#7B5EA7"}


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": "#444444", "axes.titlecolor": "#666666", "axes.titleweight": "bold",
        "axes.titlelocation": "left", "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "lines.linewidth": 1.2, "axes.grid": False,
    })


def panel(ax, label, title):
    ax.set_title(f"({label}) {title}", loc="left", pad=4)


def band(ax, x0, x1, color=PALETTE["band"], label=None):
    ax.axvspan(x0, x1, color=color, lw=0, zorder=0, label=label)


def heat(ax, M, P=None, *, cmap="YlGnBu", vmin=0.0, vmax=1.0, fmt="{:.2f}", fs=7, xlabels=None, ylabels=None, x0=0):
    """Heatmap drawn as explicit rectangles (no image resampling, so no stripe artefacts in PDF viewers). Text is dark on light cells and white on dark ones."""
    import numpy as np
    from matplotlib.patches import Rectangle
    cm = plt.get_cmap(cmap)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            t = (v - vmin) / (vmax - vmin)
            ax.add_patch(Rectangle((x0 + j - 0.5, i - 0.5), 1, 1, facecolor=cm(t), edgecolor="white", lw=0.8))
            ax.text(x0 + j, i, fmt.format(v) + ("*" if (P is not None and P[i, j]) else ""), ha="center", va="center", fontsize=fs, color="white" if t > 0.55 else "#222222")
    ax.set_xlim(x0 - 0.5, x0 + M.shape[1] - 0.5); ax.set_ylim(M.shape[0] - 0.5, -0.5)
