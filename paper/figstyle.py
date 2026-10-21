"""Shared plotting style for the paper figures (modelled on the workspace paper: small multi-panel line plots, a light shaded band for the layer
range of interest, gray panel titles, direct annotations)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"orange": "#E8762B", "blue": "#2F6DA6", "gray": "#8A8A8A", "green": "#4C9F70", "red": "#C0504D", "purple": "#7B5EA7", "teal": "#2A9D9D",
           "lightgray": "#D9D9D9", "band": "#FBF3DC"}
MODEL_COLORS = {"Qwen3.5-4B": "#E8762B", "Gemma 3 12B": "#2F6DA6", "Llama 3.1 8B": "#4C9F70", "Qwen3.6-27B": "#7B5EA7"}


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": "#444444", "axes.titlecolor": "#666666", "axes.titleweight": "bold",
        "axes.titlelocation": "left", "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "lines.linewidth": 1.2, "axes.grid": False,
    })


def panel(ax, label, title):
    ax.set_title(f"({label}) {title}", loc="left", pad=4)


def band(ax, x0, x1, color=PALETTE["band"], label=None):
    ax.axvspan(x0, x1, color=color, lw=0, zorder=0, label=label)
