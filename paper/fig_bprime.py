import glob, json, os
import numpy as np
from figstyle import *

R = "../results/h2_replication/"
MODELS = [("Qwen3.5-4B", R + "B2_real_qwen3.5-4b.json", R + "B2_null_qwen3.5-4b_*.json", 32),
          ("Gemma 3 12B", R + "B2_real_gemma-3-12b.json", R + "B2_null_gemma-3-12b_*.json", 48),
          ("Llama 3.1 8B", R + "B3_real_llama3.1-8b.json", R + "B3_null_llama3.1-8b_*.json", 32)]


def build(out):
    setup()
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.1), gridspec_kw={"wspace": 0.55})
    for i, (name, real, nulls, n) in enumerate(MODELS):
        ax = axes[i]
        res = json.load(open(real))["result"]
        L = sorted(int(l) for l in res)
        d = np.array([l / (n - 1) for l in L])
        kap = [res[str(l)]["ratio_to_random"]["j_lens"]["kappa"] for l in L]
        neu = [res[str(l)]["ratio_to_random"]["neuron"]["kappa"]["median"] for l in L]
        band(ax, 0.5, 1.0)
        ax.axhspan(0.95, 1.05, color=PALETTE["lightgray"], lw=0, zorder=1)
        ax.plot(d, neu, color=PALETTE["gray"], lw=0.9, marker="s", ms=2.6, label="MLP neuron dirs.", zorder=3)
        m = np.array([k["median"] for k in kap]); lo = np.array([k["ci95"][0] for k in kap]); hi = np.array([k["ci95"][1] for k in kap])
        ax.fill_between(d, lo, hi, color=MODEL_COLORS[name], alpha=0.25, lw=0, zorder=2)
        ax.plot(d, m, color=MODEL_COLORS[name], marker=MODEL_MARKERS[name], ms=3.2, label="J-lens directions", zorder=4)
        pts = []
        for f in sorted(glob.glob(nulls)):
            nj = json.load(open(f))
            for l, v in nj["result"].items():
                pts.append((int(l) / (n - 1), v["ratio_to_random"]["j_lens"]["kappa"]["median"]))
        if pts:
            px = np.array([p[0] for p in pts]); jit = (np.arange(len(pts)) % 5 - 2) * 0.006
            ax.scatter(px + jit, [p[1] for p in pts], s=9, facecolor="white", edgecolor="#222222", lw=0.8, zorder=5, label="randomized weights")
        ax.set_xlim(0, 1)
        ymax = max(m.max(), 1.6) * 1.12
        ax.set_ylim(0.6 if name != "Gemma 3 12B" else 0.0, ymax)
        if i == 0:
            ax.set_ylabel("curvature ratio")
        panel(ax, "abc"[i], name)
    ax = axes[3]
    band(ax, 0.5, 1.0)
    for name, real, _, n in MODELS:
        res = json.load(open(real))["result"]
        L = sorted(int(l) for l in res)
        ax.plot([l / (n - 1) for l in L], [res[str(l)]["ratio_to_random"]["j_lens"]["g1"]["median"] for l in L], color=MODEL_COLORS[name], marker=MODEL_MARKERS[name], ms=3.0, label=name)
    ax.axhline(1.0, color="#777777", lw=0.7, ls="--")
    ax.set_ylim(0.55, 11)
    ax.set_xlim(0, 1); ax.set_yscale("log")
    ax.set_ylabel("gain ratio", labelpad=1)
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax.yaxis.set_major_locator(FixedLocator([0.6, 1, 2, 4])); ax.yaxis.set_major_formatter(FixedFormatter(["0.6", "1", "2", "4"])); ax.minorticks_off()
    panel(ax, "d", "Gain $\\|Jv\\|$")
    ax.legend(loc="upper left", frameon=False, handlelength=1.2, borderaxespad=0.2, fontsize=7, labelspacing=0.2)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = [Line2D([], [], color="#333333", marker="o", ms=3.2), Line2D([], [], color=PALETTE["gray"], marker="s", ms=2.6), Line2D([], [], color="none", marker="o", mfc="white", mec="#222222", ms=4), Patch(color=PALETTE["lightgray"]), Patch(color=PALETTE["band"])]
    fig.legend(hs, ["J-lens directions (model color)", "MLP neuron directions", "randomized weights", "equivalence band [0.95, 1.05]", "depth $\\geq 0.5$ (test range)"], loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=7, handlelength=1.4, columnspacing=1.2)
    fig.supxlabel("depth (layer / last layer)", fontsize=7.5, y=-0.04)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build("figures/fig_bprime.pdf")
    build("/tmp/claude-1000/-home-mv-repos-jacobian-lens/8716fca1-8636-4929-bb78-d12970bfa47a/scratchpad/fig_bprime.png")
