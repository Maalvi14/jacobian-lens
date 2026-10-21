import json
import numpy as np
from figstyle import *
from matplotlib.colors import ListedColormap

R = "../results/"
SCR = "/tmp/claude-1000/-home-mv-repos-jacobian-lens/8716fca1-8636-4929-bb78-d12970bfa47a/scratchpad/"
ALPHAS = [round(a, 2) for a in np.arange(-0.05, 2.0001, 0.05)]

# status codes: 2 = supported, 1 = weak / partial, 0 = not supported, -1 = not tested
ROWS = [
    ("Curvature along lens directions\nexceeds random (B$'$)", [2, 2, 2, -1]),
    ("Inhibition at the ignition\nthreshold is second order (A)", [2, 0, -1, 0]),
    ("Nonlinear gain explains\nswap success (E1)", [0, 0, -1, -1]),
    ("Trust score $\\tau$ flags\nswap failures (H1)", [0, 0, -1, -1]),
    ("Ignition is a curvature\nevent (H6)", [1, 0, -1, -1]),
    ("Motor collapse (H8)", [0, 0, -1, -1]),
    ("Binding in cross-terms (H3)", [0, -1, -1, -1]),
    ("Remainder coupled (H5)", [0, -1, -1, -1]),
    ("Early-layer curvature (H7)", [0, -1, -1, -1]),
    ("Load as competition (H9)", [0, -1, -1, -1]),
    ("Persona-specific (H10)", [0, -1, -1, -1]),
]
COLS = ["Qwen\n3.5-4B", "Gemma\n3 12B", "Llama\n3.1 8B", "Qwen\n3.6-27B"]


def build(out):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.15), gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.75})
    ax = axes[0]
    d = json.load(open(R + "e1_gemma_heldout.json"))["items"]
    a = np.array(ALPHAS)
    ax.axhline(0, color="#999999", lw=0.5)
    for grp, col, lab in ((True, PALETTE["orange"], "swap flips the answer"), (False, PALETTE["gray"], "no flip")):
        items = [r for r in d if bool(r["hit"]) == grp]
        P = np.array([r["band"]["path"] for r in items]); FO = np.array([r["band"]["m0"] + r["band"]["g"] * a for r in items])
        q1, md, q3 = np.percentile(P, [25, 50, 75], axis=0)
        ax.fill_between(a, q1, q3, color=col, alpha=0.18, lw=0)
        ax.plot(a, md, color=col, lw=1.5, label=f"{lab} (n={len(items)}): actual")
        ax.plot(a, np.median(FO, axis=0), color=col, lw=1.0, ls="--", label=f"{lab}: first-order")
    ax.axvline(1.0, color="#333333", lw=0.5, ls=":")
    ax.axvspan(1.3, 2.0, color="#EEEEEE", lw=0, zorder=0)
    ax.set_ylim(-25, 75)
    ax.text(1.02, -24, "α$_{ref}$", fontsize=5.8, va="bottom")
    ax.text(1.65, 66, "beyond the\nworking range", fontsize=5.2, color="#777777", ha="center", va="top")
    ax.set_xlabel("swap strength α"); ax.set_ylabel("margin  logit(swap) − logit(answer)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.8, fontsize=5.2, borderaxespad=0.1)
    panel(ax, "a", "Gemma 3 12B swap paths (median, IQR)")
    ax = axes[1]
    M = np.array([r[1] for r in ROWS])
    cmap = ListedColormap(["#FFFFFF", "#D4D4D4", "#F2C46D", "#4C9F70"])
    ax.imshow(M + 1, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    for i in range(M.shape[0] + 1):
        ax.axhline(i - 0.5, color="white", lw=1.2)
    for j in range(M.shape[1] + 1):
        ax.axvline(j - 0.5, color="white", lw=1.2)
    sym = {2: "✓", 1: "~", 0: "✗", -1: "–"}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, sym[v], ha="center", va="center", fontsize=8, color="white" if v == 2 else "#444444")
    ax.set_xticks(range(4)); ax.set_xticklabels(COLS, fontsize=5.6); ax.xaxis.tick_top()
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS], fontsize=5.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("(b) Outcome of each pre-registered test", loc="left", pad=14, fontsize=7.5, color="#666666", fontweight="bold")
    ax.text(0.0, -0.06, "✓ supported   ~ weak   ✗ not supported   – not tested", transform=ax.transAxes, fontsize=5.4, color="#555555", va="top")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    build("figures/fig_overview.pdf"); build(SCR + "fig_overview.png")
