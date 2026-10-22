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
    ("Inhibition at ignition is\nsecond order (A)", [2, 0, -1, 0]),
    ("Nonlinear gain explains\nswap success (E1)", [0, 0, -1, -1]),
    ("Trust score $\\tau$ flags\nswap failures (H1)", [0, 0, -1, -1]),
    ("Ignition is a curvature\nevent (H6)", [1, 0, -1, -1]),
    ("Motor collapse (H8)", [0, 0, -1, -1]),
    ("Binding in cross-terms (H3)", [0, -1, -1, -1]),
    ("Remainder coupled (H5)", [0, -1, -1, -1]),
    ("Early-layer curvature (H7)", [0, -1, -1, -1]),
    ("Load as competition (H9-load)", [0, -1, -1, -1]),
    ("Persona-specific (H10)", [0, -1, -1, -1]),
]
COLS = ["Qwen\n3.5-4B", "Gemma\n3 12B", "Llama\n3.1 8B", "Qwen\n3.6-27B"]
FILL = {2: ("#2F7D55", "white", "✓"), 1: ("#E9B949", "#222222", "~"), 0: ("#D0D0D0", "#222222", "✗"), -1: ("#FFFFFF", "#777777", "–")}


def build(out):
    from matplotlib.patches import Rectangle
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.15), gridspec_kw={"width_ratios": [1.0, 1.12], "wspace": 0.95})
    ax = axes[0]
    d = json.load(open(R + "e1_gemma_heldout.json"))["items"]
    a = np.array(ALPHAS)
    ax.axhline(0, color="#999999", lw=0.5)
    for grp, col, lab in ((True, PALETTE["orange"], "flips"), (False, PALETTE["gray"], "no flip")):
        items = [r for r in d if bool(r["hit"]) == grp]
        P = np.array([r["band"]["path"] for r in items]); FO = np.array([r["band"]["m0"] + r["band"]["g"] * a for r in items])
        q1, md, q3 = np.percentile(P, [25, 50, 75], axis=0)
        ax.fill_between(a, q1, q3, color=col, alpha=0.18, lw=0)
        ax.plot(a, md, color=col, lw=1.6, label=f"{lab} (n={len(items)}): actual")
        ax.plot(a, np.median(FO, axis=0), color=col, lw=1.1, ls="--", label=f"{lab}: first-order")
    ax.axvline(1.0, color="#333333", lw=0.5, ls=":")
    ax.axvspan(1.3, 2.0, color="#EEEEEE", lw=0, zorder=0)
    ax.set_ylim(-25, 75)
    ax.text(1.03, -23, "α$_{ref}$", fontsize=7, va="bottom")
    ax.text(1.66, 32, "beyond\nthe working\nrange", fontsize=7, color="#333333", ha="center", va="center", bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=1.2))
    ax.set_xlabel("swap strength α"); ax.set_ylabel("margin  logit(swap) − logit(answer)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), handlelength=2.2, fontsize=7, ncol=2, columnspacing=1.0)
    panel(ax, "a", "Gemma 3 12B swap paths")
    ax = axes[1]
    M = np.array([r[1] for r in ROWS])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            fc, tc, sym = FILL[M[i, j]]
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=fc, edgecolor="#BBBBBB" if M[i, j] == -1 else "white", lw=0.8))
            ax.text(j, i, sym, ha="center", va="center", fontsize=9, color=tc, fontweight="bold")
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(M.shape[0] - 0.5, -0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(COLS, fontsize=7); ax.xaxis.tick_top()
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS], fontsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("(b) Outcome of each test", loc="left", pad=26, fontsize=8, color="#666666", fontweight="bold")
    ax.text(0.0, -0.05, "✓ supported   ~ weak   ✗ not supported   – not tested", transform=ax.transAxes, fontsize=7, color="#444444", va="top")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    build("figures/fig_overview.pdf"); build(SCR + "fig_overview.png")
