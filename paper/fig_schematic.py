import numpy as np
from figstyle import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

SCR = "/tmp/claude-1000/-home-mv-repos-jacobian-lens/8716fca1-8636-4929-bb78-d12970bfa47a/scratchpad/"


def box(ax, x, y, w, h, text, fc="#F3F3F3", ec="#888888", fs=6.2, tc="#333333", bold=False, lw=0.8):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.0, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


def build(out):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.05), gridspec_kw={"width_ratios": [1.5, 0.95, 1.2], "wspace": 0.08})
    ax = axes[0]; ax.set_xlim(0, 6); ax.set_ylim(0, 3.6); ax.axis("off")
    box(ax, 0.1, 1.3, 1.1, 0.9, "residual\n$h_\\ell$", fc="#E9F0F8", ec=PALETTE["blue"])
    box(ax, 1.9, 1.15, 2.1, 1.2, "layer slice $f$\nblocks $\\ell{+}1\\ldots m$", fc="#F3F3F3")
    box(ax, 4.7, 1.3, 1.2, 0.9, "readout\n$\\langle w, h_m\\rangle$", fc="#FBF3DC", ec=PALETTE["orange"])
    arrow(ax, 1.2, 1.75, 1.9, 1.75); arrow(ax, 4.0, 1.75, 4.7, 1.75)
    ax.text(1.55, 2.0, "$\\delta$", fontsize=7, ha="center", color="#333333")
    ax.annotate("", xy=(4.0, 2.95), xytext=(1.9, 2.95), arrowprops=dict(arrowstyle="-|>", color=PALETTE["blue"], lw=1.1))
    ax.text(2.95, 3.12, "first order:  $J\\,\\delta$  (what the J-lens reads)", fontsize=6.2, ha="center", color=PALETTE["blue"])
    ax.annotate("", xy=(4.0, 0.55), xytext=(1.9, 0.55), arrowprops=dict(arrowstyle="-|>", color=PALETTE["orange"], lw=1.1, connectionstyle="arc3,rad=-0.25"))
    ax.text(2.95, 0.12, "second order:  $\\frac{1}{2}\\delta^\\top \\mathrm{H}\\,\\delta$", fontsize=6.2, ha="center", color=PALETTE["orange"])
    ax.text(3.0, -0.35, "trust score  $\\tau=\\|\\frac{1}{2}\\delta^\\top \\mathrm{H}\\delta\\|\\;/\\;\\|J\\delta\\|$", fontsize=6.4, ha="center", color="#333333", fontweight="bold")
    ax.set_title("(a) A layer slice and a perturbation", loc="left", pad=3, fontsize=7.5, color="#666666", fontweight="bold")
    ax = axes[1]; ax.set_xlim(0, 4); ax.set_ylim(-0.8, 3.6); ax.axis("off")
    labels = ["$v_A$", "$v_B$", "rem."]
    for i in range(3):
        for j in range(3):
            fc = "#F7D9BE" if (i, j) in ((0, 1), (1, 0)) else ("#E9F0F8" if i == j else "#F7F7F7")
            ax.add_patch(Rectangle((0.9 + j * 0.85, 2.2 - i * 0.85), 0.8, 0.8, fc=fc, ec="#999999", lw=0.6))
    for k, l in enumerate(labels):
        ax.text(0.8, 2.6 - k * 0.85, l, ha="right", va="center", fontsize=6.4)
        ax.text(1.3 + k * 0.85, 3.12, l, ha="center", va="bottom", fontsize=6.4)
    ax.text(1.3 + 0.85, 2.6, "$c_{AB}$", ha="center", va="center", fontsize=6.4, color=PALETTE["orange"], fontweight="bold")
    ax.text(1.3, 2.6 - 0.85, "$c_{AB}$", ha="center", va="center", fontsize=6.4, color=PALETTE["orange"], fontweight="bold")
    ax.text(2.1, -0.15, "$H_{ij}=v_i^\\top \\mathrm{H}\\,v_j$", ha="center", fontsize=6.6, color="#333333")
    ax.text(2.1, -0.65, "off-diagonal $c_{AB}$: interaction\nbetween two concepts", ha="center", fontsize=5.4, color="#555555")
    ax.set_title("(b) Restricted Hessian", loc="left", pad=3, fontsize=7.5, color="#666666", fontweight="bold")
    ax = axes[2]; ax.set_xlim(0, 7); ax.set_ylim(0, 3.6); ax.axis("off")
    box(ax, 0.0, 2.3, 2.6, 0.95, "trained weights\n+ pre-fitted lens", fc="#E9F0F8", ec=PALETTE["blue"], fs=5.0)
    box(ax, 0.0, 0.65, 2.6, 0.95, "randomized weights\n+ re-fitted lens", fc="#F3F3F3", ec="#888888", fs=5.0)
    box(ax, 3.4, 1.45, 1.6, 0.95, "same\nmeasurement", fc="#FBF3DC", ec=PALETTE["orange"], fs=5.0)
    arrow(ax, 2.6, 2.75, 3.4, 2.15); arrow(ax, 2.6, 1.15, 3.4, 1.7)
    box(ax, 5.6, 1.45, 1.4, 0.95, "ratio vs\n[0.95, 1.05]", fc="#FFFFFF", ec="#888888", fs=5.0)
    arrow(ax, 5.0, 1.92, 5.6, 1.92)
    ax.text(3.5, 0.1, "no effect under randomization\n$\\Rightarrow$ needs learned structure", fontsize=5.4, ha="center", color="#555555")
    ax.set_title("(c) Weight-randomized control", loc="left", pad=3, fontsize=7.5, color="#666666", fontweight="bold")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    build("figures/fig_schematic.pdf"); build(SCR + "fig_schematic.png")
