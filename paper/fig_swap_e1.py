import json
import numpy as np
from figstyle import *

R = "../results/"
SCR = "/tmp/claude-1000/-home-mv-repos-jacobian-lens/8716fca1-8636-4929-bb78-d12970bfa47a/scratchpad/"


def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


def fig_swap(out):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"width_ratios": [1.0, 2.1], "wspace": 0.28})
    ax = axes[0]
    data = {m: json.load(open(R + f"swap_heldout_foldcase_{f}.json")) for m, f in (("Qwen3.5-4B", "qwen3.5-4b"), ("Gemma 3 12B", "gemma-3-12b"))}
    xs, labels = [], []
    k = 0
    for m, d in data.items():
        for a in ("1.25", "1.0"):
            s = d["summary"][a]
            lo, hi = s["wilson95"]
            ax.bar(k, s["rate"], width=0.62, color=MODEL_COLORS[m], alpha=1.0 if a == ("1.25" if m == "Qwen3.5-4B" else "1.0") else 0.45, lw=0, hatch=None if m == "Qwen3.5-4B" else "////", edgecolor="white")
            ax.errorbar(k, s["rate"], yerr=[[s["rate"] - lo], [hi - s["rate"]]], color="#333333", lw=0.8, capsize=1.5)
            ctrl = np.mean(s["random_control_hits"]) / s["n"]
            ax.bar(k + 0.36, ctrl, width=0.2, color=PALETTE["gray"], lw=0)
            ax.text(k, hi + 0.015, f"{s['hits']}/{s['n']}", ha="center", fontsize=7)
            xs.append(k); labels.append(a)
            k += 1
        k += 0.6
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7); ax.set_xlabel("swap strength α"); ax.set_xlim(-0.7, 4.8)
    ax.set_ylim(0, 0.8); ax.set_ylabel("swap success rate")
    ax.axhline(0.25, color="#999999", lw=0.5, ls=":")
    ax.text(0.5, 0.76, "Qwen3.5-4B", color=MODEL_COLORS["Qwen3.5-4B"], ha="center", fontsize=7, fontweight="bold")
    ax.text(3.7, 0.76, "Gemma 3 12B", color=MODEL_COLORS["Gemma 3 12B"], ha="center", fontsize=7, fontweight="bold")
    panel(ax, "a", "Held-out suite (165 items)")
    ax = axes[1]
    cats = ["city-capital", "city-language", "city-continent", "city-currency", "element-symbol", "element-state", "language-capital"]
    NS = {}
    for j, (m, a) in enumerate((("Qwen3.5-4B", "1.25"), ("Gemma 3 12B", "1.0"))):
        by = data[m]["summary"][a]["by_category"]
        rates, ns = [], []
        for c in cats:
            if c in by:
                h, n = map(int, by[c].split("/")); rates.append(h / n); ns.append(n)
            else:
                rates.append(np.nan); ns.append(0)
        x = np.arange(len(cats)) + (j - 0.5) * 0.38
        ax.bar(x, np.nan_to_num(rates), width=0.36, color=MODEL_COLORS[m], lw=0, label=f"{m} (α={a})", hatch=None if j == 0 else "////", edgecolor="white")
        NS[m] = ns
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels([c.replace("city-", "city→").replace("element-", "element→").replace("language-capital", "language→capital") + f" ({NS['Qwen3.5-4B'][i]}/{NS['Gemma 3 12B'][i]})" for i, c in enumerate(cats)], fontsize=7, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.0); ax.set_ylabel("swap success"); ax.legend(frameon=False, loc="upper right", handlelength=1.0)
    panel(ax, "b", "By relation (working strength of each model)")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def e1_items(path):
    return json.load(open(path))["items"]


def fig_e1(out):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.45), gridspec_kw={"width_ratios": [1.0, 1.0, 1.4], "wspace": 0.45})
    ax = axes[0]
    sets = [("Qwen\ndev", R + "e1_engagement_qwen3.5-4b.json", "Qwen3.5-4B"), ("Qwen\nHO", R + "e1_engagement_heldout_qwen3.5-4b.json", "Qwen3.5-4B"), ("Gemma\nHO", R + "e1_gemma_heldout.json", "Gemma 3 12B")]
    names = [("N", "nonlinear gain $N$", PALETTE["red"], "\\\\"), ("FO", "first-order $FO$", PALETTE["blue"], None), ("m", "realised margin (ceiling)", PALETTE["green"], "xx")]
    for i, (lab, path, model) in enumerate(sets):
        it = e1_items(path)
        y = [1 if r["hit"] else 0 for r in it]
        vals = {"N": auc([r["band"]["N"] for r in it], y), "FO": auc([r["band"]["FO_ref"] for r in it], y), "m": auc([r["band"]["m_ref"] for r in it], y)}
        for j, (key, nm, col, ht) in enumerate(names):
            ax.bar(i * 4 + j, vals[key], width=0.85, color=col, lw=0, label=nm if i == 0 else None, hatch=ht, edgecolor="white")
            ax.text(i * 4 + j, vals[key] + 0.012, f"{vals[key]:.2f}", ha="center", fontsize=7, rotation=90, va="bottom")
    ax.axhline(0.5, color="#555555", lw=0.6, ls="--")
    ax.set_xticks([1, 5, 9]); ax.set_xticklabels([s[0] for s in sets], fontsize=7)
    ax.set_ylim(0.2, 1.55); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("AUC for swap success")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, fontsize=7, ncol=1, borderaxespad=0.0, labelspacing=0.25)
    panel(ax, "a", "Predicting success")
    ax = axes[1]
    it = e1_items(R + "e1_gemma_heldout.json")
    fo = np.array([r["band"]["FO_ref"] for r in it]); m = np.array([r["band"]["m_ref"] for r in it]); h = np.array([r["hit"] for r in it])
    ax.scatter(fo[~h], m[~h], s=5, color=PALETTE["gray"], alpha=0.7, lw=0, label="no flip")
    ax.scatter(fo[h], m[h], s=6, color=MODEL_COLORS["Gemma 3 12B"], alpha=0.85, lw=0, label="flip")
    lim = (min(fo.min(), m.min()) - 3, max(fo.max(), m.max()) + 3)
    ax.plot(lim, lim, color="#333333", lw=0.7, ls="--"); ax.axhline(0, color="#999999", lw=0.5); ax.axvline(0, color="#999999", lw=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("first-order extrapolation $FO(1.0)$"); ax.set_ylabel("actual margin $m(1.0)$")
    ax.text(lim[1] * 0.5, lim[1] * 0.5 - 12, "$m = FO$", fontsize=7, ha="center", rotation=45, color="#333333")
    ax.legend(frameon=False, loc="upper left", handlelength=0.8, borderaxespad=0.2, markerscale=1.8)
    panel(ax, "b", "Gemma, held-out")
    ax = axes[2]
    groups = [("Qwen\ndev\nα=1.25", R + "trust_score_qwen3.5-4b.json", "hit_a1.25"), ("Qwen\nHO\nα=1.25", R + "trust_score_heldout_qwen3.5-4b.json", "hit_a1.25"), ("Qwen\nHO\nα=1.0", R + "trust_score_heldout_qwen3.5-4b.json", "hit_a1"), ("Gemma\nHO\nα=1.0", R + "trust_score_heldout_gemma-3-12b.json", "hit_a1")]
    series = [("tau_L16", "trust score $\\tau$", PALETTE["orange"], None), ("baseline_neg_loading_L16", "workspace loading", PALETTE["purple"], "\\\\"), ("baseline_neg_jdelta_mean", "small push, $-\\|J\\delta\\|$", PALETTE["blue"], "xx")]
    for i, (lab, path, key) in enumerate(groups):
        an = json.load(open(path))["analysis"][key]["auc"]
        for j, (sk, nm, col, ht) in enumerate(series):
            v = an[sk]["auc"]
            ax.bar(i * 4 + j, v, width=0.85, color=col, lw=0, label=nm if i == 0 else None, hatch=ht, edgecolor="white")
            ax.text(i * 4 + j, v + 0.012, f"{v:.2f}", ha="center", fontsize=7, rotation=90, va="bottom")
    ax.axhline(0.5, color="#555555", lw=0.6, ls="--")
    ax.set_xticks([1, 5, 9, 13]); ax.set_xticklabels([g[0] for g in groups], fontsize=7)
    ax.set_ylim(0.15, 1.4); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("AUC for swap failure")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, fontsize=7, ncol=1, borderaxespad=0.0, labelspacing=0.25)
    panel(ax, "c", "Predicting failure")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_swap("figures/fig_swap.pdf"); fig_swap(SCR + "fig_swap.png")
    fig_e1("figures/fig_e1.pdf"); fig_e1(SCR + "fig_e1.png")
