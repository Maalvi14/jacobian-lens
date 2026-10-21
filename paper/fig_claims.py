import json
import numpy as np
from figstyle import *

R = "../results/"
SCR = "/tmp/claude-1000/-home-mv-repos-jacobian-lens/8716fca1-8636-4929-bb78-d12970bfa47a/scratchpad/"


def fig_claim_a(out):
    setup()
    fig = plt.figure(figsize=(7.3, 2.25))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.0, 0.8, 0.8], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    runs = [("Qwen3.5-4B", "O", "O_qwen3.5-4b_clean"), ("Qwen3.5-4B", "HO", "HO_qwen3.5-4b_clean"), ("Gemma 3 12B", "O", "O_gemma-3-12b_clean"), ("Gemma 3 12B", "HO", "HO_gemma-3-12b_clean"),
            ("Qwen3.6-27B", "O", "O_qwen3.6-27b_clean_mixed"), ("Qwen3.6-27B", "HO", "HO_qwen3.6-27b_clean_mixed")]
    curves = {}
    for i, (m, s, f) in enumerate(runs):
        d = json.load(open(R + f"h9_replication/{f}.json"))
        a = d["analysis"]["all"]
        frac = a["exploratory"]["frac_both_cross_terms_negative_at_threshold"]
        ok = a["precondition_met"]
        x = i + (i // 2) * 0.5
        ax.bar(x, frac, width=0.8, color=MODEL_COLORS[m], hatch=None if ok else "////", edgecolor="white" if ok else MODEL_COLORS[m], lw=0.0 if ok else 0.4, alpha=1.0 if ok else 0.55)
        ax.text(x, frac + 0.02, f"{frac:.2f}", ha="center", fontsize=5.4)
        curves.setdefault(m, {})[s] = a["exploratory"]["median_I_by_offset"]
    xs = [i + (i // 2) * 0.5 for i in range(6)]
    ax.set_xticks(xs); ax.set_xticklabels(["O", "HO"] * 3, fontsize=5.6)
    for name, short, x in (("Qwen3.5-4B", "Qwen\n3.5-4B", 0.5), ("Gemma 3 12B", "Gemma\n3 12B", 3.0), ("Qwen3.6-27B", "Qwen\n3.6-27B", 5.5)):
        ax.text(x, -0.2, short, ha="center", va="top", fontsize=5.6, transform=ax.get_xaxis_transform(), color=MODEL_COLORS[name], fontweight="bold", linespacing=0.95)
    ax.set_ylim(0, 1.18); ax.set_ylabel("trials, both cross-terms negative")
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="#888888"), Patch(facecolor="#bbbbbb", hatch="////", edgecolor="#888888")], ["all three tests pass", "tests fail"], frameon=False, loc="upper right", fontsize=5.2, handlelength=1.0)
    panel(ax, "a", "Inhibition at threshold")
    ax = fig.add_subplot(gs[1])
    for m in ("Qwen3.5-4B", "Gemma 3 12B", "Qwen3.6-27B"):
        v = curves[m]["HO"]
        offs = [-0.2, 0.0, 0.2]
        y = np.array([v[str(o)] for o in offs])
        y = y / (0.5 * (y[0] + y[2]))
        ax.plot(offs, y, marker="o", ms=3, color=MODEL_COLORS[m], label=m)
    ax.axhline(1.0, color="#999999", lw=0.5, ls="--")
    ax.set_xlabel("offset from threshold α*"); ax.set_ylabel("median inhibition (flank = 1)")
    ax.legend(frameon=False, loc="lower center", handlelength=1.0, fontsize=5.4)
    panel(ax, "b", "Held-out set")
    for k, (model, title, lab) in enumerate((("qwen3.5-4b", "Qwen3.5-4B", "c"), ("gemma-3-12b", "Gemma 3 12B", "d"))):
        ax = fig.add_subplot(gs[2 + k])
        s = json.load(open(R + f"h9_scan/{model}.json"))
        srcs = sorted({v["layers"]["src"] for v in s.values()}); gaps = sorted({v["layers"]["tgt"] - v["layers"]["src"] for v in s.values()})
        M = np.full((len(srcs), len(gaps)), np.nan); P = np.zeros_like(M, dtype=bool)
        for v in s.values():
            if v["layers"]["directions"] != "lens":
                continue
            i, j = srcs.index(v["layers"]["src"]), gaps.index(v["layers"]["tgt"] - v["layers"]["src"])
            M[i, j] = v["frac_both_negative"]; P[i, j] = v["bonferroni_pass"]
        ax.imshow(M, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for i in range(len(srcs)):
            for j in range(len(gaps)):
                ax.text(j, i, f"{M[i, j]:.2f}" + ("*" if P[i, j] else ""), ha="center", va="center", fontsize=5.6, color="white" if M[i, j] > 0.6 else "#222222")
        ax.set_xticks(range(len(gaps))); ax.set_xticklabels([f"+{g}" for g in gaps]); ax.set_yticks(range(len(srcs))); ax.set_yticklabels(srcs)
        ax.set_xlabel("layers to target"); 
        if k == 0:
            ax.set_ylabel("source layer ℓ")
        for sp in ax.spines.values():
            sp.set_visible(False)
        panel(ax, lab, title + " scan")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_structure(out):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.1), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    for m, f, last in (("Qwen3.5-4B", "h8_motor_collapse_qwen3.5-4b.json", 31), ("Gemma 3 12B", "h8_motor_collapse_gemma-3-12b.json", 47)):
        d = json.load(open(R + f))
        for arm, ls, lab in (("real", "-", "trained"), ("null", "--", "randomized")):
            L = sorted(int(l) for l in d[arm])
            y = [np.median(d[arm][str(l)]) / (last - l) for l in L]
            ax.plot([l / last for l in L], y, ls=ls, marker="o", ms=2.2, color=MODEL_COLORS[m], label=f"{m}, {lab}" if True else None)
    ax.set_yscale("log"); ax.set_xlabel("depth (layer / last layer)"); ax.set_ylabel("per-block nonlinearity $\\tau_{map}/(L-\\ell)$")
    band(ax, 0.75, 1.0)
    ax.legend(frameon=False, loc="upper left", fontsize=5.0, handlelength=1.6)
    panel(ax, "a", "Per-block nonlinearity")
    ax = axes[1]
    for m, f, n in (("Qwen3.5-4B", "h8_motor_collapse_qwen3.5-4b.json", 32), ("Gemma 3 12B", "h8_motor_collapse_gemma-3-12b.json", 48)):
        d = json.load(open(R + f))["d_I"]
        L = sorted(int(l) for l in d)
        ax.plot([l / (n - 1) for l in L], [d[str(l)] for l in L], color=MODEL_COLORS[m], label=m)
    ax.axhline(1.0, color="#999999", lw=0.5, ls="--")
    ax.set_xlabel("depth (layer / last layer)"); ax.set_ylabel("distance of $J_\\ell$ to $cI$")
    ax.legend(frameon=False, loc="lower left", handlelength=1.2)
    panel(ax, "b", "$J_\\ell \\to I$")
    ax = axes[2]
    for m, f, n in (("Qwen3.5-4B", "h6/analysis.json", 32), ("Gemma 3 12B", "h6_gemma-3-12b/analysis.json", 48)):
        prof = json.load(open(R + f))["lambda"]["T1"]["Lambda_profile"]
        L = sorted(int(l) for l in prof)
        ax.plot([l / (n - 1) for l in L], [prof[str(l)] for l in L], marker="o", ms=2.6, color=MODEL_COLORS[m], label=m)
    ax.set_yscale("log"); ax.set_xlabel("depth (layer / last layer)"); ax.set_ylabel("leading eigenvalue λ at threshold")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    panel(ax, "c", "Ignition curvature")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def pair_dbar(trials):
    by = {}
    for t in trials:
        d = [t["cells"][f"{l}|0.0"]["lambda"] - 0.5 * (t["cells"][f"{l}|-0.2"]["lambda"] + t["cells"][f"{l}|0.2"]["lambda"]) for l in (20, 24)]
        by.setdefault((t["A"], t["B"]), []).append(d)
    return {p: float(np.mean(np.median(np.array(v), axis=0))) for p, v in by.items()}


def fig_h10(out):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.15), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0], "wspace": 0.62})
    sc = json.load(open(R + "h10_scaled_qwen3.5-4b.json")); P = json.load(open(R + "h10_scaled_pairs.json"))["distances"]
    fam = [("countries", "countries", PALETTE["orange"]), ("control", "object-noun control", PALETTE["gray"]), ("persona", "persona", PALETTE["purple"]), ("synonym", "near-synonym", PALETTE["teal"])]
    ax = axes[0]; box = []
    for key, lab, col in fam:
        db = pair_dbar(sc["families"][key]["trials"])
        xs = np.array([P[key][f"{a}|{b}"] for (a, b) in db]); ys = np.array(list(db.values()))
        ax.scatter(xs, ys, s=5, color=col, alpha=0.55, lw=0, label=lab)
        box.append((lab, ys, col))
    ax.axhline(0, color="#888888", lw=0.5); ax.set_ylim(-1.0, 3.2)
    ax.set_xlabel("endpoint distance (layer 24)"); ax.set_ylabel("ignition effect $\\bar D$ (pair)")
    ax.legend(frameon=False, loc="upper left", handlelength=0.8, fontsize=5.3, markerscale=1.6)
    panel(ax, "a", "100 pairs per family")
    ax = axes[1]
    bp = ax.boxplot([b[1] for b in box], vert=True, widths=0.55, showfliers=False, patch_artist=True, medianprops=dict(color="#222222", lw=0.9), whiskerprops=dict(color="#777777", lw=0.6), capprops=dict(color="#777777", lw=0.6))
    for patch, (_, _, col) in zip(bp["boxes"], box):
        patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(col)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(["coun-\ntries", "control", "per-\nsona", "near-\nsyn."], fontsize=5.6)
    ax.axhline(0, color="#888888", lw=0.5); ax.set_ylabel("ignition effect $\\bar D$"); ax.set_ylim(-0.6, 2.0)
    panel(ax, "b", "By family")
    ax = axes[2]
    cov = json.load(open(R + "h10_covariates_qwen3.5-4b.json"))["analysis"]["spearman"]
    names = {"slot_surprisal": "slot surprisal", "word_len": "word length", "emb_cos": "emb. cosine", "unemb_cos": "unemb. cosine", "d24": "distance (L24)", "d28": "distance (L28)", "js_output": "output divergence", "slot_logp_gap": "slot logp gap", "d16": "distance (L16)", "d8": "distance (L8)"}
    order = sorted(cov, key=lambda k: cov[k]["overall"])
    ax.barh(range(len(order)), [cov[k]["overall"] for k in order], color=[PALETTE["blue"] if cov[k]["significant"] else PALETTE["lightgray"] for k in order], height=0.65, lw=0)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[k] for k in order], fontsize=5.4); ax.axvline(0, color="#666666", lw=0.5)
    ax.set_xlabel("Spearman with $\\bar D$ (366 pairs)"); panel(ax, "c", "What explains $\\bar D$")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_claim_a("figures/fig_claim_a.pdf"); fig_claim_a(SCR + "fig_claim_a.png")
    fig_structure("figures/fig_structure.pdf"); fig_structure(SCR + "fig_structure.png")
    fig_h10("figures/fig_h10.pdf"); fig_h10(SCR + "fig_h10.png")
