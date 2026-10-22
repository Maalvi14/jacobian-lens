import json
import numpy as np
from figstyle import *

R = "../results/"
SCR = "/tmp/claude-1000/-home-mv-repos-jacobian-lens/8716fca1-8636-4929-bb78-d12970bfa47a/scratchpad/"


def fig_claim_a(out):
    setup()
    fig = plt.figure(figsize=(6.5, 3.55))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], width_ratios=[1, 1, 1, 1], wspace=0.55, hspace=0.75)
    ax = fig.add_subplot(gs[0, 0:2])
    runs = [("Qwen3.5-4B", "O", "O_qwen3.5-4b_clean"), ("Qwen3.5-4B", "HO", "HO_qwen3.5-4b_clean"), ("Gemma 3 12B", "O", "O_gemma-3-12b_clean"), ("Gemma 3 12B", "HO", "HO_gemma-3-12b_clean"),
            ("Qwen3.6-27B", "O", "O_qwen3.6-27b_clean_mixed"), ("Qwen3.6-27B", "HO", "HO_qwen3.6-27b_clean_mixed")]
    curves = {}
    for i, (m, s_, f) in enumerate(runs):
        d = json.load(open(R + f"h9_replication/{f}.json"))
        a = d["analysis"]["all"]
        frac = a["exploratory"]["frac_both_cross_terms_negative_at_threshold"]
        ok = a["precondition_met"]
        x = i + (i // 2) * 0.6
        ax.bar(x, frac, width=0.8, color=MODEL_COLORS[m], hatch=None if ok else "////", edgecolor="white", lw=0.0)
        ax.text(x, frac + 0.025, f"{frac:.2f}", ha="center", fontsize=7)
        curves.setdefault(m, {})[s_] = a["exploratory"]["median_I_by_offset"]
    xs = [i + (i // 2) * 0.6 for i in range(6)]
    ax.set_xticks(xs); ax.set_xticklabels(["O", "HO"] * 3, fontsize=7)
    for name, short, x in (("Qwen3.5-4B", "Qwen\n3.5-4B", 0.5), ("Gemma 3 12B", "Gemma\n3 12B", 3.1), ("Qwen3.6-27B", "Qwen\n3.6-27B", 5.7)):
        ax.text(x, -0.16, short, ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform(), color=MODEL_COLORS[name], fontweight="bold", linespacing=0.95)
    ax.set_ylim(0, 1.25); ax.set_ylabel("trials with both\ncross-terms negative")
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="#888888"), Patch(facecolor="#888888", hatch="////", edgecolor="white")], ["all three tests pass", "tests fail"], frameon=False, loc="upper right", fontsize=7, handlelength=1.2, ncol=2, columnspacing=0.8, borderaxespad=0.0)
    panel(ax, "a", "Inhibition at the threshold")
    ax = fig.add_subplot(gs[0, 2:4])
    for m in ("Qwen3.5-4B", "Gemma 3 12B", "Qwen3.6-27B"):
        v = curves[m]["HO"]
        offs = [-0.2, 0.0, 0.2]
        y = np.array([v[str(o)] for o in offs])
        y = y / (0.5 * (y[0] + y[2]))
        ax.plot(offs, y, marker=MODEL_MARKERS[m], ms=4, color=MODEL_COLORS[m], label=m)
    ax.axhline(1.0, color="#999999", lw=0.6, ls="--")
    ax.set_xticks([-0.2, 0.0, 0.2]); ax.set_xlabel("offset from threshold α*"); ax.set_ylabel("median inhibition\n(flank mean = 1)")
    ax.legend(frameon=False, loc="upper right", handlelength=1.4, fontsize=7, labelspacing=0.2, borderaxespad=0.0)
    panel(ax, "b", "Held-out set")
    for k, (model, title, lab) in enumerate((("qwen3.5-4b", "Qwen3.5-4B", "c"), ("gemma-3-12b", "Gemma 3 12B", "d"))):
        s_ = json.load(open(R + f"h9_scan/{model}.json"))
        srcs = sorted({v["layers"]["src"] for v in s_.values()}); gaps = sorted({v["layers"]["tgt"] - v["layers"]["src"] for v in s_.values()})
        for q, (dirs, dname) in enumerate((("lens", "J-lens"), ("identity", "identity"))):
            ax = fig.add_subplot(gs[1, 2 * k + q])
            M = np.full((len(srcs), len(gaps)), np.nan); P = np.zeros_like(M, dtype=bool)
            for v in s_.values():
                if v["layers"]["directions"] != dirs:
                    continue
                i, j = srcs.index(v["layers"]["src"]), gaps.index(v["layers"]["tgt"] - v["layers"]["src"])
                M[i, j] = v["frac_both_negative"]; P[i, j] = v["bonferroni_pass"]
            heat(ax, M, P, cmap="Blues", vmin=0, vmax=1.4, fs=7)
            ax.set_xticks(range(len(gaps))); ax.set_xticklabels([f"+{g}" for g in gaps]); ax.set_yticks(range(len(srcs))); ax.set_yticklabels(srcs if q == 0 else [])
            ax.set_xlabel("layers to target")
            if q == 0:
                ax.set_ylabel("source layer ℓ")
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)
            ax.set_title(f"{dname} directions", loc="center", pad=3, fontsize=7, color="#444444", fontweight="normal")
            if q == 1:
                p0 = fig.axes[-2].get_position(); p1 = ax.get_position()
                fig.text((p0.x0 + p1.x1) / 2, p1.y1 + 0.075, f"({lab}) {title}: scan over layer pairs", ha="center", va="bottom", fontsize=8, color="#666666", fontweight="bold")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def lam_profile(path, n_boot=2000, seed=0):
    d = json.load(open(path))
    layers = sorted({int(k.split("|")[0]) for k in d["trials"][0]["cells"]})
    rng = np.random.default_rng(seed); out = {}
    for l in layers:
        v = np.array([t["cells"][f"{l}|0.0"]["lambda"] for t in d["trials"]])
        bs = [np.median(rng.choice(v, len(v))) for _ in range(n_boot)]
        out[l] = (float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out


def fig_structure(out):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.15), gridspec_kw={"wspace": 0.5})
    ax = axes[0]
    for m, f, last in (("Qwen3.5-4B", "h8_motor_collapse_qwen3.5-4b.json", 31), ("Gemma 3 12B", "h8_motor_collapse_gemma-3-12b.json", 47)):
        d = json.load(open(R + f))
        for arm, ls, lab, mf in (("real", "-", "trained", None), ("null", "--", "randomized", "white")):
            L = sorted(int(l) for l in d[arm])
            y = [np.median(d[arm][str(l)]) / (last - l) for l in L]
            ax.plot([l / last for l in L], y, ls=ls, marker=MODEL_MARKERS[m], ms=3.2, mfc=mf or MODEL_COLORS[m], color=MODEL_COLORS[m], label=f"{m.split()[0]}, {'trained' if lab=='trained' else 'random'}")
    ax.set_yscale("log"); ax.set_ylim(3e-3, 3.0); ax.set_xlabel("depth (layer / last layer)"); ax.set_ylabel("per-block nonlinearity")
    band(ax, 0.75, 1.0)
    ax.legend(frameon=False, loc="upper left", fontsize=7, handlelength=1.8, labelspacing=0.2, borderaxespad=0.0)
    panel(ax, "a", "Nonlinearity")
    ax = axes[1]
    for m, f, n in (("Qwen3.5-4B", "h8_motor_collapse_qwen3.5-4b.json", 32), ("Gemma 3 12B", "h8_motor_collapse_gemma-3-12b.json", 48)):
        d = json.load(open(R + f))["d_I"]
        L = sorted(int(l) for l in d)
        ax.plot([l / (n - 1) for l in L], [d[str(l)] for l in L], color=MODEL_COLORS[m], label=m, ls="-" if m == "Qwen3.5-4B" else (0, (4, 1.5)))
    ax.axhline(1.0, color="#999999", lw=0.5, ls="--")
    ax.set_xlabel("depth (layer / last layer)"); ax.set_ylabel("distance of $J_\\ell$ to $cI$")
    ax.legend(frameon=False, loc="lower left", handlelength=1.8, labelspacing=0.2, borderaxespad=0.0)
    panel(ax, "b", "$J_\\ell \\to I$")
    ax = axes[2]
    for m, f, n in (("Qwen3.5-4B", "h6/real.json", 32), ("Gemma 3 12B", "h6_gemma-3-12b/real.json", 48)):
        prof = lam_profile(R + f)
        L = sorted(prof)
        x = [l / (n - 1) for l in L]
        ax.fill_between(x, [prof[l][1] for l in L], [prof[l][2] for l in L], color=MODEL_COLORS[m], alpha=0.22, lw=0)
        ax.plot(x, [prof[l][0] for l in L], marker=MODEL_MARKERS[m], ms=3.2, color=MODEL_COLORS[m], label=m)
    ax.set_yscale("log"); ax.set_xlabel("depth (layer / last layer)"); ax.set_ylabel("leading eigenvalue λ")
    ax.legend(frameon=False, loc="upper right", handlelength=1.8, labelspacing=0.2, borderaxespad=0.0)
    panel(ax, "c", "Curvature at ignition")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def pair_dbar(trials):
    by = {}
    for t in trials:
        d = [t["cells"][f"{l}|0.0"]["lambda"] - 0.5 * (t["cells"][f"{l}|-0.2"]["lambda"] + t["cells"][f"{l}|0.2"]["lambda"]) for l in (20, 24)]
        by.setdefault((t["A"], t["B"]), []).append(d)
    return {p: float(np.mean(np.median(np.array(v), axis=0))) for p, v in by.items()}


def fig_h10(out):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.55), gridspec_kw={"width_ratios": [1.3, 0.9, 1.0], "wspace": 0.7})
    sc = json.load(open(R + "h10_scaled_qwen3.5-4b.json")); P = json.load(open(R + "h10_scaled_pairs.json"))["distances"]
    fam = [("countries", "countries", PALETTE["orange"], "o"), ("control", "object-noun control", PALETTE["gray"], "s"), ("persona", "persona", PALETTE["purple"], "^"), ("synonym", "near-synonym", PALETTE["teal"], "D")]
    ax = axes[0]; box = []
    for key, lab, col, mk in fam:
        db = pair_dbar(sc["families"][key]["trials"])
        xs = np.array([P[key][f"{a}|{b}"] for (a, b) in db]); ys = np.array(list(db.values()))
        ax.scatter(xs, ys, s=7, color=col, alpha=0.6, lw=0, marker=mk, label=lab)
        box.append((lab, ys, col))
    ax.axhline(0, color="#888888", lw=0.5); ax.set_ylim(-1.0, 4.2)
    ax.set_xlabel("endpoint distance (layer 24)"); ax.set_ylabel("ignition effect $\\bar D$ (pair)")
    ax.legend(frameon=False, loc="upper left", handlelength=0.8, fontsize=7, markerscale=1.8, labelspacing=0.2, borderaxespad=0.0)
    panel(ax, "a", "100 pairs per family")
    ax = axes[1]
    bp = ax.boxplot([b[1] for b in box], vert=True, widths=0.55, showfliers=False, patch_artist=True, medianprops=dict(color="#222222", lw=0.9), whiskerprops=dict(color="#777777", lw=0.6), capprops=dict(color="#777777", lw=0.6))
    for patch, (_, _, col) in zip(bp["boxes"], box):
        patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(col)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(["countries", "object control", "persona", "near-synonym"], fontsize=7, rotation=40, ha="right", rotation_mode="anchor")
    ax.axhline(0, color="#888888", lw=0.5); ax.set_ylabel("ignition effect $\\bar D$"); ax.set_ylim(-0.6, 2.0)
    panel(ax, "b", "By family")
    ax = axes[2]
    cov = json.load(open(R + "h10_covariates_qwen3.5-4b.json"))["analysis"]["spearman"]
    names = {"slot_surprisal": "slot surprisal", "word_len": "word length", "emb_cos": "emb. cosine", "unemb_cos": "unemb. cosine", "d24": "distance (L24)", "d28": "distance (L28)", "js_output": "output divergence", "slot_logp_gap": "slot logp gap", "d16": "distance (L16)", "d8": "distance (L8)"}
    order = sorted(cov, key=lambda k: cov[k]["overall"])
    ax.barh(range(len(order)), [cov[k]["overall"] for k in order], color=[PALETTE["blue"] if cov[k]["significant"] else PALETTE["lightgray"] for k in order], height=0.65, lw=0, hatch=None)
    for yi, k in enumerate(order):
        if cov[k]["significant"]:
            v = cov[k]["overall"]; ax.text(v + (-0.012 if v < 0 else 0.012), yi, "*", ha="right" if v < 0 else "left", va="center", fontsize=9, color=PALETTE["blue"], fontweight="bold")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[k] for k in order], fontsize=7); ax.axvline(0, color="#666666", lw=0.5)
    ax.set_xlabel("Spearman with $\\bar D$ (366 pairs)", fontsize=7); ax.set_xlim(-0.33, 0.27); panel(ax, "c", "What explains $\\bar D$")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_claim_a("figures/fig_claim_a.pdf"); fig_claim_a(SCR + "fig_claim_a.png")
    fig_structure("figures/fig_structure.pdf"); fig_structure(SCR + "fig_structure.png")
    fig_h10("figures/fig_h10.pdf"); fig_h10(SCR + "fig_h10.png")
