"""Builds the effect-size table (appendix) from the results JSONs: estimate and 95% interval per model and test."""
import json, sys
import numpy as np
from scipy import stats
sys.path.insert(0, ".")
R = "../results/"
rng = np.random.default_rng(0)


def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


def boot_auc(score, y, n=2000):
    score, y = np.asarray(score, float), np.asarray(y, int)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append(auc(score[i], y[i]))
    return auc(score, y), np.percentile(out, [2.5, 97.5])


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


rows = []
f3 = lambda x: f"{x:.2f}"
iv = lambda a, b: f"[{a:.2f}, {b:.2f}]"
# B'
for name, f, last in (("Qwen3.5-4B", "h2_replication/B2_real_qwen3.5-4b.json", 30), ("Gemma 3 12B", "h2_replication/B2_real_gemma-3-12b.json", 44), ("Llama 3.1 8B", "h2_replication/B3_real_llama3.1-8b.json", 30)):
    r = json.load(open(R + f))["result"][str(last)]["ratio_to_random"]["j_lens"]["kappa"]
    rows.append((name, f"B$'$: $\\kappa$ ratio at L{last}", f3(r["median"]), iv(*r["ci95"]), "prompt-cluster bootstrap"))
# swaps
for name, f, a in (("Qwen3.5-4B", "swap_heldout_foldcase_qwen3.5-4b.json", "1.25"), ("Gemma 3 12B", "swap_heldout_foldcase_gemma-3-12b.json", "1.0")):
    s = json.load(open(R + f))["summary"][a]
    rows.append((name, f"swap success, $\\alpha={a}$ (held-out)", f"{s['rate']:.2f} ({s['hits']}/{s['n']})", iv(*s["wilson95"]), "Wilson, items"))
# E1
for name, f, lab in (("Qwen3.5-4B (dev)", "e1_engagement_qwen3.5-4b.json", ""), ("Qwen3.5-4B (HO)", "e1_engagement_heldout_qwen3.5-4b.json", ""), ("Gemma 3 12B (HO)", "e1_gemma_heldout.json", "")):
    it = json.load(open(R + f))["items"]
    y = [1 if r["hit"] else 0 for r in it]
    for key, nm in (("N", "AUC of $N$ (success)"), ("FO_ref", "AUC of $FO$ (success)")):
        a, ci = boot_auc([r["band"][key] for r in it], y)
        rows.append((name, nm, f3(a), iv(*ci), "bootstrap, items"))
# H1
for name, f, k in (("Qwen3.5-4B (dev)", "trust_score_qwen3.5-4b.json", "hit_a1.25"), ("Qwen3.5-4B (HO, $\\alpha=1.0$)", "trust_score_heldout_qwen3.5-4b.json", "hit_a1"), ("Qwen3.5-4B (HO, $\\alpha=1.25$)", "trust_score_heldout_qwen3.5-4b.json", "hit_a1.25"), ("Gemma 3 12B (HO)", "trust_score_heldout_gemma-3-12b.json", "hit_a1")):
    a = json.load(open(R + f))["analysis"][k]["auc"]["tau_L16" if "gemma" not in f else "tau_L24"] if False else None
import glob
for name, f, k, key in (("Qwen3.5-4B (dev)", "trust_score_qwen3.5-4b.json", "hit_a1.25", "tau_L16"), ("Qwen3.5-4B (HO, $\\alpha=1.0$)", "trust_score_heldout_qwen3.5-4b.json", "hit_a1", "tau_L16"), ("Qwen3.5-4B (HO, $\\alpha=1.25$)", "trust_score_heldout_qwen3.5-4b.json", "hit_a1.25", "tau_L16")):
    a = json.load(open(R + f))["analysis"][k]["auc"][key]
    rows.append((name, "AUC of $\\tau$ (failure)", f3(a["auc"]), iv(*a["ci95"]), "bootstrap, items"))
d = json.load(open(R + "trust_score_heldout_gemma-3-12b.json"))["analysis"]["hit_a1"]["auc"]
kk = [k for k in d if k.startswith("tau_L24")][0]
rows.append(("Gemma 3 12B (HO)", "AUC of $\\tau$ (failure)", f3(d[kk]["auc"]), iv(*d[kk]["ci95"]), "bootstrap, items"))
# claim A
for name, f in (("Qwen3.5-4B", "qwen3.5-4b"), ("Gemma 3 12B", "gemma-3-12b"), ("Qwen3.6-27B", "qwen3.6-27b")):
    for s in ("O", "HO"):
        p = R + f"h9_replication/{s}_{f}_clean" + ("_mixed" if "27b" in f else "") + ".json"
        a = json.load(open(p))["analysis"]["all"]
        fr = a["exploratory"]["frac_both_cross_terms_negative_at_threshold"]
        n = a.get("n_trials") or a["exploratory"].get("n") or None
        n = n or (24 if s == "O" else 120)
        lo, hi = wilson(round(fr * n), n)
        rows.append((name, f"both cross-terms negative ({s}, n={n})", f3(fr), iv(lo, hi), "Wilson, trials"))
# H10
from fig_claims import pair_dbar
sc = json.load(open(R + "h10_scaled_qwen3.5-4b.json"))
for k, lab in (("persona", "persona"), ("control", "object-noun control"), ("countries", "countries"), ("synonym", "near-synonym")):
    v = np.array(list(pair_dbar(sc["families"][k]["trials"]).values()))
    bs = [np.median(rng.choice(v, len(v))) for _ in range(3000)]
    rows.append(("Qwen3.5-4B", f"H10 median $\\bar D$, {lab} ({len(v)} pairs)", f3(np.median(v)), iv(*np.percentile(bs, [2.5, 97.5])), "bootstrap, pairs"))
with open("effect_table_rows.tex", "w") as fh:
    for r in rows:
        fh.write(" & ".join(r) + " \\\\\n")
print(len(rows)); print(open("effect_table_rows.tex").read())
