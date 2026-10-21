"""H10 control (distinct non-persona nouns matched on endpoint distance), per hlens/docs/preregistration_h10_control.md.

Run:  .venv/bin/python -m hlens.experiments.h10_control distances   # endpoint distances + matching (no curvature)
      .venv/bin/python -m hlens.experiments.h10_control run         # control trials (checkpointed, resumable)
      .venv/bin/python -m hlens.experiments.h10_control analyze
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
from scipy import stats

from hlens.data import load_ignition
from hlens.experiments.register_persona import family_signature, run
from hlens.interpolation import forward_at_alpha, make_setup

PREREG = "hlens/docs/preregistration_h10_control.md"
DIST = "results/h10_control_distances.json"
OUT = "results/h10_control_qwen3.5-4b.json"
CKPT = "results/h10_control_qwen3.5-4b.partial.jsonl"
EXTENDED = "results/h10_extended_qwen3.5-4b.json"
CARRIERS, N_PAIRS, DIST_LAYER = slice(6, 12), 10, 24
CANDIDATES = [("anchor", "pencil"), ("river", "sandwich"), ("mountain", "violin"), ("candle", "hammer"), ("ladder", "pizza"), ("window", "banana"),
              ("bottle", "saddle"), ("blanket", "trumpet"), ("lantern", "carrot"), ("umbrella", "helmet"), ("bridge", "cookie"), ("pillow", "wrench"),
              ("basket", "rocket"), ("mirror", "pumpkin"), ("kettle", "compass"), ("garden", "telescope"), ("button", "volcano"), ("spoon", "castle"),
              ("feather", "engine"), ("clock", "melon"), ("shovel", "piano"), ("rope", "lemon"), ("bucket", "guitar"), ("needle", "tractor"),
              ("sofa", "cherry"), ("tent", "microscope"), ("stamp", "trousers"), ("magnet", "onion"), ("drum", "sock"), ("truck", "lamp"),
              ("tower", "biscuit"), ("anvil", "carpet")]


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def endpoint_distance(q, pair, templates, layer: int = DIST_LAYER) -> float:
    """Mean over carriers of ||h_A - h_B|| / mean(||h_A||, ||h_B||) at `layer` at the {W} position (pure-A vs pure-B input)."""
    out = []
    for t in templates:
        setup = make_setup(q, t, pair[0], pair[1])
        hb = forward_at_alpha(q, setup, 0.0, [layer])[layer][0, setup.position].float()
        ha = forward_at_alpha(q, setup, 1.0, [layer])[layer][0, setup.position].float()
        out.append(((ha - hb).norm() / (0.5 * (ha.norm() + hb.norm()))).item())
    return float(np.mean(out))


def match(persona_d: dict, cand_d: dict) -> list[dict]:
    """Deterministic nearest-neighbour matching without replacement; persona pairs in order of decreasing d."""
    left = dict(cand_d)
    rows = []
    for p, dp in sorted(persona_d.items(), key=lambda kv: -kv[1]):
        c = min(left, key=lambda k: (abs(left[k] - dp), k))
        rows.append({"persona": p, "control": c, "d_persona": dp, "d_control": left.pop(c)})
    return rows


def match_quality(rows: list[dict]) -> dict:
    dp = np.array([r["d_persona"] for r in rows])
    delta = np.abs(dp - np.array([r["d_control"] for r in rows]))
    med_rel, max_rel = float(np.median(delta) / np.median(dp)), float(delta.max() / np.median(dp))
    return {"median_abs_delta": float(np.median(delta)), "max_abs_delta": float(delta.max()), "median_rel": med_rel, "max_rel": max_rel,
            "adequate": bool(med_rel <= 0.10 and max_rel <= 0.25)}


def pair_dbar(trials: list[dict]) -> dict:
    """pair -> mean over L20 and L24 of the median over its trials of D = lambda(center) - mean lambda(+-0.2)."""
    by = {}
    for t in trials:
        d = [t["cells"][f"{l}|0.0"]["lambda"] - 0.5 * (t["cells"][f"{l}|-0.2"]["lambda"] + t["cells"][f"{l}|0.2"]["lambda"]) for l in (20, 24)]
        by.setdefault((t["A"], t["B"]), []).append(d)
    return {p: float(np.mean(np.median(np.array(v), axis=0))) for p, v in by.items()}


def analyze(extended: dict, control_trials: list[dict], control_pairs: list, dist: dict, *, n_boot: int = 2000, seed: int = 0) -> dict:
    pers, reg = pair_dbar(extended["persona"]["trials"]), pair_dbar(extended["register"]["trials"])
    ctrl = pair_dbar(control_trials)
    sig = family_signature(control_trials, 6 * len(control_pairs))
    sig.pop("_D", None)
    p_pc = float(stats.mannwhitneyu(list(pers.values()), list(ctrl.values()), alternative="greater").pvalue)
    p_cr = float(stats.mannwhitneyu(list(ctrl.values()), list(reg.values()), alternative="greater").pvalue)
    d_of = lambda fam, p: dist[fam][f"{p[0]}|{p[1]}"]
    allp = [(fam, p, v) for fam, dd in (("persona", pers), ("register", reg), ("control", ctrl)) for p, v in dd.items()]
    d_all, D_all = np.array([d_of(f, p) for f, p, _ in allp]), np.array([v for _, _, v in allp])
    rho = float(stats.spearmanr(d_all, D_all)[0])
    sub = [(d_of(f, p), 1.0 if f == "persona" else 0.0, v) for f, p, v in allp if f in ("persona", "control")]
    X = np.array([[1.0, d, ind] for d, ind, _ in sub])
    y = np.array([v for *_, v in sub])
    coef = lambda idx: np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][2]
    rng = np.random.default_rng(seed)
    boots = [coef(rng.integers(0, len(y), len(y))) for _ in range(n_boot)]
    out = {"control_signature": sig, "persona_dbar": {f"{a}-{b}": v for (a, b), v in pers.items()}, "register_dbar": {f"{a}-{b}": v for (a, b), v in reg.items()},
           "control_dbar": {f"{a}-{b}": v for (a, b), v in ctrl.items()},
           "primary_persona_gt_control": {"p": p_pc, "passed": bool(p_pc < 0.01), "median_persona": float(np.median(list(pers.values()))), "median_control": float(np.median(list(ctrl.values())))},
           "control_gt_register": {"p": p_cr, "median_register": float(np.median(list(reg.values())))},
           "spearman_d_vs_dbar_all_pairs": rho, "persona_indicator_coef": {"estimate": float(coef(np.arange(len(y)))), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}}
    out["outcome"] = ("control reproduces the signature; persona not larger" if sig["reproduces_signature"] and not out["primary_persona_gt_control"]["passed"] else
                      "control reproduces the signature; persona larger" if sig["reproduces_signature"] else
                      "control does not reproduce the signature; persona larger" if out["primary_persona_gt_control"]["passed"] else
                      "control does not reproduce the signature; persona not larger")
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "distances":
        import torch

        from hlens.loading import load_qwen

        q = load_qwen(dtype=torch.float32)
        cfg = load_ignition("data/experiments/ignition.json")
        templates = cfg.noun_ctx_templates[CARRIERS]
        ext = json.load(open(EXTENDED))["families"]
        ok = lambda w: len(q.tokenizer.encode(" " + w, add_special_tokens=False)) == 1
        dist = {"persona": {}, "register": {}, "candidate": {}}
        for fam in ("persona", "register"):
            for a, b in ext[fam]["pairs"]:
                dist[fam][f"{a}|{b}"] = endpoint_distance(q, (a, b), templates)
        for a, b in CANDIDATES:
            if ok(a) and ok(b):
                dist["candidate"][f"{a}|{b}"] = endpoint_distance(q, (a, b), templates)
        rows = match({tuple(k.split("|")): v for k, v in dist["persona"].items()}, {tuple(k.split("|")): v for k, v in dist["candidate"].items()})[:N_PAIRS]
        mq = match_quality(rows)
        dist["control"] = {f"{r['control'][0]}|{r['control'][1]}": r["d_control"] for r in rows}
        res = {"prereg_sha256": prereg_sha256(), "layer": DIST_LAYER, "distances": dist, "matching": rows, "quality": mq,
               "control_pairs": [list(r["control"]) for r in rows]}
        json.dump(res, open(DIST, "w"))
        print(json.dumps({"quality": mq, "persona_median_d": float(np.median(list(dist["persona"].values()))),
                          "register_median_d": float(np.median(list(dist["register"].values()))), "pairs": [(r["persona"], r["control"], round(r["d_persona"], 3), round(r["d_control"], 3)) for r in rows]}, indent=1))
    elif mode == "run":
        import torch

        from hlens.loading import load_qwen

        d = json.load(open(DIST))
        fam = run(load_qwen(dtype=torch.float32), families={"control": [tuple(p) for p in d["control_pairs"]]}, carriers=CARRIERS, n_pairs=N_PAIRS, checkpoint=CKPT)
        os.makedirs("results", exist_ok=True)
        json.dump({"prereg_sha256": prereg_sha256(), "families": fam}, open(OUT, "w"))
    else:
        d = json.load(open(DIST))
        res = json.load(open(OUT))
        out = analyze(json.load(open(EXTENDED))["families"], res["families"]["control"]["trials"], d["control_pairs"], d["distances"])
        out["matching_quality"] = d["quality"]
        out["prereg_sha256"] = {"recorded": res["prereg_sha256"], "distances_file": d["prereg_sha256"], "current": prereg_sha256()}
        res["analysis"] = out
        json.dump(res, open(OUT, "w"))
        print(json.dumps(out, indent=1))
