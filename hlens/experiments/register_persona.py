"""H10 (proxy): the H6 ignition protocol on register-contrast and persona noun pairs, per hlens/docs/preregistration_h10.md.

Run:  uv run python -m hlens.experiments.register_persona
      uv run python -m hlens.experiments.register_persona --analyze
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy import stats

from hlens.data import load_ignition
from hlens.experiments.ignition import run_trial

PREREG = "hlens/docs/preregistration_h10.md"
OUT = "results/h10_register_persona_qwen3.5-4b.json"
LAYERS, OFFSETS = [4, 12, 16, 20, 24], [-0.2, 0.0, 0.2]
FAMILIES = {
    "register": [("automobile", "car"), ("residence", "house"), ("physician", "doctor"), ("attire", "clothes"), ("beverage", "drink"), ("feline", "cat")],
    "persona": [("doctor", "pirate"), ("teacher", "soldier"), ("king", "beggar"), ("wizard", "farmer"), ("nurse", "pilot"), ("robot", "poet")],
}


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def family_pairs(tok, name: str, families: dict | None = None, n: int = 4) -> list[tuple[str, str]]:
    ok = lambda w: len(tok.encode(" " + w, add_special_tokens=False)) == 1
    return [p for p in (families or FAMILIES)[name] if ok(p[0]) and ok(p[1])][:n]


def run(q32, *, families: dict | None = None, carriers: slice = slice(0, 6), n_pairs: int = 4, clean: bool = False,
        checkpoint: str | None = None, log=print) -> dict:
    """The H10 proxy. Defaults are the original pre-registration (4 pairs per family, noun carriers 0-5); the extended
    pre-registration passes new pair lists, more pairs, other carriers and a checkpoint (.jsonl, resumable)."""
    cfg = load_ignition("data/experiments/ignition.json")
    families = families or FAMILIES
    done = {}
    if checkpoint and os.path.exists(checkpoint):
        for line in open(checkpoint):
            r = json.loads(line)
            done[(r["family"], r["template"], r["A"], r["B"])] = r
        log(f"  resuming: {len(done)} trials in {checkpoint}")
    out, t0 = {}, time.time()
    for fam in families:
        pairs = family_pairs(q32.tokenizer, fam, families, n_pairs)
        trials = []
        carrier_list = cfg.noun_ctx_templates[carriers]
        for t in carrier_list:
            for a, b in pairs:
                key = (fam, t, a, b)
                if key in done:
                    trials.append(done[key])
                    continue
                r = run_trial(q32, t, a, b, layers=LAYERS, offsets=OFFSETS, clean=clean)
                if r is not None:
                    trials.append(r)
                    if checkpoint:
                        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
                        with open(checkpoint, "a") as fh:
                            fh.write(json.dumps({**r, "family": fam}, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
                log(f"  {fam}: {len(trials)} trials, {time.time() - t0:.0f}s")
        out[fam] = {"pairs": pairs, "n_planned": len(carrier_list) * len(pairs), "trials": trials}
    return out


def _cell(t, l, o):
    return t["cells"][f"{l}|{o}"]["lambda"]


def family_signature(trials: list[dict], n_planned: int) -> dict:
    W = lambda t, m: t["width"].get(str(m), t["width"].get(m))
    a = len(trials) / n_planned
    both = [t for t in trials if W(t, 24) is not None and W(t, 8) is not None]
    b = float(np.mean([W(t, 24) < W(t, 8) for t in both])) if both else 0.0
    per_layer = {}
    for l in LAYERS:
        c = np.array([_cell(t, l, 0.0) for t in trials])
        f = np.array([0.5 * (_cell(t, l, -0.2) + _cell(t, l, 0.2)) for t in trials])
        d = c - f
        p = float(stats.wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
        per_layer[l] = {"frac_center_gt_flank": float(np.mean(c > f)), "p": p, "median_lambda": float(np.median(c)), "D": d.tolist()}
    c_ok = all(per_layer[l]["p"] < 0.01 for l in (20, 24))
    return {"a_threshold_fraction": a, "a_pass": bool(a >= 0.8), "b_width_shrinks_fraction": b, "b_pass": bool(b >= 0.8),
            "c_per_layer": {str(k): {x: v for x, v in d.items() if x != "D"} for k, d in per_layer.items()}, "c_pass": bool(c_ok),
            "reproduces_signature": bool(a >= 0.8 and b >= 0.8 and c_ok), "_D": {str(k): d["D"] for k, d in per_layer.items()},
            "median_width": {str(m): float(np.median([W(t, m) for t in trials if W(t, m) is not None])) for m in (0, 4, 8, 12, 16, 20, 24)}}


def analyze_h10(res: dict, countries: dict | None = None) -> dict:
    out = {f: family_signature(v["trials"], v["n_planned"]) for f, v in res.items()}
    if countries is not None:
        ref = [t for t in countries["trials"] if all(f"{l}|{o}" in t["cells"] for l in LAYERS for o in OFFSETS)]
        out["countries_reference"] = family_signature(ref, len(ref))
        for f in res:
            out[f]["mannwhitney_D_vs_countries_p"] = {
                str(l): float(stats.mannwhitneyu(out[f]["_D"][str(l)], out["countries_reference"]["_D"][str(l)]).pvalue) for l in LAYERS}
    for v in out.values():
        v.pop("_D", None)
    out["H10_proxy_supported"] = bool(out["register"]["reproduces_signature"] and out["persona"]["reproduces_signature"])
    return out


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        d = json.load(open(OUT))
        d["analysis"] = analyze_h10(d["families"], json.load(open("results/h6/real.json")))
        d["analysis"]["prereg_sha256"] = {"recorded": d["prereg_sha256"], "current": prereg_sha256()}
        json.dump(d, open(OUT, "w"))
        print(json.dumps(d["analysis"], indent=1))
        sys.exit(0)
    import torch

    from hlens.loading import load_qwen

    fam = run(load_qwen(dtype=torch.float32))
    os.makedirs("results", exist_ok=True)
    json.dump({"prereg_sha256": prereg_sha256(), "families": fam}, open(OUT, "w"))
