import numpy as np

from hlens.experiments.h10_control import analyze, match, match_quality, pair_dbar
from hlens.experiments.register_persona import LAYERS, OFFSETS


def test_match_is_nearest_neighbour_without_replacement_and_deterministic():
    persona = {("a", "b"): 0.9, ("c", "d"): 0.5}
    cand = {("x", "y"): 0.88, ("z", "w"): 0.52, ("u", "v"): 0.51}
    rows = match(persona, cand)
    assert [r["control"] for r in rows] == [("x", "y"), ("u", "v")] or [r["control"] for r in rows] == [("x", "y"), ("z", "w")]
    assert len({r["control"] for r in rows}) == 2
    assert rows == match(persona, cand)  # deterministic
    q = match_quality(rows)
    assert q["adequate"] and q["median_rel"] <= 0.1
    bad = match_quality(match({("a", "b"): 0.9}, {("x", "y"): 0.3}))
    assert not bad["adequate"]


def _trial(pair, gain, seed):
    r = np.random.default_rng(seed)
    cells = {f"{l}|{o}": {"lambda": 1.0 + (gain if o == 0.0 else 0.0) + 0.01 * r.random()} for l in LAYERS for o in OFFSETS}
    return {"A": pair[0], "B": pair[1], "cells": cells, "width": {str(m): 0.9 - 0.02 * m for m in range(29)}}


def test_pair_dbar_and_analyze_outcomes():
    mk = lambda pairs, gain: [_trial(p, gain, i * 7 + j) for i, p in enumerate(pairs) for j in range(6)]
    pp = [(f"p{i}", f"q{i}") for i in range(10)]
    rp = [(f"r{i}", f"s{i}") for i in range(10)]
    cp = [(f"c{i}", f"e{i}") for i in range(10)]
    ext = {"persona": {"trials": mk(pp, 0.5)}, "register": {"trials": mk(rp, 0.0)}}
    dist = {"persona": {f"{a}|{b}": 0.8 for a, b in pp}, "register": {f"{a}|{b}": 0.2 for a, b in rp}, "control": {f"{a}|{b}": 0.8 for a, b in cp}}
    assert abs(list(pair_dbar(mk(pp[:1], 0.5)).values())[0] - 0.5) < 0.05
    same = analyze(ext, mk(cp, 0.5), cp, dist)
    assert same["control_signature"]["reproduces_signature"] and same["outcome"] == "control reproduces the signature; persona not larger"
    flat = analyze(ext, mk(cp, 0.0), cp, dist)
    assert not flat["control_signature"]["reproduces_signature"] and flat["primary_persona_gt_control"]["passed"]
    assert flat["outcome"] == "control does not reproduce the signature; persona larger"


def test_scaled_pair_sampling_and_analysis():
    from hlens.experiments.h10_scaled import analyze as analyze_scaled
    from hlens.experiments.h10_scaled import sample_pairs

    pool = [f"w{i}" for i in range(30)]
    a, b = sample_pairs(pool, 20, max_use=3, seed=0), sample_pairs(pool, 20, max_use=3, seed=0)
    assert a == b and len(set(a)) == 20
    use = {}
    for x, y in a:
        assert x != y
        use[x] = use.get(x, 0) + 1
        use[y] = use.get(y, 0) + 1
    assert max(use.values()) <= 3

    rng = np.random.default_rng(0)
    n = 30
    pp = [(f"p{i}", f"q{i}") for i in range(n)]
    sp = [(f"s{i}", f"t{i}") for i in range(n)]
    cp = [(f"c{i}", f"e{i}") for i in range(n)]
    dd = rng.uniform(0.5, 1.2, n)

    def mk(pairs, d_vals, slope):
        trials = []
        for p, dv in zip(pairs, d_vals):
            for j in range(2):
                r = np.random.default_rng(int(dv * 1000) + j)
                cells = {f"{l}|{o}": {"lambda": 1.0 + (slope * dv if o == 0.0 else 0.0) + 0.01 * r.random()} for l in LAYERS for o in OFFSETS}
                trials.append({"A": p[0], "B": p[1], "cells": cells, "width": {str(m): 0.9 - 0.02 * m for m in range(29)}})
        return {"trials": trials, "pairs": pairs, "n_planned": 2 * len(pairs)}

    fams = {"persona": mk(pp, dd, 1.0), "synonym": mk(sp, dd * 0.6, 1.0), "control": mk(cp, dd, 1.0)}
    dist = {"persona": {f"{a}|{b}": float(v) for (a, b), v in zip(pp, dd)}, "synonym": {f"{a}|{b}": float(v) * 0.6 for (a, b), v in zip(sp, dd)},
            "control": {f"{a}|{b}": float(v) for (a, b), v in zip(cp, dd)}}
    matching = [{"persona": list(p), "control": list(c)} for p, c in zip(pp, cp)]
    out = analyze_scaled(fams, dist, matching, n_boot=200)
    # persona and control have identical response to distance: no persona effect, equivalent, and distance explains
    assert not out["persona_effect_beyond_distance"] and out["regression"]["persona_equivalent_to_control"] and out["distance_explains"]
