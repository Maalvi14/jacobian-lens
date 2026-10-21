import numpy as np

from hlens.experiments.register_persona import LAYERS, analyze_h10, family_signature


def _trials(n, center_boost, shrink=True, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        cells = {f"{l}|{o}": {"lambda": float(1.0 + (center_boost if o == 0.0 and l in (20, 24) else 0.0) + 0.05 * rng.normal()), "tau": 0.1, "first_order": 1.0}
                 for l in LAYERS for o in (-0.2, 0.0, 0.2)}
        out.append({"alpha_star": 0.5, "width": {str(m): (0.8 - 0.02 * m if shrink else 0.5 + 0.01 * m) for m in range(29)}, "cells": cells})
    return out


def test_family_signature_and_decision():
    good = family_signature(_trials(24, 1.0), 24)
    assert good["reproduces_signature"] and good["a_pass"] and good["b_pass"] and good["c_pass"]
    assert not family_signature(_trials(24, 0.0), 24)["c_pass"]  # no center effect
    assert not family_signature(_trials(24, 1.0, shrink=False), 24)["b_pass"]  # widths do not shrink
    assert not family_signature(_trials(10, 1.0), 24)["a_pass"]  # too few trials with a threshold
    res = {"register": {"trials": _trials(24, 1.0), "n_planned": 24}, "persona": {"trials": _trials(24, 0.0, seed=1), "n_planned": 24}}
    out = analyze_h10(res, {"trials": _trials(24, 1.0, seed=2)})
    assert not out["H10_proxy_supported"] and out["register"]["reproduces_signature"]


def test_h10_extended_contrast_and_dissociation_rule():
    import numpy as np

    from hlens.experiments.h10_extended import analyze
    from hlens.experiments.register_persona import LAYERS, OFFSETS

    def trial(center_gain, seed):
        r = np.random.default_rng(seed)
        cells = {f"{l}|{o}": {"lambda": 1.0 + (center_gain if o == 0.0 else 0.0) + 0.01 * r.random()} for l in LAYERS for o in OFFSETS}
        width = {str(m): (0.9 - 0.02 * m) for m in range(0, 29)}
        return {"cells": cells, "width": width}

    mk = lambda gain: {"pairs": [], "n_planned": 30, "trials": [trial(gain, i) for i in range(30)]}
    res = {"register": mk(0.0), "persona": mk(0.5)}
    out = analyze(res, {"trials": [trial(0.5, 100 + i) for i in range(10)]})
    assert out["persona"]["reproduces_signature"] and not out["register"]["reproduces_signature"]
    assert all(c["p_persona_gt_register"] < 0.01 for c in out["contrast"].values()) and out["dissociation_replicates"]
    out2 = analyze({"register": mk(0.5), "persona": mk(0.5)}, {"trials": [trial(0.5, 100 + i) for i in range(10)]})
    assert not out2["dissociation_replicates"] and out2["H10_proxy_supported"]
