import statistics

import pytest

from hlens.data import load_ignition
from hlens.experiments.stress_test import run_stress_test


@pytest.mark.real_model
def test_pathological_stress_test(qwen_fp32):
    cfg = load_ignition("data/experiments/ignition.json")
    trials = run_stress_test(qwen_fp32, cfg)
    print(f"\n{len(trials)} trials", flush=True)
    print("template | pair | alpha* || knee: |v| single m4 m16 rel16 valid(score) || control: |v| single m4 m16 rel16 valid(score)", flush=True)
    f = lambda p: f"{p.path_norm:5.2f} {p.single_err:6.2f} {p.multi4_err:6.2f} {p.multi16_err:6.2f} {p.reliability16:6.2f} {p.valid!s:5}({p.validity_score})"
    for t in trials:
        print(f"{t.template[:22]:22} | {t.word_a[:7]}-{t.word_b[:7]} | {t.alpha_star:.2f} || {f(t.knee)} || {f(t.control)}", flush=True)
    med = statistics.median
    print("median single-step err  knee/control:", med([t.knee.single_err for t in trials]), med([t.control.single_err for t in trials]))
    print("median multi16 err      knee/control:", med([t.knee.multi16_err for t in trials]), med([t.control.multi16_err for t in trials]))
    print("gate flags invalid      knee/control:", sum(not t.knee.valid for t in trials), sum(not t.control.valid for t in trials), "of", len(trials))

    # Phase-2 gate. Thresholds set from the observed run (15 trials) with margin:
    # knee single-step median 0.72 vs control 0.06; multi16 0.048; gate 15/15 vs 0/15.
    import json
    import os

    os.makedirs("results", exist_ok=True)
    with open("results/stress_test_qwen3.5-4b.json", "w") as fh:
        json.dump([{"template": t.template, "pair": [t.word_a, t.word_b], "alpha_star": t.alpha_star,
                    "knee": vars(t.knee), "control": vars(t.control)} for t in trials], fh)
    knee_single, ctrl_single = med([t.knee.single_err for t in trials]), med([t.control.single_err for t in trials])
    knee_multi = med([t.knee.multi16_err for t in trials])
    assert len(trials) >= 10
    assert knee_single > 0.3 and knee_single > 5 * ctrl_single, "single-step must visibly degrade at the knee"
    assert knee_multi < 0.15 and knee_multi < 0.5 * knee_single, "multi-step must recover"
    assert sum(not t.knee.valid for t in trials) >= 0.8 * len(trials), "the gate must flag knee paths"
    assert sum(not t.control.valid for t in trials) <= 0.2 * len(trials), "the gate must pass control paths"
    assert med([t.knee.reliability16 for t in trials]) > 3 * med([t.control.reliability16 for t in trials]), \
        "ms_hvp's own single-vs-multi divergence flags the knee without ground truth"
