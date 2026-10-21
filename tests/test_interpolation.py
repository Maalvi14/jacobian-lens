import pytest
import torch

from hlens.data import load_ignition
from hlens.interpolation import lens_share_curve, make_setup


@pytest.mark.real_model
def test_ignition_share_curves(qwen):
    cfg = load_ignition("data/experiments/ignition.json")
    template = cfg.ctx_templates[0]
    alphas = [i / 20 for i in range(21)]
    layers = list(range(4, 31, 2))
    setup = make_setup(qwen, template, "France", "Germany")
    print(f"\ntemplate: {template!r}  W position {setup.position} of {setup.input_ids.shape[1]}", flush=True)
    curves = lens_share_curve(qwen, setup, alphas, layers)
    print("alpha: " + " ".join(f"{a:4.2f}" for a in alphas), flush=True)
    for l in layers:
        print(f"L{l:02d}:   " + " ".join(f"{s:4.2f}" for s in curves[l]), flush=True)
