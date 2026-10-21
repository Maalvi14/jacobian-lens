import numpy as np

from hlens.experiments.h9_scan import GRID, configs, summarize_config


def _row(i, I, cA, cB, bump=0.01):
    cells = {str(o): {"cA": cA, "cB": cB, "I": I + bump * (1 if o == 0.0 else 0), "I_control": 0.0, "J_cross": 0.2} for o in (-0.2, 0.0, 0.2)}
    return {"template": f"t{i}", "A": "a", "B": "b", "alpha_star": 0.5, "cells": cells}


def test_configs_cover_the_grid_for_both_directions():
    c = configs("gemma-3-12b")
    assert len(c) == 2 * 3 * 3 and {x[0] for x in c} == {"lens", "identity"}
    assert all(t - s in GRID["gemma-3-12b"]["gaps"] for _, s, t in c)
    assert len(configs("qwen3.5-4b")) == 18


def test_summarize_config_bonferroni_flag():
    rows = [_row(i, 0.1 + 0.001 * i, -0.1, -0.1) for i in range(24)]
    s = summarize_config(rows, 18)
    assert s["uncorrected_pass"] and s["bonferroni_pass"] and s["frac_both_negative"] == 1.0
    flat = [_row(i, 0.01 * (-1) ** i, 0.01 * (-1) ** i, 0.01 * (-1) ** i, bump=0.0) for i in range(24)]
    f = summarize_config(flat, 18)
    assert not f["bonferroni_pass"]
