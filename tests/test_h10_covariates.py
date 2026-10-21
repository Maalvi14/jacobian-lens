import numpy as np

from hlens.experiments.h10_covariates import COVARIATES, FAMILIES, analyze


def test_analysis_recovers_a_planted_covariate_and_explains_the_family_gap():
    rng = np.random.default_rng(0)
    cov, dbar = {}, {}
    for f, shift in zip(FAMILIES, (0.0, 0.5, 0.0, 1.0)):  # persona, synonym, control, countries; the gap is carried by `js_output`
        cov[f], dbar[f] = {}, {}
        for i in range(40):
            c = {k: float(rng.normal()) for k in COVARIATES}
            c["js_output"] = float(rng.normal() + shift * 2)
            key = f"{f}{i}|x"
            cov[f][key] = c
            dbar[f][key] = 0.5 * c["js_output"] + 0.05 * rng.normal()
    out = analyze(cov, dbar, n_boot=100)
    assert out["spearman"]["js_output"]["significant"] and not out["spearman"]["word_len"]["significant"]
    assert out["r2"]["family_and_covariates"] > out["r2"]["family_only"]
    assert out["family_coef_vs_control"]["countries"]["shrinkage"] > 0.5  # the covariate explains the family gap
