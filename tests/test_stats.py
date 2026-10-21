import numpy as np
import pytest

from hlens.stats import auc, bootstrap_auc, logistic_loo_auc, permutation_pvalue, spearman


def test_auc_extremes_and_ties():
    assert auc([3, 4, 1, 2], [1, 1, 0, 0]) == 1.0
    assert auc([1, 2, 3, 4], [1, 1, 0, 0]) == 0.0
    assert auc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5  # all ties
    assert np.isnan(auc([1, 2], [1, 1]))
    # hand count: positives {0.8, 0.4}, negatives {0.5, 0.2, 0.9}: wins 0.8>0.5,0.8>0.2,0.4>0.2 = 3 of 6
    assert auc([0.8, 0.4, 0.5, 0.2, 0.9], [1, 1, 0, 0, 0]) == pytest.approx(0.5)


def test_bootstrap_and_permutation_behave_sensibly():
    rng = np.random.default_rng(0)
    y = np.array([1] * 30 + [0] * 30)
    informative = y + 0.5 * rng.standard_normal(60)
    noise = rng.standard_normal(60)
    lo, hi = bootstrap_auc(informative, y, n=300)
    assert 0.5 < lo < auc(informative, y) < hi <= 1.0
    assert permutation_pvalue(informative, y, n=500) < 0.01
    assert permutation_pvalue(noise, y, n=500) > 0.05


def test_logistic_loo_auc_needs_signal_and_does_not_reward_noise():
    rng = np.random.default_rng(1)
    y = np.array([1] * 40 + [0] * 40)
    signal = y + 0.7 * rng.standard_normal(80)
    noise = rng.standard_normal((80, 3))
    assert logistic_loo_auc(signal, y) > 0.75
    assert abs(logistic_loo_auc(noise, y) - 0.5) < 0.2
    # an uninformative extra feature must not help out of sample
    assert logistic_loo_auc(np.column_stack([signal, noise[:, 0]]), y) < logistic_loo_auc(signal, y) + 0.05


def test_spearman_monotone():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 50]) == pytest.approx(1.0)


def test_stratified_auc_removes_between_group_confounding():
    from hlens.stats import stratified_auc, stratified_permutation_pvalue

    rng = np.random.default_rng(0)
    # group B has high scores AND mostly positives; within each group the score is pure noise
    g = np.array(["a"] * 60 + ["b"] * 60)
    y = np.concatenate([(rng.random(60) < 0.2), (rng.random(60) < 0.8)]).astype(int)
    s = np.concatenate([rng.standard_normal(60), 5 + rng.standard_normal(60)])
    assert auc(s, y) > 0.7  # naive AUC is fooled by the groups
    assert abs(stratified_auc(s, y, g) - 0.5) < 0.12
    assert stratified_permutation_pvalue(s, y, g, n=300) > 0.05
    # a genuinely informative within-group signal survives stratification
    s2 = s + 2.0 * y
    assert stratified_auc(s2, y, g) > 0.8
