"""Small, dependency-free statistics for the experiment scripts (numpy + scipy only)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import optimize, stats


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """P(score of a random positive > score of a random negative), ties counted
    half. `labels` are 1 for the positive class. NaN if either class is empty."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_auc(scores, labels, *, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for the AUC (resampling items)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(s), len(s))
        a = auc(s[idx], y[idx])
        if not np.isnan(a):
            vals.append(a)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def permutation_pvalue(scores, labels, *, n: int = 5000, seed: int = 0) -> float:
    """One-sided p: how often a label permutation gives an AUC >= the observed one."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    obs = auc(s, y)
    rng = np.random.default_rng(seed)
    hits = sum(auc(s, rng.permutation(y)) >= obs for _ in range(n))
    return float((hits + 1) / (n + 1))


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    Xb = np.hstack([X, np.ones((len(X), 1))])

    def loss(w):
        z = Xb @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * np.sum(w[:-1] ** 2)

    def grad(w):
        p = 1 / (1 + np.exp(-(Xb @ w)))
        g = Xb.T @ (p - y)
        g[:-1] += l2 * w[:-1]
        return g

    return optimize.minimize(loss, np.zeros(Xb.shape[1]), jac=grad, method="L-BFGS-B").x


def logistic_loo_auc(X, labels, *, l2: float = 1.0) -> float:
    """Leave-one-out AUC of an L2-regularised logistic regression on standardised
    features: out-of-sample, so adding a feature has to earn its place."""
    X, y = np.asarray(X, float), np.asarray(labels, int)
    if X.ndim == 1:
        X = X[:, None]
    preds = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        w = _fit_logistic((X[tr] - mu) / sd, y[tr], l2)
        z = np.append((X[i] - mu) / sd, 1.0) @ w
        preds[i] = z
    return auc(preds, y)


def spearman(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)


def stratified_auc(scores, labels, groups) -> float:
    """Common AUC across strata: within-group AUCs averaged with weight n_pos * n_neg
    (the Mantel-Haenszel-style pooled estimate), so differences BETWEEN groups cannot
    produce the result. Groups lacking either class are skipped."""
    s, y, g = np.asarray(scores, float), np.asarray(labels, int), np.asarray(groups)
    num = den = 0.0
    for grp in np.unique(g):
        m = g == grp
        n1, n0 = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if n1 and n0:
            num += auc(s[m], y[m]) * n1 * n0
            den += n1 * n0
    return float(num / den) if den else float("nan")


def stratified_permutation_pvalue(scores, labels, groups, *, n: int = 5000, seed: int = 0, greater: bool = True) -> float:
    """p for the stratified AUC under label permutations WITHIN groups. `greater`
    selects the tail (observed >= null, or observed <= null)."""
    s, y, g = np.asarray(scores, float), np.asarray(labels, int), np.asarray(groups)
    obs = stratified_auc(s, y, g)
    rng = np.random.default_rng(seed)
    idx = [np.where(g == grp)[0] for grp in np.unique(g)]
    hits = 0
    for _ in range(n):
        yp = y.copy()
        for ii in idx:
            yp[ii] = rng.permutation(y[ii])
        v = stratified_auc(s, yp, g)
        hits += (v >= obs) if greater else (v <= obs)
    return float((hits + 1) / (n + 1))
