"""Shared helpers for the UF check-default case: binning / WoE and model evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# ---------------------------------------------------------------- binning / WoE

def is_categorical(x):
    return x.dtype == object or str(x.dtype) in ("category", "str", "string")


def make_bins(x, q=10, edges=None):
    """Quantile bins with right-closed edges plus a separate "Missing" bin.

    A zero-inflated feature automatically gets its own "<= 0" bin. Pass `edges` to reuse bins fitted elsewhere.
    """
    if edges is None:
        inner = np.unique(x.dropna().quantile(np.linspace(0, 1, q + 1)[1:-1]).values)
        edges = np.r_[-np.inf, inner, np.inf]
    b = pd.cut(x, edges, right=True, include_lowest=True).astype(str)
    return b.where(x.notna(), "Missing"), edges


def to_bins(x, q=10, edges=None):
    """Bin labels for numeric or categorical x (categorical: one bin per level)."""
    if is_categorical(x):
        return x.astype(str).where(x.notna(), "Missing"), None
    return make_bins(x, q, edges)


def _bin_sort_key(label):
    # Interval labels like "(1.5, 3.0]" in numeric order, categories alphabetically, "Missing" last.
    if label == "Missing":
        return (2, 0.0, "")
    if label[:1] in "([":
        return (0, float(label[1:].split(",")[0]), "")
    return (1, 0.0, label)


def woe_table(x, y, q=10, edges=None, eps=0.5):
    """Per-bin counts, default rate, WoE = ln(%goods / %bads) and IV contribution."""
    b, edges = to_bins(x, q, edges)
    t = pd.DataFrame({"bin": b, "y": y.values}).groupby("bin")["y"].agg(n="size", bads="sum")
    t = t.loc[sorted(t.index, key=_bin_sort_key)]
    t["goods"] = t["n"] - t["bads"]
    t["default_rate"] = t["bads"] / t["n"]
    pg = (t["goods"] + eps) / (t["goods"].sum() + eps * len(t))
    pb = (t["bads"] + eps) / (t["bads"].sum() + eps * len(t))
    t["woe"] = np.log(pg / pb)
    t["iv"] = (pg - pb) * t["woe"]
    return t, edges


class WoEEncoder:
    """Fits WoE bins on training data and maps any data onto them (unseen bins -> WoE 0 = average risk)."""

    def __init__(self, q=5, custom_edges=None):
        self.q = q
        self.custom_edges = custom_edges or {}
        self.tables, self.edges = {}, {}

    def fit(self, X, y):
        for c in X.columns:
            self.tables[c], self.edges[c] = woe_table(X[c], y, self.q, self.custom_edges.get(c))
        return self

    def bins(self, X):
        return pd.DataFrame({c: to_bins(X[c], self.q, self.edges[c])[0] for c in X.columns}, index=X.index)

    def transform(self, X):
        b = self.bins(X)
        return pd.DataFrame({c: b[c].map(self.tables[c]["woe"]).astype(float).fillna(0.0) for c in X.columns},
                            index=X.index)

    def iv(self):
        return pd.Series({c: t["iv"].sum() for c, t in self.tables.items()}).sort_values(ascending=False)


# ---------------------------------------------------------------- evaluation

def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p):
    """Max distance between the cumulative score distributions of defaults and non-defaults."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    y = y[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def metrics(y, p):
    return {"gini": gini(y, p), "ks": ks(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "mean_pd": float(np.mean(p)), "default_rate": float(np.mean(y))}


def bootstrap_ci(y, preds, n_boot=500, seed=0, alpha=0.05):
    """Paired bootstrap of Gini for several models scored on the same rows.

    `preds` is {name: scores}. Returns point estimate and CI per model, plus the difference of every model vs the
    first one (same resamples, so the difference CI accounts for the models being correlated).
    """
    y = np.asarray(y)
    names = list(preds)
    P = {k: np.asarray(v) for k, v in preds.items()}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in names}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for k in names:
            draws[k].append(gini(y[idx], P[k][idx]))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = []
    for k in names:
        d = np.array(draws[k])
        rows.append({"model": k, "gini": gini(y, P[k]), "ci_low": np.percentile(d, lo), "ci_high": np.percentile(d, hi)})
        if k != names[0]:
            diff = d - np.array(draws[names[0]])
            rows.append({"model": f"{k} - {names[0]}", "gini": gini(y, P[k]) - gini(y, P[names[0]]),
                         "ci_low": np.percentile(diff, lo), "ci_high": np.percentile(diff, hi)})
    return pd.DataFrame(rows).set_index("model")


def calibration_table(y, p, n_bins=10):
    """Mean predicted PD vs actual default rate per predicted-PD decile."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    return d.groupby("bin").agg(mean_pd=("p", "mean"), default_rate=("y", "mean"), n=("y", "size"))


def cutoff_table(y, p, amount, decline_rates=(0.05, 0.10, 0.15, 0.20, 0.30)):
    """Business view: decline the riskiest X% of checks -> what happens to approvals, defaults and defaulted amount."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "amt": np.asarray(amount)}).sort_values("p", ascending=False)
    n, bads, bad_amt = len(d), d["y"].sum(), (d["y"] * d["amt"]).sum()
    rows = []
    for r in decline_rates:
        k = int(round(r * n))
        dec, acc = d.iloc[:k], d.iloc[k:]
        rows.append({
            "decline_%": r * 100,
            "pd_cutoff": dec["p"].min(),
            "defaults_avoided_%": dec["y"].sum() / bads * 100,
            "defaulted_amount_avoided_%": (dec["y"] * dec["amt"]).sum() / bad_amt * 100,
            "default_rate_declined_%": dec["y"].mean() * 100,
            "default_rate_approved_%": acc["y"].mean() * 100,
        })
    out = pd.DataFrame(rows).set_index("decline_%")
    out.attrs["baseline_default_rate_%"] = d["y"].mean() * 100
    return out
