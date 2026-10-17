"""Audit probe: reconstruct per-question monitor features and compute AUROCs.

Why this exists: the reported 'Fisher-Rao AUROC 0.844' cannot be traced in the
notebook's own feature list, so every candidate FR score needs recomputation
from token_traces.csv with explicit score directions on the held-out test split.
"""
import pandas as pd
import numpy as np
from scipy.stats import rankdata

DATA = "geodesic_study/"


def auroc(y, s):
    """AUROC for labels y (1 = wrong) scored by s (higher = predicted wrong)."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return np.nan
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def build_features(tr, pred):
    f = tr.groupby(["method", "question_id"]).agg(
        n_steps=("step", "count"),
        mean_H=("current_entropy", "mean"),
        max_H=("current_entropy", "max"),
        late_H=("current_entropy", lambda s: s.tail(max(1, len(s) // 4)).mean()),
        mean_nll=("nll", "mean"),
        gate_frac=("gate", "mean"),
        mean_next_H=("next_entropy", "mean"),
        mean_fr=("fr", "mean"),
        mean_dcent=("d_centroid", "mean"),
        n_look=("fr", "count"),
    ).reset_index()
    return f.merge(pred[["question_id", "method", "correct", "split"]],
                   on=["method", "question_id"])


def main():
    tr = pd.read_csv(DATA + "token_traces.csv")
    pred = pd.read_csv(DATA + "predictions.csv", usecols=[
        "question_id", "method", "correct", "split"])
    f = build_features(tr, pred)
    f.to_csv("analysis/outputs/monitor_features.csv", index=False)

    t = f[f.split == "test"].copy()
    print("test questions per method:", t.groupby("method").size().to_dict())
    print("prevalence (fraction wrong) per method on test:")
    print((t.groupby("method").correct.agg(lambda s: 1 - s.mean())).round(4).to_string())

    print()
    print("=== AUROC (1=wrong as positive; score must point toward 'wrong') ===")
    heads = ["mean_nll", "mean_H", "max_H", "late_H", "gate_frac",
             "mean_next_H", "mean_fr", "mean_dcent", "n_FRsteps"]
    print(f"{'method':>14} | " + " ".join(f"{h:>9}" for h in heads))
    t["n_FRsteps"] = t["n_look"]
    for m in ["greedy", "nll_only", "fr_only", "entropy_only",
              "full_geodesic", "conservative"]:
        s = t[t.method == m]
        vals = [f"{auroc(s.correct.astype(float), s[h]):9.3f}"
                for h in heads]
        print(f"{m:>14} | " + " ".join(vals))

    # Notebook-faithful COMBINED monitor: percentile ranks of mean_H, max_H,
    # mean_next_H, gate_frac (NaN -> neutral 0.5), conf = 1 - mean(ranks),
    # then AUROC of -conf == AUROC of mean-rank risk. Recompute per method.
    print()
    print("=== COMBINED score (notebook-faithful), test split ===")
    def pct_risk(s):
        s = np.asarray(s, float)
        out = np.full(len(s), 0.5)
        m = ~np.isnan(s)
        if m.sum() > 1:
            r = rankdata(s[m])
            span = r.max() - r.min()
            out[m] = (r - 1) / span if span > 0 else 0.5
        return out

    for m in ["greedy", "nll_only", "fr_only", "entropy_only",
              "full_geodesic", "conservative"]:
        s = t[t.method == m]
        risk = np.mean([pct_risk(s.mean_H), pct_risk(s.max_H),
                        pct_risk(s.mean_next_H), pct_risk(s.gate_frac)], axis=0)
        print(f"{m:>14} | COMBINED AUROC = {auroc(s.correct.astype(float), risk):.3f}")

    # FR-based scores, both directions (is 0.844 reproducible anywhere?)
    print()
    print("=== FR score variants (AUROC, test) ===")
    for m in ["fr_only", "entropy_only", "full_geodesic", "conservative"]:
        s = t[t.method == m]
        has = s[s.n_look > 0]
        print(f"{m:>14} | questions with >=1 lookahead step: {len(has)}/{len(s)}")
        if len(has):
            au_hi = auroc(has.correct.astype(float), has.mean_fr)
            au_lo = auroc(has.correct.astype(float), -has.mean_fr)
            print(f"{'':>16} mean_fr AUROC (hi=wrong): {au_hi:.3f} | "
                  f"(hi=right): {au_lo:.3f}")


if __name__ == "__main__":
    main()
