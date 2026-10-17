"""Run-2 analysis — CORRECTED to match the pre-registered definitions in
experiment_protocol.md (run-2 addendum). Changelog vs run2_analysis_prefix.py
(the audited pre-fix version; both kept for provenance):

1. All per-question aggregates are computed over the FULL trajectory; only
   late_H uses the last-quarter window (pre-fix code filtered the whole frame
   to the last quarter before aggregating — a bug that redefined mean_H,
   max_H, mean_nll, mean_next_H, gate_frac and every FR aggregate).
2. mean_hgap now = mean over steps of [mean_{j>0} H(q_j) - H(q_0)] (prereg G5);
   pre-fix it was a duplicate of mean_next_H.
3. std_fr now = std of FR(q_j, q_greedy) over the j>0 candidate grid within a
   question (prereg G6); pre-fix it was the std of rank-0 rows (identically ~0).
4. mean_altspread now = mean over steps of (per-step max-min of FR(q_j,q_greedy),
   j>0) (prereg G4).
5. mean_prob now = mean of rank-0 p_value (pre-fix: mean p_gap01).
6. GEO confirmatory set = the 6 pre-registered features (G1-G6);
   max_frcons + GEO-only retained separately as EXPLORATORY.
7. sklearn cross-checks for AUROC/AUPRC; decision scores exported to
   test_scores.csv; risk-coverage at 80%/90% coverage exported; paired
   bootstrap p-values added. Seed 42 throughout.

Pre-registered definitions (experiment_protocol.md, run-2 addendum):
  CONV = mean_H, max_H, late_H, mean_nll, mean_next_H, mean_prob, gate_frac, log_len
  GEO  = G1 mean_frcons, G2 mean_altdist, G3 mean_altdist_max, G4 mean_altspread,
         G5 mean_hgap, G6 std_fr
  Pass criteria for H2 support: paired-delta AUPRC > 0 AND paired dAUROC CI
  excluding 0 AND risk-coverage gain at 80%/90% coverage (all pre-specified).
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
CONV = ["mean_H", "max_H", "late_H", "mean_nll", "mean_next_H", "mean_prob",
        "gate_frac", "log_len"]
GEO = ["mean_frcons", "mean_altdist", "mean_altdist_max", "mean_altspread",
       "mean_hgap", "std_fr"]                     # pre-registered G1-G6
GEO_EXPLORATORY = ["max_frcons"]                  # implementation extra, not prereg


def auroc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return np.nan
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def auprc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind="stable"); y = y[order]
    tp = np.cumsum(y); prec = tp / (np.arange(len(y)) + 1)
    rec = tp / max(y.sum(), 1)
    d = np.diff(np.concatenate([[0], rec]))
    return float((d * prec).sum())


def brier(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    c = expit(s)      # decision function -> calibrated-ish probability
    return float(np.mean((c - y) ** 2))


def boot_ci(metric, y, s, n_boot=2000, seed=SEED):
    """Single shared index per replicate (correct paired resampling)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float); s = np.asarray(s, float)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def boot_paired(y, s_a, s_b, metric=auroc, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, sa, sb = np.asarray(y, float), np.asarray(s_a, float), np.asarray(s_b, float)
    n = len(y)
    base = metric(y, sa) - metric(y, sb)
    disp = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = metric(y[idx], sa[idx]) - metric(y[idx], sb[idx])
        if not np.isnan(d):
            disp.append(d)
    disp = np.asarray(disp)
    p = 2 * min((disp <= 0).mean(), (disp >= 0).mean())
    return base, float(np.quantile(disp, .025)), float(np.quantile(disp, .975)), float(p)


# --------------------------------------------------- per-question features --
def build_question_table(indir):
    steps = pd.read_csv(f"{indir}/monitor_steps.csv")
    cands = pd.read_csv(f"{indir}/monitor_candidates.csv")
    ans = pd.read_csv(f"{indir}/monitor_answers.csv",
                      usecols=["question_id", "split", "gold", "correct",
                               "stopped_on", "seconds", "forward_calls",
                               "generated_tokens", "model"])

    g = cands[cands.is_greedy == 1][["question_id", "step", "p_value",
                                     "q_entropy", "fr_consensus"]]
    alt = cands[cands.is_greedy == 0]
    alt_step = alt.groupby(["question_id", "step"]).agg(
        alt_fr_mean=("fr_greedy", "mean"),
        alt_fr_max=("fr_greedy", "max"),
        alt_fr_min=("fr_greedy", "min"),
        alt_qent_mean=("q_entropy", "mean")).reset_index()
    std_fr = (alt.groupby("question_id").fr_greedy
              .std().rename("std_fr").reset_index())          # prereg G6

    s = (steps.merge(g, left_on=["question_id", "step"],
                     right_on=["question_id", "step"])
         .merge(alt_step, on=["question_id", "step"]))
    s["hh_gap"] = s.alt_qent_mean - s.q_entropy               # prereg G5 numerator
    s["spread"] = s.alt_fr_max - s.alt_fr_min                 # prereg G4 numerator

    n_steps = steps.groupby("question_id").step.max().rename("n_steps")
    n_steps = n_steps.reset_index()

    # full-trajectory aggregates (the FIX: no step filter here)
    agg = s.groupby("question_id").agg(
        mean_H=("h_p", "mean"),
        max_H=("h_p", "max"),
        mean_nll=("nll_greedy", "mean"),
        mean_next_H=("q_entropy", "mean"),
        mean_prob=("p_value", "mean"),
        tie_frac=("tie_flag", "mean"),
        gate_frac=("h_p", lambda x: (x > 1.0).mean()),
        mean_frcons=("fr_consensus", "mean"),                  # G1
        max_frcons=("fr_consensus", "max"),                    # exploratory extra
        mean_altdist=("alt_fr_mean", "mean"),                  # G2
        mean_altdist_max=("alt_fr_max", "mean"),               # G3
        mean_altspread=("spread", "mean"),                     # G4
        mean_hgap=("hh_gap", "mean"),                          # G5
    ).reset_index().merge(std_fr, on="question_id")

    # late window for late_H only (prereg)
    late = s[s.step >= (s.merge(n_steps, on="question_id").n_steps // 4 * 3)]
    late_H = late.groupby("question_id").h_p.mean().rename("late_H").reset_index()
    agg = agg.merge(late_H, on="question_id")
    agg = agg.merge(n_steps.rename(columns={"n_steps": "steps"}), on="question_id")
    agg["log_len"] = np.log(agg.steps.clip(1))
    agg = agg.merge(ans, on="question_id")
    agg["wrong"] = 1 - agg.correct
    return agg


def fit_score(dev, test, cols):
    med = {c: (dev[c].median() if np.isfinite(dev[c].median()) else 0.0)
           for c in cols}
    Xd = dev[cols].fillna(pd.Series(med)).values
    Xt = test[cols].fillna(pd.Series(med)).values
    mu, sd = Xd.mean(0), Xd.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=4000, random_state=SEED)
    lr.fit((Xd - mu) / sd, dev.wrong.values)
    return lr.decision_function((Xt - mu) / sd), lr, (mu, sd, med)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="kaggle_out/full/geodesic_study2")
    ap.add_argument("--outdir", default="analysis/outputs2")
    ap.add_argument("--figdir", default="analysis/figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    # merge part files if needed
    for stem in ["monitor_steps", "monitor_candidates", "monitor_answers"]:
        merged = f"{a.indir}/{stem}.csv"
        if not os.path.exists(merged):
            parts = sorted(p for p in os.listdir(a.indir)
                           if p.startswith(stem + "_part") and p.endswith(".csv"))
            pd.concat([pd.read_csv(f"{a.indir}/{p}") for p in parts],
                      ignore_index=True).to_csv(merged, index=False)

    tab = build_question_table(a.indir)
    tab.to_csv(f"{a.outdir}/question_features.csv", index=False)
    dev = tab[tab.split == "dev"]
    test = tab[tab.split == "test"]
    print("n dev/test:", len(dev), len(test),
          "| test prevalence wrong:", test.wrong.mean().round(4))

    # ---- sanity checks ---------------------------------------------------
    checks = []
    steps = pd.read_csv(f"{a.indir}/monitor_steps.csv")
    cands = pd.read_csv(f"{a.indir}/monitor_candidates.csv")
    ans = pd.read_csv(f"{a.indir}/monitor_answers.csv")
    joins = steps.groupby("question_id").size()
    _gen = ans.set_index("question_id").generated_tokens.reindex(joins.index)
    checks.append(("steps == generated_tokens", bool((joins == _gen).all())))
    sizes = cands.groupby(["question_id", "step"]).size()
    checks.append(("4 candidates per step", bool((sizes == 4).all())))
    mg = cands[cands.is_greedy == 1].set_index(["question_id", "step"])
    ch = steps.set_index(["question_id", "step"]).chosen_token
    checks.append(("greedy candidate == chosen token",
                   bool(mg.token_id.reindex(ch.index).eq(ch).all())))
    checks.append(("NaN count in candidate features = 0",
                   int(cands[["q_entropy", "fr_greedy", "fr_consensus"]]
                       .isna().sum().sum()) == 0))
    checks.append(("rank0 |fr_greedy| < 3e-3 (GPU fp32 floor)",
                   float(mg.fr_greedy.abs().max()) < 3e-3))
    checks.append(("ties recorded", int(steps.tie_flag.sum())))
    checks.append(("tie rows with p_gap01>0 (inconsistencies)",
                   int(((steps.tie_flag == 1) & (steps.p_gap01 > 0)).sum())))
    pd.DataFrame(checks, columns=["check", "result"]).to_csv(
        f"{a.outdir}/sanity_checks.csv", index=False)
    print(checks, "\n")

    # ---- monitor table (dev-fit / test-eval) ------------------------------
    rows = []
    models = ([("CONV", CONV)]
              + [(f"CONV+{c} (prereg G)", CONV + [c]) for c in GEO]
              + [("CONV+ALL_GEO(prereg)", CONV + GEO),
                 ("GEO only (prereg)", GEO)]
              + [(f"CONV+{c} (exploratory)", CONV + [c]) for c in GEO_EXPLORATORY])
    scores = {}
    for name, cols in models:
        p, lr, _ = fit_score(dev, test, cols)
        au_hand = auroc(test.wrong.values, p)
        au_sk = roc_auc_score(test.wrong.values, p)
        ap_hand = auprc(test.wrong.values, p)
        ap_sk = average_precision_score(test.wrong.values, p)
        lo, hi = boot_ci(auroc, test.wrong.values, p)
        rows.append({"model": name, "n_dev": len(dev), "n_test": len(test),
                     "test_auroc": au_sk, "auroc_hand_vs_sklearn_maxdiff":
                         abs(au_hand - au_sk),
                     "lo": lo, "hi": hi,
                     "test_auprc": ap_sk, "auprc_hand_vs_sklearn_maxdiff":
                         abs(ap_hand - ap_sk),
                     "test_brier": brier(test.wrong.values, p)})
        scores[name] = p
    mon = pd.DataFrame(rows)
    mon.round(6).to_csv(f"{a.outdir}/trained_monitors.csv", index=False)
    print(mon.round(3).to_string(index=False))

    score_export = pd.DataFrame({"question_id": test.question_id.values,
                                 "split": "test",
                                 "wrong": test.wrong.values})
    for name, p in scores.items():
        score_export[name] = p
    score_export.to_csv(f"{a.outdir}/test_scores.csv", index=False)

    # ---- paired deltas vs CONV --------------------------------------------
    base_p = scores["CONV"]
    drows = []
    for name, p in scores.items():
        if name == "CONV":
            continue
        d, lo, hi, pval = boot_paired(test.wrong.values, p, base_p)
        drows.append({"model": name, "d_auroc": d, "lo": lo, "hi": hi,
                      "p_two_sided": pval})
    deltas = pd.DataFrame(drows)
    deltas.round(6).to_csv(f"{a.outdir}/paired_deltas.csv", index=False)
    print("\npaired deltas vs CONV:\n", deltas.round(3).to_string(index=False))

    # ---- risk-coverage at fixed coverage (prereg criterion) ----------------
    rc_rows = []
    for name in ("CONV", "CONV+ALL_GEO(prereg)", "GEO only (prereg)"):
        p = scores[name]
        yw = test.wrong.values
        order = np.argsort(-p)  # riskiest first -> drop
        acc_ordered = 1 - yw[order]
        for cov in (0.80, 0.90):
            k = int(round(cov * len(yw)))
            kept = acc_ordered[-k:]  # retain the k LEAST-risky
            rc_rows.append({"model": name, "coverage": cov,
                            "acc_answered": kept.mean()})
    rc = pd.DataFrame(rc_rows)
    rc.to_csv(f"{a.outdir}/risk_coverage_crit.csv", index=False)
    print("\nrisk-coverage at fixed coverage:\n", rc.to_string(index=False))

    # ---- univariate --------------------------------------------------------
    u_rows = []
    for c in CONV + GEO + GEO_EXPLORATORY + ["tie_frac"]:
        sv = test[c].astype(float).values
        m = ~np.isnan(sv)
        if len(np.unique(test.wrong.values[m])) < 2:
            continue
        u_rows.append({"feature": c,
                       "test_auroc": roc_auc_score(test.wrong.values[m], sv[m]),
                       "dev_spearman_wrong": spearmanr(
                           dev[c].astype(float), dev.wrong).correlation})
    u = pd.DataFrame(u_rows).sort_values("test_auroc", ascending=False)
    u.round(4).to_csv(f"{a.outdir}/univariate_auroc.csv", index=False)
    print("\nunivariate:\n", u.round(3).to_string(index=False))

    dev[list(CONV + GEO + GEO_EXPLORATORY)].corr(method="spearman").round(3) \
        .to_csv(f"{a.outdir}/spearman_dev.csv")

    # ---- figures -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in ("CONV", "CONV+ALL_GEO(prereg)", "GEO only (prereg)"):
        p = scores[label]
        yw = test.wrong.values
        order = np.argsort(-p)
        cov = (np.arange(len(yw)) + 1) / len(yw)
        acc = (1 - yw[order]).cumsum() / (np.arange(len(yw)) + 1)
        ax.plot(100 * cov, 100 * acc, lw=2, label=label)
    ax.axvline(80, ls="--", c="k", lw=1)
    ax.set_xlabel("coverage (%) (skip riskiest first)")
    ax.set_ylabel("accuracy among answered (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{a.figdir}/fig_run2_risk_coverage.png", dpi=200)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    prob = expit(scores["CONV"])
    bins = np.clip((prob * 8).astype(int), 0, 7)
    xs, ys = [], []
    for b in range(8):
        sel = bins == b
        if sel.sum() >= 8:
            xs.append(prob[sel].mean()); ys.append(test.wrong.values[sel].mean())
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(xs, ys, "o-", c="#0077BB")
    ax.set_xlabel("predicted P(wrong)"); ax.set_ylabel("empirical P(wrong)")
    fig.tight_layout()
    fig.savefig(f"{a.figdir}/fig_run2_reliability.png", dpi=200)

    print("\nwrote", a.outdir)


if __name__ == "__main__":
    main()
