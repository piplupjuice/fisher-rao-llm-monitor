"""Run-2 analysis: does Fisher-Rao add correctness information beyond
entropy/NLL, with all features measured on the SAME deterministic greedy
trajectory?

Consumes geodesic_study2/monitor_{steps,candidates,answers}.csv.
Never modifies inputs. All pre-registered in experiment_protocol.md (run-2
addendum), written BEFORE any labeled run-2 data existed.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression

SEED = 42


def auroc(y, s):
    m = ~np.isnan(s); y, s = y[m], s[m]
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
    lo, hi = np.nanmin(s), np.nanmax(s)
    c = (s - lo) / (hi - lo) if hi > lo else s * 0 + 0.5
    return float(np.mean((c - y) ** 2))


def boot_ci(metric, y, s, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float); s = np.asarray(s, float)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    vals = [v for v in vals if not np.isnan(v)]
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
    return base, float(np.quantile(disp, .025)), float(np.quantile(disp, .975))


def pct_risk(x):
    x = np.asarray(x, float)
    out = np.full(len(x), 0.5)
    m = ~np.isnan(x)
    if m.sum() > 1:
        r = rankdata(x[m]); span = r.max() - r.min()
        out[m] = (r - 1) / span if span > 0 else 0.5
    return out


# --------------------------------------------------- per-question features --
def build_question_table(indir):
    steps = pd.read_csv(f"{indir}/monitor_steps.csv")
    cands = pd.read_csv(f"{indir}/monitor_candidates.csv")
    ans = pd.read_csv(f"{indir}/monitor_answers.csv",
                      usecols=["question_id", "split", "gold", "correct",
                               "stopped_on", "seconds", "forward_calls",
                               "generated_tokens", "model"])

    g = cands[cands.is_greedy == 1].set_index(["question_id", "step"])
    alt = cands[cands.is_greedy == 0]
    alt_greedy = alt.groupby(["question_id", "step"]).agg(
        alt_fr_greedy_mean=("fr_greedy", "mean"),
        alt_fr_greedy_max=("fr_greedy", "max"),
        alt_fr_greedy_min=("fr_greedy", "min"),
        alt_q_ent_min=("q_entropy", "min")).reset_index()

    s = steps.merge(g[["q_entropy", "fr_greedy", "fr_consensus"]],
                    left_on=["question_id", "step"], right_index=True,
                    suffixes=("", "_lk"))
    s["late_flag"] = s.step >= (s.step.max() + 1) // 2   # dev-only proxy, refine below
    s = s.merge(alt_greedy, on=["question_id", "step"])
    s["h_gap"] = s.q_entropy - s.h_p
    s["fr_gap"] = s.fr_greedy - s.fr_consensus

    # window: last quarter of STEPS for late_H (needs per-question step count)
    n_steps = steps.groupby("question_id").step.max().rename("n_steps").reset_index()
    s = s.merge(n_steps, on="question_id")
    s = s[s.step >= (s.n_steps // 4 * 3)]

    agg = s.groupby("question_id").agg(
        mean_H=("h_p", "mean"), max_H=("h_p", "max"), late_H=("h_p", "mean"),
        mean_nll=("nll_greedy", "mean"), mean_next_H=("q_entropy", "mean"),
        mean_prob=("p_gap01", "mean"), tie_frac=("tie_flag", "mean"),
        gate_frac=("h_p", lambda x: (x > 1.0).mean()),
        # GEOMETRIC (lookahead, per step, greedy-anchored)
        mean_frcons=("fr_consensus", "mean"), max_frcons=("fr_consensus", "max"),
        mean_altdist=("alt_fr_greedy_mean", "mean"),
        mean_altdist_max=("alt_fr_greedy_max", "mean"),
        mean_altspread=("alt_fr_greedy_max", lambda x: x.max() - x.min()),
        mean_hgap=("q_entropy", "mean"), std_fr=("fr_greedy", "std"),
    ).reset_index()
    agg["log_len"] = np.log(agg.merge(n_steps, on="question_id").n_steps.clip(1))
    agg = agg.merge(ans, on="question_id")
    agg["wrong"] = 1 - agg.correct
    return agg


CONV = ["mean_H", "max_H", "late_H", "mean_nll", "mean_next_H", "mean_prob",
        "gate_frac", "log_len"]
GEO = ["mean_frcons", "max_frcons", "mean_altdist", "mean_altdist_max",
       "mean_altspread", "mean_hgap", "std_fr"]


def fit_score(dev, test, cols):
    med = {c: (dev[c].median() if np.isfinite(dev[c].median()) else 0.0)
           for c in cols}
    Xd = dev[cols].fillna(pd.Series(med)).values
    Xt = test[cols].fillna(pd.Series(med)).values
    mu, sd = Xd.mean(0), Xd.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=4000, random_state=SEED)
    lr.fit((Xd - mu) / sd, dev.wrong.values)
    return lr.decision_function((Xt - mu) / sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="geodesic_study2")
    ap.add_argument("--outdir", default="analysis/outputs2")
    ap.add_argument("--figdir", default="analysis/figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    # merge part files if needed (crash-safe appends from the runner)
    for stem in ["monitor_steps", "monitor_candidates", "monitor_answers"]:
        merged = f"{a.indir}/{stem}.csv"
        if not os.path.exists(merged):
            parts = sorted(
                p for p in os.listdir(a.indir)
                if p.startswith(stem + "_part") and p.endswith(".csv"))
            df = pd.concat([pd.read_csv(f"{a.indir}/{p}") for p in parts],
                           ignore_index=True)
            df.to_csv(merged, index=False)
            print(f"[merge] wrote {merged} from {parts}")

    tab = build_question_table(a.indir)
    tab.to_csv(f"{a.outdir}/question_features.csv", index=False)

    # sanity checks (RQ 10)
    checks = []
    steps = pd.read_csv(f"{a.indir}/monitor_steps.csv")
    cands = pd.read_csv(f"{a.indir}/monitor_candidates.csv")
    ans = pd.read_csv(f"{a.indir}/monitor_answers.csv")
    joins = steps.groupby("question_id").size()
    _gen = ans.set_index("question_id").generated_tokens.reindex(joins.index)
    checks.append(("steps == generated_tokens for all q",
                  bool((joins == _gen).all())))
    nv = cands.groupby(["question_id", "step"]).size()
    checks.append(("4 candidates for every step", bool((nv == 4).all())))
    mg = cands[cands.is_greedy == 1].set_index(["question_id", "step"])
    chosen = steps.set_index(["question_id", "step"]).chosen_token
    checks.append(("greedy candidate == chosen token",
                   bool(mg.token_id.reindex(chosen.index).eq(chosen).all())))
    checks.append(("zero NaN in geometric features",
                   int(cands[["q_entropy", "fr_greedy", "fr_consensus"]]
                       .isna().sum().sum()) == 0))
    checks.append(("fr_greedy of rank0 ~ 0 (GPU fp32 tol < 3e-3)",
                   float(mg.fr_greedy.abs().max()) < 3e-3))
    checks.append(("fr_consensus of rank0 positive somewhere",
                   float(mg.fr_consensus.max()) > 0))
    checks.append(("tie events recorded",
                   int(steps.tie_flag.sum())))
    checks.append(("ties with p_gap01 > 0 inconsistency",
                   int(((steps.tie_flag == 1) & (steps.p_gap01 > 0)).sum())))
    pd.DataFrame(checks, columns=["check", "result"]).to_csv(
        f"{a.outdir}/sanity_checks.csv", index=False)
    print(checks)

    dev = tab[tab.split == "dev"]; test = tab[tab.split == "test"]
    y = dev.wrong.values
    print("test prevalence wrong:", test.wrong.mean())
    if len(dev) == 0 or len(np.unique(dev.wrong.values)) < 2:
        print("[warn] no/lone-class dev questions: dry-run only (smoke data), "
              "skipping monitor fit — full run has n_dev=100")
        tab[CONV + GEO[:6] + ["wrong", "split", "question_id"]].to_csv(
            f"{a.outdir}/question_features.csv", index=False)
        return

    rows = []
    for name, cols in ([("CONV", CONV)]
                       + [(f"CONV+{c}", CONV + [c]) for c in GEO]
                       + [("CONV+ALL_GEO", CONV + GEO)]
                       + [("GEO only", GEO)]):
        p = fit_score(dev, test, cols)
        au = auroc(test.wrong.values, p)
        lo, hi = boot_ci(auroc, test.wrong.values, p)
        rows.append({"model": name, "test_auroc": au, "lo": lo, "hi": hi,
                     "auprc": auprc(test.wrong.values, p),
                     "brier": brier(test.wrong.values, p)})
    mon = pd.DataFrame(rows)
    mon.round(4).to_csv(f"{a.outdir}/trained_monitors.csv", index=False)
    print(mon.round(3).to_string(index=False))

    print("\npaired deltas vs CONV (same questions):")
    base = fit_score(dev, test, CONV)
    delta_rows = []
    for c in GEO:
        pdelta = fit_score(dev, test, CONV + [c])
        b, lo, hi = boot_paired(test.wrong.values, pdelta, base)
        delta_rows.append({"delta_model": f"+{c}", "d_auroc": b,
                           "lo": lo, "hi": hi})
    pd_all = fit_score(dev, test, CONV + GEO)
    b, lo, hi = boot_paired(test.wrong.values, pd_all, base)
    delta_rows.append({"delta_model": "+ALL_GEO", "d_auroc": b, "lo": lo, "hi": hi})
    deltas = pd.DataFrame(delta_rows)
    deltas.round(4).to_csv(f"{a.outdir}/paired_deltas.csv", index=False)
    print(deltas.round(3).to_string(index=False))

    # correlation/redundancy on dev
    corr = dev[CONV + GEO].corr(method="spearman")
    corr.round(3).to_csv(f"{a.outdir}/spearman_dev.csv")

    # risk-coverage (cumulative accuracy among retained prefix, skip riskiest)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, cols in (("CONV", CONV), ("CONV+ALL_GEO", CONV + GEO),
                        ("GEO only", GEO)):
        p = fit_score(dev, test, cols)
        order = np.argsort(-p)  # riskiest first -> skip
        yw = test.wrong.values[order]
        cov = (np.arange(len(yw)) + 1) / len(yw)
        acc = ((1 - yw).cumsum()) / cov
        ax.plot(100 * cov, 100 * acc, lw=2, label=label)
    ax.axvline(80, ls="--", c="k", lw=1)
    ax.set_xlabel("coverage (%) (skip riskiest first)")
    ax.set_ylabel("accuracy among answered (%)")
    ax.set_title("risk-coverage (test)")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{a.figdir}/fig_run2_risk_coverage.png", dpi=200)

    # reliability diagram of the CONV+ALL_GEO monitor
    p = fit_score(dev, test, CONV + GEO)
    from scipy.special import expit
    prob = expit(p)
    bins = np.clip((prob * 8).astype(int), 0, 7)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    xs, ys = [], []
    for b in range(8):
        sel = bins == b
        if sel.sum() >= 8:
            xs.append(prob[sel].mean()); ys.append(test.wrong.values[sel].mean())
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.plot(xs, ys, "o-", c="#0077BB")
    ax.set_xlabel("predicted P(wrong)"); ax.set_ylabel("empirical P(wrong)")
    ax.set_title("reliability (CONV+ALL_GEO, test)")
    fig.tight_layout(); fig.savefig(f"{a.figdir}/fig_run2_reliability.png", dpi=200)

    print("\nwrote", a.outdir)

if __name__ == '__main__':
    main()
