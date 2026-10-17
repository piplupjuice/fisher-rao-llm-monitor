"""Phase 1+3+4 audit of the geodesic decoding study.

Reproduces, verifies, and extends the study's statistics from the raw CSVs.
Reads ONLY the raw study files; writes corrected tables to analysis/outputs/
and figures to analysis/figures/. Deterministic: all randomness seeded with
SEED=42. Original raw files are never modified.

Monitors are evaluated as RISK scores: higher score => predicts WRONG.
For AUROC we take the positive class = wrong answer. Where the notebook used
'confidence' (higher = right), AUROC and AUPRC are 1 - confidence-AUROC.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression

SEED = 42
DATA = "geodesic_study"
OUT = "analysis/outputs"
FIGS = "analysis/figures"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

METHODS = ["greedy", "nll_only", "fr_only", "entropy_only",
           "full_geodesic", "conservative"]
LOOKAHEAD = ["fr_only", "entropy_only", "full_geodesic", "conservative"]
ALPHA = 0.5   # notebook full-geodesic weight on z(d_greedy)
BETA = 0.5    # weight on z(H(q_j))

# ---------------------------------------------------------------- helpers ---
def auroc(y, s):
    """AUROC with 1 = wrong as positive class.
    Efron-style rank formula (handles ties exactly like the notebook's)."""
    y = np.asarray(y, float); s = np.asarray(s, float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    if len(np.unique(y)) < 2 or len(y) < 2:
        return np.nan
    r = rankdata(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def auprc(y, s):
    """AUPRC with 1 = wrong as positive class, score must point to wrong."""
    y = np.asarray(y, float); s = np.asarray(s, float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    if y.sum() == 0 or len(y) == 0:
        return np.nan
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y); prec = tp / (np.arange(len(y)) + 1)
    rec = tp / max(y.sum(), 1)
    d = np.diff(np.concatenate([[0], rec]))
    return float((d * prec).sum())


def brier(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    if not len(y):
        return np.nan
    lo, hi = np.nanmin(s), np.nanmax(s)
    c = (s - lo) / (hi - lo) if hi > lo else s * 0 + 0.5
    return float(np.mean((c - y) ** 2))


def pct_risk(x):
    x = np.asarray(x, float)
    out = np.full(len(x), 0.5)
    m = ~np.isnan(x)
    if m.sum() > 1:
        r = rankdata(x[m])
        span = r.max() - r.min()
        out[m] = (r - 1) / span if span > 0 else 0.5
    return out


def boot_metric(y, s, metric, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float); s = np.asarray(s, float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        v = metric(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def boot_diff(y, s_a, s_b, metric=auroc, n_boot=2000, seed=SEED):
    """Paired bootstrap of metric(a) - metric(b) over the SAME questions."""
    rng = np.random.default_rng(seed)
    m = ~np.isnan(s_a) & ~np.isnan(s_b)
    y, sa, sb = np.asarray(y, float)[m], np.asarray(s_a)[m], np.asarray(s_b)[m]
    n = len(y)
    n_a = metric(y, sa); n_b = metric(y, sb)
    ds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = metric(y[idx], sa[idx]) - metric(y[idx], sb[idx])
        if not np.isnan(d):
            ds.append(d)
    ds = np.asarray(ds)
    return (n_a, n_b,
            float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975)))


def mcnemar(a, b):
    a = np.asarray(a).astype(float); b = np.asarray(b).astype(float)
    from scipy.stats import binomtest
    w = int(((a == 0) & (b == 1)).sum())  # B: greedy right, method wrong
    l = int(((a == 1) & (b == 0)).sum())  # C: greedy wrong, method right
    pv = binomtest(w, n=w + l, p=0.5).pvalue if w + l else 1.0
    return w, l, pv


# ------------------------------------------------------------- load & fix ---
pred_s = pd.read_csv(f"{DATA}/predictions.csv", dtype=str)

def norm_num(s):
    """Replicate the study's normalize_number on CSV round-tripped values.
    The script's read_parts -> to_csv round-trip turned '18' into '18.0'
    (pandas float inference), so the audit must normalize before comparing."""
    s = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s

pred_s["pred_n"] = pred_s["prediction"].map(norm_num)
pred_s["gold_n"] = pred_s["gold"].map(norm_num)
pred_s["correct_raw"] = (pred_s["pred_n"] == pred_s["gold_n"]).astype(int)

# numeric view for convenience
pred = pd.read_csv(f"{DATA}/predictions.csv")
pred["correct_checked"] = pred_s["correct_raw"].values
mismatch = (pred.correct != pred.correct_checked).sum()
pred["answer_len"] = pred.generated_tokens

summary = pd.read_csv(f"{DATA}/summary.csv")
split_map = pd.read_csv(f"{DATA}/split_map.csv")
SPLIT = dict(zip(split_map.question_id.astype(int), split_map.split))

counts = pred.pivot_table(index="split", columns="method", values="question_id",
                          aggfunc="count")
expected = {(sp, m): (100 if sp == "dev" else 400) for sp in ["dev", "test"]
            for m in METHODS}
counts_ok = all(int(counts.loc[sp, m]) == n for (sp, m), n in expected.items())
audit_rows = []
def audit(name, value):
    audit_rows.append({"check": name, "result": value})
    print(f"[audit] {name}: {value}")

audit("prediction/gold exact-match recompute mismatches", int(mismatch))
audit("rows in predictions.csv", len(pred))
audit("duplicate (qid,method) rows", int(pred.duplicated(subset=["question_id", "method"]).sum()))
audit("all method x split cells at exact expected counts (dev=100, test=400)",
      bool(counts_ok))

# recompute summary
rec = pred.groupby("method").agg(acc=("correct_checked", "mean"),
                                 n=("correct_checked", "size"))
j = rec.join(summary.set_index("method").accuracy, how="left")
max_diff = float((j.acc - j.accuracy).abs().max())
audit("max |recomputed-summary| accuracy diff", max_diff)
j.round(6).to_csv(f"{OUT}/verified_summary.csv")

# parts consistency: merged traces should equal de-duplicated parts
p_keys = pd.concat([pd.read_csv(f"{DATA}/token_traces_part0.csv", usecols=["question_id", "method", "step"]),
                    pd.read_csv(f"{DATA}/token_traces_part1.csv", usecols=["question_id", "method", "step"])])
merged = pd.read_csv(f"{DATA}/token_traces.csv", usecols=["question_id", "method", "step"])
audit("trace rows dedup'd parts == merged rows",
      len(p_keys.drop_duplicates()) == len(merged.drop_duplicates()))

# paired comparison vs predictions pivot
wide = pred.pivot(index="question_id", columns="method", values="correct_checked")
paired_file = pd.read_csv(f"{DATA}/paired_comparison.csv").set_index("question_id")
pair_match = all((wide[m].fillna(-9).astype(int) == paired_file[m].fillna(-9).astype(int)).all()
                 for m in METHODS)
audit("paired_comparison.csv matches predictions pivot", bool(pair_match))

# --------------------------------------------- aggregate monitor features ---
tr = pd.read_csv(f"{DATA}/token_traces.csv",
                 usecols=["method", "question_id", "step", "nll", "fr",
                          "d_centroid", "next_entropy", "current_entropy",
                          "gate"])
feat = tr.groupby(["method", "question_id"]).agg(
    n_steps=("step", "count"),
    mean_H=("current_entropy", "mean"),
    max_H=("current_entropy", "max"),
    late_H=("current_entropy", lambda s: s.tail(max(1, len(s) // 4)).mean()),
    mean_nll=("nll", "mean"),
    mean_prob=("nll", lambda s: float(np.exp(-np.asarray(s)).mean())),
    gate_frac=("gate", "mean"),
    mean_next_H=("next_entropy", "mean"),
    mean_fr=("fr", "mean"),
    mean_dcent=("d_centroid", "mean"),
    n_look=("fr", "count"),
).reset_index()
feat = feat.merge(pred[["question_id", "method", "correct_checked", "answer_len"]],
                  on=["method", "question_id"])
feat["log_len"] = np.log(feat.answer_len.clip(lower=1))
feat["split"] = feat.question_id.map(SPLIT)

# notebook-faithful COMBINED risk (mean of 4 percentile-ranks; NaN -> neutral 0.5)
def add_grouped_score(df, colname, parts_fn):
    df[colname] = np.nan
    for m, g in df.groupby("method", sort=False):
        df.loc[g.index, colname] = np.mean(parts_fn(g), axis=0)

add_grouped_score(feat, "risk_comb",
                  lambda g: [pct_risk(g.mean_H), pct_risk(g.max_H),
                             pct_risk(g.mean_next_H), pct_risk(g.gate_frac)])
add_grouped_score(feat, "risk_conventional",
                  lambda g: [pct_risk(g.mean_H), pct_risk(g.max_H),
                             pct_risk(g.mean_nll), pct_risk(g.mean_next_H),
                             pct_risk(g.gate_frac)])

feat.to_csv(f"{OUT}/monitor_features.csv", index=False)

# ------------------------------------------------------------------ stats ---
t = feat[feat.split == "test"].copy()

FEATS = [("mean_nll", "token/answer NLL"), ("mean_H", "mean entropy"),
         ("max_H", "max entropy"), ("late_H", "late mean entropy"),
         ("mean_prob", "mean chosen prob"), ("gate_frac", "gate open frac"),
         ("mean_next_H", "next-step entropy"), ("mean_fr", "FR->greedy future"),
         ("mean_dcent", "FR->consensus"), ("n_look", "#lookahead steps"),
         ("log_len", "log answer length"), ("risk_conventional", "CONV combo"),
         ("risk_comb", "notebook COMBINED")]

rows = []
for m in METHODS:
    s = t[t.method == m]
    n_pos = int((s.correct_checked == 0).sum())
    n_neg = int((s.correct_checked == 1).sum())
    for col, label in FEATS:
        y = s.correct_checked.astype(float).values  # 1 = correct => wrong = 1-y
        y_wrong = 1 - y
        val = auroc(y_wrong, s[col].values)
        lo, hi = boot_metric(y_wrong, s[col].values, auroc)
        rows.append({"method": m, "feature": label, "col": col, "auroc": val,
                     "lo": lo, "hi": hi, "auprc": auprc(y_wrong, s[col].values),
                     "brier": brier(y_wrong, s[col].values),
                     "n_wrong": n_pos, "n_right": n_neg,
                     "prevalence_wrong": n_pos / len(s)})
au = pd.DataFrame(rows)
au.to_csv(f"{OUT}/auroc_table.csv", index=False)
print(au[au.col.isin(["mean_H", "max_H", "mean_nll", "mean_fr", "mean_dcent",
                      "risk_conventional", "risk_comb"])]
      .pivot(index="col", columns="method", values="auroc").round(3).to_string())

# expected-value baseline (ingrained prevalence) AUPRC = prevalence
prev = t.groupby("method").apply(lambda s: (s.correct_checked == 0).mean())
prev.round(4).to_csv(f"{OUT}/prevalence_wrong.csv")
print("\ntest prevalence (wrong) per method:", prev.round(4).to_dict())

# ---------------------------------------------- paired bootstrap of AUROC ---
# Question: does geometric info (mean_fr / mean_dcent) add AUROC beyond the
# conventional-only combo risk? All on test questions of each method.
inc_rows = []
for m in METHODS:
    s = t[t.method == m]
    y = 1 - s.correct_checked.astype(float).values
    conv_risk = pct_risk(s.risk_conventional)
    base = s.risk_conventional.values
    for geo in ["mean_fr", "mean_dcent"]:
        combo = np.mean([conv_risk, pct_risk(s[geo])], axis=0)
        n_a, n_b, lo, hi = boot_diff(y, combo, base)
        inc_rows.append({"method": m, "geometric": geo,
                         "auroc_base": n_b, "auroc_geo_added": n_a,
                         "diff": n_a - n_b, "lo": lo, "hi": hi})
inc = pd.DataFrame(inc_rows)
inc.to_csv(f"{OUT}/incremental_percentile.csv", index=False)
print("\nincremental AUROC (percentile-rank combo):")
print(inc.round(3).to_string(index=False))

# --------------------------------------------- redundancy (Spearman) --------
corr_rows = []
for m in METHODS:
    s = feat[(feat.split == "test") & (feat.method == m)]
    cols = ["mean_H", "max_H", "late_H", "mean_nll", "mean_next_H",
            "gate_frac", "mean_fr", "mean_dcent", "n_look", "log_len"]
    use = [c for c in cols if s[c].notna().sum() > 20]
    for i in range(len(use)):
        for k in range(i + 1, len(use)):
            cc = s[[use[i], use[k]]].dropna()
            r = spearmanr(cc.iloc[:, 0], cc.iloc[:, 1]).statistic
            corr_rows.append({"method": m, "a": use[i], "b": use[k], "rho": r})
pd.DataFrame(corr_rows).to_csv(f"{OUT}/spearman_redundancy.csv", index=False)

# ----------------------------------- trained monitor: dev fit -> test eval --
# tiny dev (100 q) => logistic with fixed C=1, prespecified features, no tuning
def xy(s, cols):
    X = s[cols].copy()
    for c in cols:
        med = X[c].median()
        X[c] = X[c].fillna(med if np.isfinite(med) else 0.0)
    return X.values, (s.correct_checked == 0).astype(int).values  # wrong = 1

CONV = ["mean_H", "max_H", "late_H", "mean_nll", "mean_next_H", "gate_frac",
        "mean_prob", "log_len"]
GEO = ["mean_fr", "mean_dcent", "n_look"]

trained_rows = []
pred_cache = {}
for m in METHODS:
    dev = feat[(feat.split == "dev") & (feat.method == m)]
    test = feat[(feat.split == "test") & (feat.method == m)]
    for name, cols in [("CONVENTIONAL", CONV),
                       ("CONV+GEOM", CONV + GEO),
                       ("GEOM only", GEO)]:
        Xd, yd = xy(dev, cols)
        Xt, yt = xy(test, cols)
        lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        lr.fit(Xd, yd)
        pa = lr.predict_proba(Xt)[:, 1]
        pb = lr.predict_proba(Xd)[:, 1]
        lo, hi = boot_metric(yt, pa, auroc)
        pred_cache[(m, name)] = (yt, pa)
        trained_rows.append({
            "method": m, "model": name, "dev_auroc": auroc(yd, pb),
            "test_auroc": auroc(yt, pa), "test_ci_lo": lo, "test_ci_hi": hi,
            "test_auprc": auprc(yt, pa), "test_brier": brier(yt, pa),
        })
# paired comparison between model variants on the same test questions
for m in METHODS:
    y = pred_cache[(m, "CONV+GEOM")][0]
    s_all = pred_cache[(m, "CONV+GEOM")][1]
    s_conv = pred_cache[(m, "CONVENTIONAL")][1]
    s_geo = pred_cache[(m, "GEOM only")][1]
    n_a, n_b, lo2, hi2 = boot_diff(y, s_all, s_conv)
    trained_rows.append({"method": m, "model": "CONV+GEOM minus CONVENTIONAL",
                         "dev_auroc": np.nan, "test_auroc": n_a - n_b,
                         "test_ci_lo": lo2, "test_ci_hi": hi2,
                         "test_auprc": np.nan, "test_brier": np.nan})
tr_out = pd.DataFrame(trained_rows)
tr_out.round(4).to_csv(f"{OUT}/trained_monitors.csv", index=False)
print("\ntrained monitors (dev-fit, test-eval):")
print(tr_out.round(3).to_string(index=False))

# --------------------------------.. transfer test for the 0.844 artifact ----
# If the fr_only monitor's power were a property of FR geometry, a monitor
# trained on fr_only generations should transfer to full_geodesic/greedy.
transfer_rows = []
for src in ["fr_only", "entropy_only", "conservative"]:
    src_dev = feat[(feat.split == "dev") & (feat.method == src)]
    Xd, yd = xy(src_dev, CONV + GEO)
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    lr.fit(Xd, yd)
    for dst in ["full_geodesic", "conservative", "greedy"]:
        dst_test = feat[(feat.split == "test") & (feat.method == dst)]
        Xt, yt = xy(dst_test, CONV + GEO)
        p = lr.predict_proba(Xt)[:, 1]
        transfer_rows.append({"source": src, "target": dst,
                              "test_auroc": auroc(yt, p)})
tf = pd.DataFrame(transfer_rows).round(4)
tf.to_csv(f"{OUT}/monitor_transfer.csv", index=False)
print("\ntransfer (train on source generations, eval on target's test):")
print(tf.pivot(index="source", columns="target", values="test_auroc").to_string())

# --------------------------------------------------------- error cases ------
# (divergence analysis handled in probe scripts; failure_cases.csv verified)
fail = pd.read_csv(f"{DATA}/failure_cases.csv", dtype=str)
fail_counts = fail.groupby(["method", "category"]).size().unstack(fill_value=0)
fail_counts.to_csv(f"{OUT}/failure_category_counts.csv")
print("\nfailure case categories:")
print(fail_counts.to_string())


# ---------------------------------------------- decoding deltas re-check ----
dec_rows = []
wide_s = wide.copy()
wide_s["split"] = [SPLIT[q] for q in wide_s.index]
test_wide = wide_s[wide_s.split == "test"]
for m in METHODS:
    if m == "greedy":
        continue
    a = test_wide[m].astype(float); b = test_wide["greedy"].astype(float)
    d = (a - b).values
    rng = np.random.default_rng(SEED)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    w, l, pv = mcnemar(b.values, a.values)
    dec_rows.append({"method": m, "delta_pp": 100 * d.mean(),
                     "lo": 100 * np.quantile(boots, .025),
                     "hi": 100 * np.quantile(boots, .975),
                     "greedy_right": w, "method_right": l,
                     "mcnemar_p": pv})
dec = pd.DataFrame(dec_rows)
dec.round(5).to_csv(f"{OUT}/decoding_paired_stats.csv", index=False)
print("\ndecoding deltas vs greedy (test):")
print(dec.round(4).to_string(index=False))

# --------------------------------------------------------- figures ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10})

# Fig 1: AUROC forest for key features, per method
piv = au[au.col.isin(["mean_nll", "mean_H", "max_H", "mean_fr", "mean_dcent",
                      "risk_conventional", "risk_comb"])]
fig, ax = plt.subplots(figsize=(10, 6))
ys = np.arange(len(piv))
ax.hlines(ys, piv.lo, piv.hi, color="#999", lw=1)
ax.scatter(piv.auroc, ys, s=28, zorder=3)
ax.set_yticks(ys, [f"{r.method} · {r.feature}" for r in piv.itertuples()])
ax.axvline(0.5, color="k", lw=1, ls="--")
ax.set_xlim(0.3, 1.0); ax.invert_yaxis()
ax.set_xlabel("AUROC (wrong = positive, higher risk score => wrong)")
ax.set_title("Recomputed monitor AUROC, 95% bootstrap CI (test n=400)")
fig.tight_layout(); fig.savefig(f"{FIGS}/fig_auroc_forest.png", dpi=200)

# Fig 2: reliability diagrams for notebook-COMBINED risk on the two methods
# whose decoding matches greedy (full_geodesic, conservative) + fr_only artifact
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, m in zip(axes, ["full_geodesic", "conservative", "fr_only"]):
    s = t[t.method == m]
    y = (s.correct_checked == 0).astype(float).values
    r = s.risk_comb.values
    bins = np.linspace(0, 1, 9)
    ids = np.clip(((r - r.min()) / max(r.max() - r.min(), 1e-9) * 7.999).astype(int), 0, 7)
    xs, ys = [], []
    for b in range(8):
        if (ids == b).sum() > 5:
            xs.append(r[ids == b].mean()); ys.append(y[ids == b].mean())
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(xs, ys, "o-", color="#0077BB")
    ax.set_title(f"{m} (Brier={brier(y, r):.3f})")
    ax.set_xlabel("predicted P(wrong) (rank-scaled)")
axes[0].set_ylabel("empirical P(wrong)")
fig.suptitle("Reliability: notebook COMBINED risk (test) — rank-scaled, not calibrated")
fig.tight_layout(); fig.savefig(f"{FIGS}/fig_reliability.png", dpi=200)

# Fig 3: risk-coverage curves, test, comparing monitors
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for m, colr in [("full_geodesic", "#0077BB"), ("conservative", "#EE7733"),
                ("fr_only", "#CC3311"), ("greedy", "#6B7280")]:
    s = t[t.method == m].copy()
    for monitor, label, ls in [("risk_comb", "notebook COMBINED", "-"),
                               ("risk_conventional", "conventional combo", "--")]:
        s2 = s.dropna(subset=[monitor]).sort_values(monitor)
        yw = (s2.correct_checked == 0).astype(float).values
        cov = (np.arange(len(s2)) + 1) / len(s2)
        acc = (1 - yw).cumsum() / (np.arange(len(s2)) + 1)  # accuracy among the answered prefix
        ax.plot(100 * cov, 100 * acc, color=colr, ls=ls, lw=1.8,
                label=f"{m} · {label}")
ax.set_ylim(0, 102); ax.legend(fontsize=7)
ax.set_xlabel("coverage (%) — skip riskiest first")
ax.set_ylabel("accuracy among answered (%)")
ax.set_title("Risk-coverage (test): COMBINED vs conventional-only monitors")
fig.tight_layout(); fig.savefig(f"{FIGS}/fig_risk_coverage.png", dpi=200)

with open(f"{OUT}/audit_lines.txt", "w") as f:
    for r in audit_rows:
        f.write(f"{r['check']}: {r['result']}\n")

print("\n[done] wrote", OUT, "and", FIGS)
