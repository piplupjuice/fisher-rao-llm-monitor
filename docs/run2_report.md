# Run-2 Report — Full-Coverage Greedy-Anchored Monitor

**Date:** 2026-10-17 · **Model:** Qwen2.5-1.5B-Instruct (bf16) · **Kaggle 2×T4, wall 3.65 h**
**Code:** `analysis/monitor_run.py` (kernel `pikanigga/geodesic-run-2-full`, v1)
**Data:** 500 GSM8K questions, frozen run-1 split (100 dev / 400 test), run-1 golds

## Verdict (pre-registered criteria, docs/experiment_protocol.md addendum)

| Hypothesis | Result |
|---|---|
| **H1 detectability** — monitor ranks correctness above chance | **PASS.** CONV monitor test AUROC **0.704** [0.641, 0.762], AUPRC 0.533 (prevalence 0.275), Brier 0.178 |
| **H2 incremental FR information** — adding Fisher–Rao features to conventional features helps on held-out data | **FAIL.** Every prespecified FR addition is non-positive; AUPRC drops in **all** 8 variants; paired ΔAUROC ≤ 0; `+max_frcons` is significantly **negative** (−0.011, CI [−0.022, −0.001]) |

**Answer to the driving question: no — Fisher–Rao features provide no incremental
correctness information beyond entropy/NLL when all features are measured on the
same greedy-decoded trajectories. The 0.844 from run 1 came from entropy
features on degraded, gate-censored generations, not from Fisher–Rao geometry.**

## Main results (test = 400 held-out questions)

| Monitor | AUROC [95% CI] | AUPRC | Brier |
|---|---|---|---|
| CONV (entropy/NLL/length only) | **0.704** [0.641, 0.762] | **0.533** | **0.178** |
| CONV + ALL_GEO | 0.670 [0.611, 0.727] | 0.432 | 0.257 |
| GEO only (all 7 FR features) | 0.669 [0.611, 0.724] | 0.401 | 0.239 |

## Paired ΔAUROC vs CONV (bootstrap, 2000 resamples, seed 42)

| + feature | Δ | 95% CI |
|---|---|---|
| +mean_frcons | −0.011 | [−0.059, +0.039] |
| +max_frcons | **−0.011** | **[−0.022, −0.001]** (significant, wrong direction) |
| +mean_altdist | −0.016 | [−0.066, +0.037] |
| +mean_altdist_max | −0.002 | [−0.022, +0.017] |
| +mean_altspread | 0.000 | [−0.003, +0.004] |
| +mean_hgap | −0.001 | [−0.004, +0.002] |
| +std_fr | −0.001 | [−0.017, +0.015] |
| +ALL_GEO | −0.033 | [−0.087, +0.023] |

## Why FR fails even though it correlates (the redundancy story)

Individually, the FR aggregates look promising — on dev, `mean_altdist`
(Spearman with wrongness **0.46**), `mean_frcons` (0.43) beat plain `mean_H`
(0.28); their univariate test AUROCs are 0.664 / 0.651. But run 1 + run 2
agree on the mechanism: those correlations ride on the same underlying factors
already captured by entropy (mean/max/late), answer length (`log_len`, univariate
AUROC 0.669), and gate-fraction. In the multivariate monitor the FR features add
no unique axis — worse, some (max_frcons) harm. See
`analysis/outputs2/univariate_auroc.csv` and `spearman_dev.csv`.

## Execution verification (user-required alignment gate)

Desktop CPU smoke (Qwen2.5-0.5B) and Kaggle GPU smoke (real 1.5B) both passed
all pre-GPU checks:

- **Alignment:** steps == generated_tokens (500/500), 4 candidate rows per step
  (519,356 rows total), greedy candidate's token == decoded token on every step
- **Deterministic decoding:** smoke candidates **byte-identical across three
  independent runs and two separate Kaggle GPU sessions** (md5 d7db02eb…)
- **Exact arithmetic:** forward_calls == 1 + 5·steps − eos (verified per question;
  495 eos-stops, 5 token-limit stops)
- **No failures:** zero per-question failures on the full run; no NaN features;
  FR rank-0 floor 1.5e-3 (fp32 acos over 152k vocab), signal range [0, π]
- **Reproducibility vs run 1:** predictions agree with run-1's greedy arm on
  **500/500** questions; accuracy identical (0.714). Ties occurred at 969 of
  129,839 steps (0.75%) — all resolved by smaller-token-id rule.

## Limitations (honest bounds)

1. Power: n=400 test; the paired CI can exclude FR gains larger than ≈ +0.04
   AUROC, but cannot exclude very small gains (< ~0.01).
2. One model (1.5B, bf16), one dataset (GSM8K), TOP_K=4, one-step lookahead
   only — FR geometry beyond one token ahead untested. H3 (model/dataset
   transfer) remains open but is moot unless a stronger FR variant is found.
3. FR measured on candidate futures only, never across decoded positions of the
   full trajectory (different quantity from run-1's "merged" score).
4. Monitor is question-level (mean/max aggregates); a step-level sequence
   model could in principle extract more — not tested (would need dev much
   larger than 100).
5. The bootstrap point estimates and paired deltas were correct throughout;
   an intermediate version of the univariate-CI helper resampled y and s
   independently (CIs collapsed to ~0.5); fixed before any inference was drawn
   (journal-documented).

## Files

- `analysis/outputs2/` — trained_monitors.csv, paired_deltas.csv,
  univariate_auroc.csv, spearman_dev.csv, question_features.csv, sanity_checks.csv
- `analysis/figures/` — fig_run2_risk_coverage.png, fig_run2_reliability.png
- `kaggle_out/full/geodesic_study2/` — raw monitor outputs (58 MB candidate
  table; 129,839 steps; all frozen run-1 IDs/splits/golds)
- `analysis/fetch_and_verify.py`, `analysis/run2_analysis.py` — reproduction:
  `python analysis/run2_analysis.py --indir kag_out/full/geodesic_study2 --outdir analysis/outputs2`

---

## ADDENDUM (2026-10-17, post-audit) — numbers above superseded by audit-corrected analysis

The figures in the original body of this report were produced by
`analysis/run2_analysis_prefix.py`, which an audit (A5-A7) found deviated
from the pre-registered feature definitions (late-quarter-only aggregation
bug; `mean_hgap`/`std_fr`/`mean_altspread` mismatches; `max_frcons`
was an implementation extra, now exploratory). The corrected analysis
(`analysis/run2_analysis.py`; pre-fix artifacts preserved in
`analysis/outputs2_prefix/`) gives:

- CONventional monitor: AUROC **0.770 [0.713, 0.821]** (was 0.704), AUPRC
  **0.596**, Brier 0.159 - the bug had *understated* the conventional monitor,
  which makes the H2 failure **sharper**, not weaker.
- All six pre-registered FR additions: dAUROC <= 0 (one nominally significant
  in the harmful direction at p=0.042 before multiplicity correction);
  AUPRC decreases in every variant; risk-coverage at 80/90% worse.
- Exploratory (+max_frcons): +0.005 [-0.002, +0.012], not significant.
- sklearn cross-check of AUROC/AUPRC: max abs deviation 0.0 (all monitors).

Authoritative numbers for the paper: `analysis/outputs2/` (corrected) +
`paper/paper_draft.md`. The original report body is retained verbatim as the
audited pre-fix record.
