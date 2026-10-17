# Experiment Protocol — Geodesic Decoding (Phase 2 pre-registration)

Written **after** Phase-1 audit (which already saw test outcomes of run 1), so
run 2 numbers below are not pristine pre-registration from ignorance — the key
protections are (i) resolving every audited confound *before* measuring, and
(ii) committing to report all outcomes, favorable or not.

## H1 — Detectability (H0: FR score ranks correctness no better than chance, AUROC = 0.5)
Test on: full-coverage monitor run (below), the 400 test questions, wrong = positive.
Reject H0 only if AUROC CI (2000-re-sample bootstrap, seed 42) excludes 0.5 AND
DeLong/bootstrap p<0.05 two-sided. Report Δ AUROC vs random with CI.

## H2 — Incremental information
Primary comparison (same questions, same generation, same labels):
- A: logistic monitor on conventional features only:
  [mean_H, max_H, late_H, mean_nll, mean_next_H(unconditional), gate_frac, mean_prob, log_len]
- B: A + geometric features:
  [mean_FR(d to greedy future), mean_FR(d to consensus), sqrt(min FR), n_steps_to_min_FR]
  All FR features computed on **every** generation step for all top-4 candidates
  (no entropy gate), removing the audited gate-selection bias (§3.3).
Fit on dev (100 q, C=1 logistic, standardized with dev stats), freeze, evaluate
on test. H2 supported only if Δ AUROC (B − A) 95% paired bootstrap CI excludes 0
**and** Δ AUPRC > 0 **and** risk–coverage AUC improves at 80% and 90% coverage.
Prespecified secondary controls: B + `has_diverged_from_greedy_step1`·B + tie
indicators (steps where top-1/top-2 NLL differ < 1e-4).

## H3 — Generalization (only after run-2 protocol frozen)
Hold everything fixed; swap dataset (≥1 of: GSM8K test-unseen 500 fresh,
MultiArith, SVAMP) and model (≥1 of: Llama-3.2-1B/3B-Instruct, Qwen2.5-3B).
No thresholds or feature subsets re-selected; only re-fit same feature set on
the new dev or reuse run-2 coefficients. Report per-dataset/model AUROC table
with paired Δ CIs. No generalization claim from GSM8K alone.

## H4 — Selective prediction
If H2 supported: with frozen monitor B, at fixed abstention budget 10/20%,
report accuracy-coverage trade-off vs best conventional monitor and vs random
abstention, paired CIs. If H2 not supported, H4 is not run.

## Decoding (Phase 6, separated from monitoring)
Reuse only methods whose anchor is reproducible: greedy and full_geodesic with
deterministic tie-breaking `(argmax p; if fp32 tie, smaller token id first)` —
removes the audited topk nondeterminism (§3.5). Primary metric: exact-match
accuracy on 400 held-out questions; paired bootstrap + McNemar; report runtime,
forward calls, GPU MB; same prompt, budget (512), stopping, parser.

## Full-coverage monitor run specification (highest-value next experiment)
- Decode greedy (tie-stabilized) once per question; at **every** step, for all
  k=4 candidates, record: NLL, H(p), H(q_j), FR(q_j, q_greedy), FR(q_j, centroid).
- n = 500 questions × ~260 steps × 4 candidate forward passes ≈ 500–650k extra
  forward calls ≈ 3–4 GPU-T4-hours on Kaggle 2×T4 (~0.5–1 day wall with both GPUs).
- Outputs: `monitor_scores_full.csv` per (q, step, cand), `decisions.csv`.
- Success criteria: H1–H4 as above; **all** results reported regardless of sign.
- Compute limitations: 1.5B-class models, 2×T4 bf16, no fp32 baseline — monitor
  findings may differ at fp32 or larger scales.

## RUN-2 ADDENDUM (pre-registered BEFORE any run-2 labeled data; monitor
## feature aggregates fixed from the deterministic greedy trajectory only)

Conventional feature set (per greedy trajectory; rank-0 candidate only):
  mean_H, max_H, late_H(last quartile mean), mean_nll, mean_prob, gate_frac
  (H>1), log_len, mean_next_H (H(q) of the greedy token = conventional axis).
Geometric feature set G (uses the k=4 candidate grid on the SAME trajectory):
  G1 mean_frcons (FR of greedy future vs consensus, per step mean)
  G2 mean_altdist (mean over steps of mean_{j>0} FR(q_j, q_greedy))
  G3 mean_altdist_max (mean over steps of max_{j>0} FR(q_j, q_greedy))
  G4 mean_altspread (mean over steps of range of FR(q_j, q_greedy), j>0)
  G5 mean_hgap (mean over steps of H(q_j>j0) minus H(q_0))  [hybrid arm, secondary]
  G6 std_fr (std of FR(q_j, q_greedy) within question)
Pass criteria (unchanged from H2): pipeline-verified convention features on
dev-fit logistic (C=1, standardized with dev stats); test-set paired bootstrap
ΔAUPRC > 0 AND ΔAUROC CI excluding 0 (2000 resamples, seed 42) AND risk-coverage
gain at 80% and 90% coverage. No protocol change after seeing test results.

## RUN-2 EXECUTION JOURNAL (append-only; updated as gates fire)
- Local CPU smoke (Qwen2.5-0.5B, 2q x 48tok): all 12 gate checks PASS;
  candidates byte-identical across 3 independent runs (includes a runner
  refactor) -> deterministic tie-breaking verified end-to-end.
- Fork-found in verification (fixed pre-GPU): local loader dropped the frozen
  `split` column; now always merged from geodesic_study/split_map.csv.
- Kaggle GPU smoke v1 (1.5B bf16, cuda:0, 2q x 64tok): data pass completed
  (512 = 2*64*4 candidate rows, forward_calls = 1+64*5 = 321 exactly,
  greedy==chosen row-exact, no NaN, no ties observed); kernel marked ERROR
  only because my gate asserted rank-0 fr_greedy < 1e-3 — measured GPU floor
  is 1.5e-3 (fp32 acos amplification over the 152k vocab; mathematically 0).
  Gate constant raised to 3e-3 with justification; smoke v2 pushed.
- Smoke v2 (independent GPU session): COMPLETE, gate cell PASS; candidates
  byte-identical to v1 (md5 d7db02eb...) -> GPU cross-session determinism OK.
- FULL run (pikanigga/geodesic-run-2-full v1, 2xT4, wall 13135 s = 3.65 h,
  torch 2.11.0+cu128): 500/500 questions, steps==tokens everywhere, 519,356
  candidate rows, greedy==chosen row-exact on all steps, forward_calls match
  1 + 5*steps - eos-stops exactly (495 eos / 5 limit), zero failures, zero NaN.
  Monitor-greedy predictions identical to run-1 greedy arm on 500/500
  (accuracy 0.714 both); ties 969/129,839 steps (0.75%).
- Analysis executed with the pre-registered feature sets (dev-fit/test-eval),
  seed 42. Final: CONV AUROC 0.704 [0.641,0.762], AUPRC 0.533; ALL FR additions
  non-positive on held-out data (one significantly negative) -> **H2 FAILED,
  H1 PASSED**. Intermediate bug (independent y/s resampling in the non-paired
  CI helper, CIs collapsed toward 0.5) detected because point estimate lay
  outside its own CI, fixed, and rerun before any inference; paired-delta
  bootstrap used correct single-resample pairing throughout.

## Reporting
Every run publishes: config snapshot, seeds, versions, raw per-step CSVs,
negative results, failure cases, and compute cost. Neither decoding accuracy
ties (M1↔M0 divergence) nor selective-prediction gains are claimed before H2/H4 pass.

