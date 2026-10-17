# Replication Protocol — Cross-Scale Test of the Incremental-FR Null

**Status: DRAFT — awaiting author approval. Nothing has been launched.**
**Preregistration invariant: this file is the frozen protocol. Once approved
and timestamped (git commit / registry entry), NO change to hypotheses,
features, metrics, or analysis is permitted after any replication test
outcome is observed. Deviations go to a separate appendix file and are labeled
as deviations.**

## H3-R (primary hypothesis)

- **H₀:** On Qwen2.5-3B-Instruct GSM8K greedy trajectories, the pre-registered
  G1–G6 Fisher–Rao aggregates provide no incremental held-out correctness
  information beyond CONV: paired ΔAUROC (CONV+ALL_GEO − CONV) CI includes 0
  and/or ΔAUPRC ≤ 0.
- **Directional alternative (as in run 2):** Δ ≥ 0 is only "interesting" if
  the preregistered three-prong criterion of H2 passes: ΔAUPRC > 0 AND
  ΔAUROC CI excludes 0 AND risk–coverage accuracy gain at 80% and 90%
  coverage. **The point of this replication is to test the null's
  transfer across scale, not to search for a positive result.**

## Primary metric and minimal practically meaningful effect

- Primary: paired ΔAUROC on the untouched 400-question test set
  (2000-resample percentile bootstrap, seed 42), plus preregistered
  three-prong criterion (above).
- **Minimal practically meaningful effect (MDE): ΔAUROC ≥ +0.05.** Justification:
  run-2's paired CI half-width was ≈ ±0.021–0.043; a cross-scale claim that
  "FR becomes useful at 3B" needs an effect visible at that resolution; any
  smaller effect would also be within run-2's noise floor and thus uninformative
  about a scale-boundary claim. The replication is powered to *detect* ≥0.05
  or to *bound* replication of the null at < ±0.05 CI width.
- Secondary (same release): per-feature deltas (6 prereg), AUPRC deltas,
  Brier, risk–coverage, univariate AUROCs — all reported regardless of sign,
  all labeled confirmatory as a set, single family (8 comparisons); the paper
  will report Holm-corrected conclusions beside nominal ones.

## Design (frozen, mirrors run 2 exactly)

| Component | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct`, bf16, transformers ≥4.45 |
| Data | Same 500 GSM8K questions (IDs 0–499), same frozen split (100 dev/400 test), same golds — NO new sampling |
| Decoding | Deterministic greedy; argmax p; exact fp32 ties → smaller token id; stop = EOS or 512 |
| Lookahead | Every step, all k = 4 candidates, no gate; record nll, q_entropy of future, FR(q_j, q_greedy-future), FR(q_j, consensus), p_value |
| Monitor protocol | Logistic C=1; dev-standardized; dev-median imputation; frozen features; identical code (`analysis/monitor_run.py`, same env-flag model switch) |
| Metrics | AUROC/AUPRC (sklearn-verified), Brier, paired bootstrap, risk–coverage; same seeds |

## Compute / memory feasibility (do-not-start gate)

**Model choice rationale.** Preferred-by-author 7B is **not reliable** on
Kaggle: Qwen2.5-7B-Instruct bf16 weights ≈ 15.2 GB vs 16 GB T4 → no headroom
for per-worker KV (≈0.9 MB/token × 512 tokens × 4-lookahead batches) and
activations; data-parallel 2-worker sharding is impossible (each worker needs
the whole model); pipeline-sharded single worker removes data parallelism and
pushes wall-clock to ≈ 17–20 h > the 12 h Kaggle session cap; 4-bit/8-bit
quantization violates the frozen-numerics constraint. **Decision (subject to
your approval): replicate at Qwen2.5-3B-Instruct (bf16 = 6.2 GB, comfortable
on T4).** 7B becomes viable only on > 24 GB single-GPU or reliable 2×24 GB
classical sharding — not available on this account's free quota.

**Cost estimate (2×T4, measured throughput basis).** Run-2 (1.5B): 3.65 h
for 500 q × 129,839 steps × 5 forwards. FLOPs scale ≈ linearly with params
(3B/1.5B ≈ 2×; attention component slightly superlinear) → est. **7–9.5 h
wall on 2×T4** (fits in one 12 h session with margin; crash-resume is
supported per question as in run 2). GPU-quota cost ≈ 15–19 T4·h.

**Do-not-start conditions:** if a dry-run throughput probe (smoke, below)
shows > 14 h projected wall, or if the model+KV does not fit a T4 in the
smoke, the run is aborted and re-scoped (e.g., 300-question test-only arm
with dev frozen — will be proposed as a NEW protocol revision and simply
relabeled "pilot", never silently mixed into the frozen n=500 claim).

## Smoke test and artifact-integrity gates (before the full run)

1. **Smoke (≈3 min GPU):** 2 questions × 64 tokens → all run-2 gates
   (greedy==chosen row-exact; 4 candidates/step; NaN = 0; forward-call
   arithmetic absolute; rank-0 FR floor < 3e-3; tie handling) — reuse
   `analysis/fetch_and_verify.py` unchanged.
2. **Throughput probe:** wall per 10 questions must imply < 14 h projected.
3. **Cross-session determinism:** smoke re-run hash equality (as in run 2).
4. **Full run integrity (post-run):** 500/500 questions; steps == tokens;
   forward_calls formula; split row-exact; zero failures; candidate row
   count in [equal-expected range] per question; no NaN; md5 manifest of all
   outputs committed with results.
5. **Fresh-model sanity:** 3B must not be degenerate (accuracy between
   60–85% expected; if outside, investigate before analysis — recorded, not
   tuned around).

## Execution plan (after your approval)

1. Commit `experiment/replication_protocol.md` (timestamped).
2. Extend `analysis/monitor_run.py` model env knob (already supported:
   `GEODESIC2_MODEL`) → build kernel `geodesic-repl-3b` from the same
   `analysis/build_kernels.py` path with embedded items (identical code path).
3. Push smoke kernel → verify gates → push full kernel → poll/fetch →
   `analysis/fetch_and_verify.py replication` → `analysis/run2_analysis.py
   --indir … --outdir analysis/outputs2_rep3b` (code identical; seed 42).
4. Write `analysis/replication_report.md`; add H3 section to the paper if
   informative (either direction), with zero post-hoc criterion changes.
5. **Service budget:** ≈ 15–19 Kaggle T4·h + ~1 h overhead. Risk register:
   weekend queue delay; session cap requires resume workflow (part files);
   any resume counts steps to keep forward-call accounting exact.

## What is NOT in scope of this replication (explicit!)

- k-step futures, JS divergence, sequence-level monitors, other datasets,
  temperature anchors, larger k. Those belong to separate preregistrations
  (`experiment/next_experiment_recommendation.md`).
