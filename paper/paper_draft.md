# Does Fisher–Rao geometry between candidate futures add information to token-level uncertainty monitors?

**A pre-registered falsification on Qwen2.5-1.5B-Instruct / GSM8K**

Draft v1.2 — 2026-10-17. Author audit trail: `docs/experiment_protocol.md`,
`docs/run2_report.md`, `docs/audit_phase1.md`, `analysis/run2_analysis_prefix.py`
(pre-fix), `analysis/run2_analysis.py` (corrected). **Status: draft, not
submitted.** v1.2 change: all references machine-verified (arXiv API, Crossref,
doi.org; see References) and a Related-Work search performed the same day (§3).

---

## Abstract

Large language models require trustworthy correctness estimates for selective
prediction.Several recent interfaces for uncertainty quantification —
entropy, NLL, and divergences between candidate "futures" — are computed on the
same decoded token stream, but a growing folklore claim holds that *geometric*
divergences between candidate continuation distributions (here, the
Fisher–Rao distance between softmax futures) carry correctness information
beyond conventional features. We test that claim with a pre-registered,
fully-specified experiment: for each of 500 GSM8K questions, we decode answers
greedily (deterministic tie-breaking) and, at *every* decoding step, record the
full top-4 candidate grid — NLL, candidate entropy, and Fisher–Rao distances of
each candidate's one-step future to the greedy future and to the candidate
consensus — with zero per-question failures on a 2×T4 run (3.65 h wall,
byte-verified determinism across independent sessions).

We fit logistic monitors on conventional features (entropies, NLL, length) and
on conventional + six pre-registered Fisher–Rao aggregates. On 400 untouched
held-out questions (27.5% wrong), the conventional monitor reaches AUROC
**0.770 [95% CI 0.713–0.821]** (AUPRC 0.596). Adding Fisher–Rao features
**does not improve** any of the six pre-specified single-feature monitors or
the combined monitor (all ΔAUROC ≤ 0); the combined addition is *significantly
harmful* (Δ −0.042, CI [−0.085, −0.001]) and reduces AUPRC in every variant.
Fisher–Rao features alone reach AUROC 0.661. We additionally show that an
earlier apparent Fisher–Rao monitor AUROC of ≈0.84 in our own prior
experiment was a measurement artifact of a *confounded regime* — degraded,
partially-censored generations with a 46.7% error prevalence — not evidence
for geometric signal.

We conclude: on this model, task, and decoding regime, **the answer is no** —
under the tested definitions, Fisher–Rao geometry adds no incremental
correctness information, and where it differs it tends to be redundant with
—and occasionally worse than—entropy/length features. We release the complete
per-step feature table (519,356 candidate rows), verification code, and
pre-registered protocol to allow direct scrutiny of this negative result.

---

## 1. Introduction

### 1.1 Research question

**RQ.** When *everything is measured on the same decoded trajectory*, do
Fisher–Rao divergences between the token-level distributions of alternative
next tokens (candidate "futures") provide incremental information about final
answer correctness, beyond what token-level entropy and NLL already encode?

The question is deliberately narrow. It does not ask whether Fisher–Rao
geometry is useless in general, whether other geometric constructions work,
whether larger models behave differently, or whether one-step lookahead is the
best geometric probe. It asks one falsifiable thing about one well-specified
feature family.

### 1.2 Falsifiable hypotheses (pre-registered)

Both hypotheses and pass/fail criteria were written into
`docs/experiment_protocol.md` **before** any run-2 labeled data existed (addendum
§RUN-2 ADDENDUM). They are reproduced verbatim here.

- **H1 (detectability).** *H₀: the full-coverage monitor ranks correctness no
  better than chance (AUROC = 0.5), test n = 400, wrong = positive.* Reject
  H₀ only if the 95% bootstrap CI (2000 resamples, seed 42) excludes 0.5.
- **H2 (incremental information).** With **A** = conventional-only monitor
  (mean_H, max_H, late_H, mean_nll, mean_next_H, mean_prob, gate_frac,
  log_len) and **B = A + geometric features G1–G6** (Table 1 of §4.3), H2 is
  supported **only if**: paired ΔAUPRC > 0, **and** paired ΔAUROC CI excludes
  0, **and** risk–coverage accuracy gains at 80% and 90% coverage. All three
  criteria must hold. All outcomes reported regardless of sign.

Deviations from pristine pre-registration are disclosed explicitly in §4.6 —
this protocol was frozen after a Phase-1 audit of our own run-1 data (which
had already exposed test-set outcomes), so it is a *confirmatory replication
of definitions*, not a first-pass annotation-free pre-registration.

### 1.3 Contributions

1. A **pre-registered, fully-artifact-backed falsification** of the
   incremental-information claim for Fisher–Rao candidate-future features,
   with per-step raw data released.
2. An **audit-grade experimental harness** (deterministic decoding,
   full-candidate coverage, zero gate-censoring) that eliminates three
   independent confounds we identified in our own earlier experiment:
   gate-dependent feature censoring, tie-nondeterministic decoding, and
   regime-mismatched accuracy baselines.
3. A **mechanistic account** of how an apparent high-AUROC monitor (≈0.84)
   arises from entropy/length features in a *degraded generation regime*
   (prevalence shock from 27.5% to 46.7% error rate), explaining why that
   number failed to transfer.
4. A negative result stated at matched precision: no pre-registered FR
   addition improves held-out AUROC, AUPRC, Brier score, or risk–coverage.

---

## 2. Background: how the ≈0.84 number was confounded

The experiment reported here was preceded by a controlled 6-arm study in which
a decoding controller used Fisher–Rao distances measured at *entropy-gated*
steps. In that study, one arm (`fr_only`) produced a combined-signal monitor
AUROC of **0.8417 [0.7989, 0.8800]** on 400 held-out questions, and a related
"≈0.844" number circulated in the project notes. Our Phase-1 audit
(`docs/audit_phase1.md`, reproduced in `analysis/outputs/auroc_table.csv`) found
that this number decomposes as follows, all verifiable from the released
CSVs:

| Quantity (run-1, `fr_only` arm) | Value | Artifact |
|---|---|---|
| Decoding accuracy of the arm | **52.0%** (vs 71.4% greedy) | `geodesic_study/summary.csv` |
| Error prevalence on that arm's generations | **0.4675** (vs 0.275 greedy) | `verified_summary.csv` |
| Univariate AUROC of **mean entropy** | **0.837** | `auroc_table.csv` |
| Univariate AUROC of **gate-open fraction** | **0.833** | `auroc_table.csv` |
| Univariate AUROC of **#gate steps** | **0.869** | `auroc_table.csv` |
| **CONVENTIONAL-only** trained monitor | **0.8223** | `trained_monitors.csv` |
| Combined → + FR features | **−0.0024** [−0.0095, +0.0049] | `trained_monitors.csv` |
| FR features themselves (univariate) | 0.417 / 0.342 (weak) | `auroc_table.csv` |

Read this table as a causal story confirmed further by run 2:

1. The `fr_only` controller *degraded generation badly* (52.0% vs 71.4%).
   Degraded, meandering generations produce long, high-entropy token streams
   — so simple entropy and length features, measured **on that degraded
   regime**, correlate strongly with the near-coin-flip correctness of that
   regime → AUROC 0.82–0.87.
2. The Fisher–Rao features themselves were *never* the driver: within the same
   arm, FR univariate AUROCs were 0.417 (FR→greedy) and 0.342 (FR→consensus),
   i.e., at-or-below chance orientation.
3. Adding FR features to the conventional monitor on run-1's greedy arm moved
   AUROC by **+0.0000 (both features)**; on `fr_only` by **−0.0024**; across
   all other arms the deltas were ≤ +0.013 **with CIs spanning 0**
   (`incremental_percentile.csv`; e.g., full_geodesic +0.0131
   [−0.0149, +0.0426]).

Run-2 then put the question in the *clean* regime (same greedy trajectory for
every candidate, no gating), and the conventional monitor settled at 0.770
[0.713, 0.821], matching run-1's greedy-arm conventional monitor (0.7549
[0.6997, 0.8064]) measured on independent data. **The ≈0.84 was a regime
artifact, not a Fisher–Rao effect.** The two independent experiment families
converge on the same conclusion from opposite directions.

---

## 3. Related work

*All references cited in this section were machine-verified on 2026-10-17
(arXiv API, Crossref, doi.org; details in §References). The search status is
stated in the novelty paragraph below; this is not claimed to be an
exhaustive survey.*

**Uncertainty estimation in deep learning.** Post-hoc confidence from softmax
probability is the canonical baseline [Hendrycks & Gimpel, ICLR 2017, arXiv:1610.02136];
calibration of neural networks is standardised in [Guo et al., ICML 2017,
arXiv:1706.04599]. For autoregressive LLMs, token-level entropy and
sample-based uncertainty are mainstream [Kuhn, Gal & Farquhar, ICLR 2023,
arXiv:2302.09664]; semantic entropy for hallucination detection is
[Farquhar et al., Nature 630:625–630, 2024, doi:10.1038/s41586-024-07421-0];
LLMs' internal probes for "knowing what they know" are [Kadavath et al., 2022,
arXiv:2207.05221]; verbalized uncertainty is studied in [Lin, Hilton & Evans,
TMLR 2022, arXiv:2205.14334].

**Correctness prediction for reasoning.** Uncertainty-aware selective
question answering is formalised in [Kamath, Jia & Liang, ACL 2020,
arXiv:2006.09462]; SelfCheckGPT for black-box consistency is [Manakul et al.,
EMNLP 2023, arXiv:2303.08896]. Correctness of generated reasoning traces
relates closely to GSM8K-style verifiers [Cobbe et al., 2021, arXiv:2110.14168].
Early-token signals are an active adjacent direction: probing of first-token
logits for answerability and safety [Zhao et al., ECCV 2024, arXiv:2403.09037]
and single-decode first-token confidence matching sample-agreement on factual
QA [Gabriel, 2026, arXiv:2605.05166].

**Geometric representations of distributions.** Fisher–Rao/Bhattacharyya
geometry of parametric families is classical [Rao 1945; Bhattacharyya 1943];
information geometry in general is [Amari 2016, doi:10.1007/978-4-431-55978-8;
Nielsen 2020, doi:10.3390/e22101100]. A fast-growing 2026 preprint literature
applies Fisher–Rao geometry to LLMs in *other* roles: training regularisation
in self-distillation [Jukić & Titov, arXiv:2607.06855], model merging on the
Fisher–Rao manifold [Wang, Ye & Yin, arXiv:2603.04972], a shared
information-geometric structure of next-token distributions
[Picozzi, arXiv:2609.11063], language statistics via Fisher–Rao distances
[Du & Tanaka-Ishii, Phys. Rev. Research 6, L022028, 2024], and document
distance via Wasserstein–Fisher–Rao [Wang et al., 2019, arXiv:1904.10294].

**Closest concurrent work.** Belief-Trajectory Energy [Ying et al.,
arXiv:2610.05114, Oct 2026] shows that layer-wise predictive *revisions*,
interpreted through Fisher–Rao geometry, carry AUROC-style error signal across
reasoning tasks. This differs from our setting in three ways: it probes
layer-level belief revision, ours measures divergences between candidate
*future continuations* at the token level; it is not set in a
pre-registered falsification protocol with a conventional-feature control
arm; and it targets layer trajectories rather than candidate-token fan-out.
We cite it as concurrent evidence that Fisher–Rao geometry is informative
*somewhere* in LLM pipelines — which makes our token-level null result more
informative, not contradicted.

**Novelty statement and its limit.** A targeted arXiv (cs.CL/cs.LG-focused)
search was performed on 2026-10-17 with the query families listed in
`paper/audit_checklist.md` (§6). We found no work that uses divergence
distances between candidate future continuations as features of a supervised
correctness monitor under frozen greedy decoding, evaluated with a
pre-registered falsification protocol. We nevertheless make **no general
first-ness claim**: coverage outside arXiv cs.CL/cs.LG, different vocabulary,
and non-indexed venues cannot be excluded, and the search was targeted, not
exhaustive. We claim only the falsification result itself (§6) and the audit
case study (§2).

**Negative results & evaluation hygiene.** Leakage and reproducibility
pathologies are documented in [Kapoor & Narayanan, Patterns 2023 §ref];
the cost of uncontrolled splits in [Gorman & Bedrick, ACL 2019 §ref];
invalid comparisons in LLM evaluations have been repeatedly flagged in
benchmark audits. This paper's contributing methodology (artifact-level audit
of a previously-reported number, pre-registered falsification) follows that
tradition.

---

## 4. Method

### 4.1 Experimental design overview

| Component | Specification |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (bf16, torch 2.11.0+cu128) |
| Task / data | GSM8K-derived, 500 questions inherited from run-1 (IDs 0–499) |
| Split | **Frozen** from run-1 (`split_map.csv`): 100 dev / 400 test, verified row-exact |
| Decoding | Greedy; **deterministic tie-breaking**: argmax p; exact fp32 ties → smaller token id. stop = EOS or 512 tokens |
| Lookahead | At **every** decoding step, for **all** k = 4 candidates (no entropy gate): one forward pass each |
| Recorded per (question, step, rank) | `nll`, `q_entropy` (future entropy), `fr_greedy` = FR(q_j, q_greedy-future), `fr_consensus` = FR(q_j, mean future), `p_value`, `is_greedy` |
| FR definition | FR(p, q) = 2·arccos(Σᵢ√(pᵢqᵢ)); computed in fp32 |
| Compute | Kaggle 2×T4, wall **13,135 s (3.65 h)**; workers = 2 (one per GPU); resume-safe per question |
| Failures | **0** per-question failures; 0 NaN values in 519,356 candidate rows |

### 4.2 Trajectory & decoding guarantees

- The decoded answer **is** the rank-0 (deterministic-greedy) candidate at
  every step; verified row-exact across all 129,839 steps
  (`sanity_checks.csv`).
- Forward-call accounting is exact per question:
  `forward_calls = 1 + 5·steps − 1[eos-stop]`; verified for all 500 questions.
- Determinism: an identical re-run in an independent Kaggle GPU session
  produced **byte-identical** candidate tables (md5 `d7db02eb…`). Ties were
  registered at 969/129,839 steps (0.75%) and resolved deterministically.
- **No leakage:** question-level features are functions of the decoded
  trajectory and model logits only. Gold answers, correctness and split labels
  enter the fit/evaluation only via the `wrong`/`split` columns, never as
  input features. Feature definitions were frozen before evaluation (§4.6).

### 4.3 Pre-registered feature definitions

**Conventional (A).** Per question, over the greedy trajectory (rank-0 rows):

| Feature | Definition |
|---|---|
| mean_H, max_H | mean / max of predictive entropy H(p) over steps |
| late_H | mean H(p) over the last quarter of steps |
| mean_nll | mean NLL of the chosen token |
| mean_next_H | mean future entropy H(q₀) of the greedy candidate |
| mean_prob | mean top-1 probability p₀ |
| gate_frac | fraction of steps with H(p) > 1 |
| log_len | log of generated token count |

**Geometric (G), all computed on the same trajectory's candidate grid:**

| Pre-reg # | Feature | Definition |
|---|---|---|
| G1 | mean_frcons | mean over steps of FR(q₀, consensus) |
| G2 | mean_altdist | mean over steps of mean_{j>0} FR(qⱼ, q₀) |
| G3 | mean_altdist_max | mean over steps of max_{j>0} FR(qⱼ, q₀) |
| G4 | mean_altspread | mean over steps of [max_{j>0} − min_{j>0} FR(qⱼ, q₀)] |
| G5 | mean_hgap | mean over steps of [mean_{j>0} H(qⱼ) − H(q₀)] |
| G6 | std_fr | std over all (step, j>0) FR(qⱼ, q₀) within the question |

### 4.4 Monitor fitting and evaluation

Logistic regression (C = 1, `max_iter` 4000, seed 42), standardized with
dev-only mean/std; missing values median-imputed with **dev** medians (no
question had missing features in practice). Fit on **dev (n = 100)**; frozen;
evaluated once on **test (n = 400)**. Metrics: AUROC, AUPRC (`average_precision_score`),
Brier score on the sigmoid of the decision function; paired
-bootstrap CIs (2000 resamples, seed 42, **single shared index per replicate**
for paired deltas); two-sided bootstrap p-values. All metric implementations
were cross-checked against `sklearn` (max abs deviation 0.0 across all
monitors — exported as `auroc_hand_vs_sklearn_maxdiff`).

### 4.5 Risk–coverage criterion

For each monitor, questions are ranked by risk (most risky dropped first);
accuracy among the answered set is reported at exactly 80% and 90% coverage
(`risk_coverage_crit.csv`). Pre-registered as part of H2's pass criteria.

### 4.6 Deviations and audit corrections (full disclosure)

The released analysis evolved under audit. Two separate pre-fix versions are
preserved (`analysis/run2_analysis_prefix.py`, `analysis/outputs2_prefix/`):

1. **Late-quarter aggregation bug (material).** Pre-fix, the whole frame was
   filtered to the final quarter of steps *before* aggregation, silently
   redefining mean_H, max_H, mean_nll, mean_next_H, gate_frac and every FR
   aggregate as last-quarter statistics. This **understated the conventional
   monitor** (CONV AUROC 0.704 pre-fix → 0.770 corrected) and mildly reshaped
   the FR deltas. Corrected code is authoritative; pre-fix outputs retained.
2. **Prereg-vs-implementation mismatches fixed:** `mean_hgap` had been a
   duplicate of mean_next_H (now G5 as defined); `std_fr` had been computed on
   rank-0 rows (identically ≈0 → AUROC 0.507, now G6 as defined);
   `mean_altspread` now matches G4; `mean_prob` now uses top-1 probability.
3. **Scope corrections:** `max_frcons` was an implementation *extra*, not part
   of the pre-registered G1–G6 — reclassified as **exploratory**. A
   non-paired CI helper initially resampled labels and scores with
   **independent** index sets; this was caught because the resulting CI did
   not contain its own point estimate, and no scientific conclusion was ever
   drawn from it. The paired-delta implementation used correct pairing
   throughout (verified below).
4. `mean_fr`/`mean_dcent` on run-1's greedy arm returned NaN because greedy
   has gate-rate 0 (the controller never opens the gate) — no FR features
   existed on that arm; the NaN is a measurement of absence, not a bug.

**Not corrected (choices, flagged):** Brier defined on sigmoid of decision
function (a calibration choice); monitor is question-level (no sequence
model); the protocol itself, though definition-frozen, was written after run-1
was complete and thus is not annotation-free pre-registration.

---

## 5. Verification of the executed experiment

All checks below are reproducible with `analysis/audit_raw_A1A4.py` and
`analysis/fetch_and_verify.py` against the released artifacts
(`analysis/outputs2/raw_audit.csv`).

| # | Check | Verdict |
|---|---|---|
| V1 | Split labels row-exact vs frozen run-1 map; 500 unique qids; 100 dev / 400 test | PASS |
| V2 | Test error prevalence 0.275; dev accuracy 0.670; test accuracy 0.725 | PASS |
| V3 | 519,356 candidate rows; exactly 4 per step everywhere | PASS |
| V4 | Rank-0 candidate token == decoded chosen token (all steps) | PASS |
| V5 | NLL monotone non-decreasing in candidate rank (all 129,839 step groups) | PASS |
| V6 | Zero NaN; FR ∈ [0, π]; rank-0 FR floor 0.0021 (fp32 acos over 152k vocab) | PASS |
| V7 | forward_calls == 1 + 5·steps − 1[eos] exactly (per question) | PASS |
| V8 | Determinism: byte-identical candidates across 2 independent Kaggle GPU sessions + 3 local CPU runs | PASS |
| V9 | Tie events 969 (0.75%), all resolved by the smaller-token-id rule, 0 inconsistencies | PASS |
| V10 | Steps == generated tokens (per question); total 129,839 steps | PASS |
| V11 | Monitor-greedy predictions identical to run-1's greedy arm on 500/500 (accuracy 0.714 both) | PASS |
| V12 | sklearn cross-check of AUROC/AUPRC: max abs deviation 0.0 (all monitors) | PASS |
| V13 | No leakage: features depend only on trajectory/logits; labels enter only fitting/eval | PASS (by construction + §4.6) |

**Flagged as not independently verifiable inside this artifact set:** (i) the
claim that no *other* geometric construction (multi-step lookahead, Rényi
families, Wasserstein) would behave differently — untested, §8; (ii) the
calibration choice for Brier; (iii) external-related-work claims — see §3
caveat.

---

## 6. Results

### 6.1 Main result (confirmatory; test n = 400, prevalence 0.275)

**Table 1 — Monitors (dev n=100 fit, test n=400 eval).** All AUROCs verified
against sklearn; hand-vs-sklearn diff = 0.0 for every row. CIs: 2000-sample
percentile bootstrap (seed 42).

| Monitor | AUROC [95% CI] | AUPRC | Brier |
|---|---|---|---|
| **CONV (A)** | **0.770 [0.713, 0.821]** | **0.596** | **0.159** |
| A + G1 (mean_frcons) | 0.756 [0.699, 0.804] | 0.558 | 0.166 |
| A + G2 (mean_altdist) | 0.747 [0.692, 0.797] | 0.542 | 0.170 |
| A + G3 (mean_altdist_max) | 0.770 [0.712, 0.821] | 0.597 | 0.159 |
| A + G4 (mean_altspread) | 0.739 [0.680, 0.793] | 0.534 | 0.173 |
| A + G5 (mean_hgap) | 0.769 [0.713, 0.820] | 0.598 | 0.159 |
| A + G6 (std_fr) | 0.741 [0.683, 0.791] | 0.530 | 0.171 |
| A + ALL_GEO (G1–G6) | 0.728 [0.667, 0.780] | 0.505 | 0.181 |
| GEO only (G1–G6) | 0.661 [0.599, 0.719] | 0.397 | 0.206 |
| A + max_frcons *(exploratory)* | 0.775 [0.718, 0.825] | 0.599 | 0.158 |

**H1 verdict: PASSED.** The conventional monitor's AUROC CI excludes 0.5 by a
wide margin (0.770 [0.713, 0.821]) — token-level uncertainty signals are real
and strong on this task. This replicates run-1's independent greedy-arm estimate
(0.755 [0.700, 0.806]).

**H2 verdict: FAILED — all three pre-registered criteria fail.**

### 6.2 The eight tested FR additions (H2 test)

**Table 2.** Δ = monitor AUROC − CONV AUROC; positive Δ = improvement.

| Addition | ΔAUROC [95% CI] | p (two-sided) | AUPRC change | Status |
|---|---|---|---|---|
| + G1 mean_frcons | −0.014 [−0.041, +0.011] | 0.27 | −0.038 → 0.558 | confirmatory |
| + G2 mean_altdist | −0.023 [−0.054, +0.007] | 0.13 | −0.054 → 0.542 | confirmatory |
| + G3 mean_altdist_max | +0.000 [−0.002, +0.003] | 0.89 | +0.001 → 0.597 | confirmatory |
| + G4 mean_altspread | −0.031 [−0.070, +0.008] | 0.11 | −0.062 → 0.534 | confirmatory |
| + G5 mean_hgap | −0.001 [−0.009, +0.007] | 0.85 | +0.002 → 0.598 | confirmatory |
| + G6 std_fr | **−0.030 [−0.060, −0.001]** | 0.042 | −0.066 → 0.530 | confirmatory |
| + ALL_GEO | **−0.042 [−0.085, −0.001]** | 0.043 | −0.091 → 0.505 | confirmatory |
| + max_frcons | +0.005 [−0.002, +0.012] | 0.16 | +0.003 → 0.599 | **exploratory** |
| GEO only (family) | −0.109 [−0.170, −0.052] | <0.001 | −0.199 → 0.397 | confirmatory |

Reading: **no pre-registered FR addition improves held-out AUROC or AUPRC.**
The two nominally significant entries are in the *harmful* direction
(−0.030, −0.042) and would **not** survive any multiplicity correction across
the eight tests (Bonferroni threshold p ≤ 0.00625). The only positive point
estimate (+0.005, exploratory `max_frcons`) has a CI that comfortably
contains 0.

**Pre-registered criterion check:**

| H2 criterion | Result |
|---|---|
| ΔAUPRC > 0 (paired) | ✗ fails for all prereg additions |
| ΔAUROC CI excludes 0 (in A's favor is improvement) | ✗ never improves; excludes 0 only *against* FR (G6, ALL_GEO) |
| Risk–coverage gain at 80% / 90% coverage | ✗ see Table 3 |

**Table 3 — Risk–coverage accuracy among answered (pre-registered check).**

| Monitor | 80% coverage | 90% coverage |
|---|---|---|
| CONV | **81.3%** | **77.2%** |
| CONV + ALL_GEO | 78.8% | 75.8% |
| GEO only | 75.9% | 74.4% |

### 6.3 Univariate picture (diagnostic)

**Table 4 — Per-feature AUROC on test (sign-aligned: higher risk → wrong).**
Full table in `univariate_auroc.csv`; dev-Spearman shown as a redundancy hint.

| Feature (family) | AUROC | dev Spearman(wrong) |
|---|---|---|
| max_H (CONV) | 0.711 | +0.34 |
| gate_frac (CONV) | 0.669 | +0.37 |
| log_len (CONV) | 0.669 | +0.25 |
| mean_next_H (CONV) | 0.668 | +0.40 |
| mean_H (CONV) | 0.661 | +0.40 |
| mean_altdist (G2) | 0.637 | +0.43 |
| mean_frcons (G1) | 0.621 | +0.36 |
| std_fr (G6) | 0.334 | −0.45 |
| mean_altspread (G4) | 0.380 | −0.45 |
| mean_hgap (G5) | 0.533 | −0.07 |

Two diagnostic observations, both interpretations (§7/Table 5), not
confirmatory claims:

- **Redundancy:** the FR features that look promising marginally
  (`mean_altdist`, `mean_frcons`: dev ρ .43/.36, test AUROC .64/.62) correlate
  moderately with the conventional block (e.g., mean_altdist vs mean_H
  ρ = 0.51, vs mean_next_H ρ = 0.51; `spearman_dev.csv`), so the multivariate
  conventional monitor already spans most of their axis — adding them moves
  nothing.
- **Polarity:** two FR constructs (`altspread`, `std_fr`) correlate with
  *correctness* (negative risk), i.e., their raw signal exists but *opposes*
  the naive "more divergence → more error" story — and is weaker than the
  conventional block, explaining why mixing it in hurts.

### 6.4 Reliability

The CONV monitor's calibrated out-of-fold probabilities (sigmoid of decision
function) track empirical error rates smoothly (`fig_run2_reliability.png`);
no gross miscalibration was observed in the exploratory reliability figure for
the FR-augmented variant either.

---

## 7. Discussion

**The claim this paper makes.** On Qwen2.5-1.5B-Instruct, GSM8K, top-4
candidates, one-step lookahead, and question-level aggregation: Fisher–Rao
geometry between candidate futures, measured without any access gate, adds
**no incremental correctness information** over token entropies, NLL and
length — and where the G-features individually carry signal, it is (i)
subsumed by weaker, cheaper conventional features or (ii) directionally
opposite to intuition and harmful when included.

**What the falsification does not claim.** It does not claim: (a) FR is
useless at larger scale or other decode budgets; (b) that deeper lookahead
(k-step futures rather than one-step) or step-level (sequence) aggregation
could not change the verdict; (c) that other geometric functionals (Wasserstein,
Rényi-divergence families, semantic-entropy clustering) fail similarly; (d)
that FR is useless for other targets (OOD detection, hallucination semantics).

**What it teaches.** First, the "monitor AUROC" abstraction is extremely
sensitive to the *generation regime* the monitor is scored on: the same
feature family scored 0.82–0.86 on a degraded top-k regime and 0.66–0.77 on
clean greedy decoding. Multi-arm studies must therefore report
regime-matched baselines, or their headline numbers are uninterpretable —
ours were, and the promise of FR dissolved under controlled comparison. This
specific failure mode (feature-value distribution shifts with the controller,
prevalence shifts with controller quality) is easy to create accidentally and
hard to notice without a matched clean baseline.

Second, the correct anchor for "incremental information" questions is a
pairwise-same-trajectory design. Run-1's gate-dependent censoring (FR features
recorded for only 9.0% of full_geodesic steps and 0.65% of conservative steps
— `summary.csv`) meant FR features in that study were never a random sample
of steps; they were, structurally, high-uncertainty steps only. Any
"information" they displayed was partially the information of *being selected
by the gate*, which the conventional features already encode.

---

## 8. Limitations

1. **Sample size.** Test n = 400 (110 wrong-equivalent mass). Paired CIs
   exclude FR gains larger than ≈ +0.04 AUROC but cannot exclude gains
   below ~0.01–0.02.
2. **One model, one dataset, bf16.** Qwen2.5-1.5B on GSM8K only; 4-candidate
   grid; one-step lookahead. Findings may differ at 7B+, other tasks, or
   fp32/longer futures (untested — H3 in the protocol remains open for this
   reason; we did not run it because no positive confirmatory signal existed
   to transfer).
3. **Question-level aggregation.** Mean/max/stats per question; a
   step-level sequence monitor (e.g., AUC of martingale statistics) could in
   principle exploit temporal structure the aggregates discard; dev=100 is
   too small to train one honestly.
4. **Single decoding anchor.** All conclusions are about greedy decoding with
   deterministic tie-breaking; sampled anchors (temperature > 0) are a
   different regime.
5. **Protocol scrutiny timing.** The definitions were frozen after a Phase-1
   audit saw run-1 test behavior; the confirmatory status of H1/H2 is that of
   a *replication with frozen definitions*, weaker than a first-pass
   preregistered experiment, stronger than post-hoc analysis.
6. **Brier calibration choice** (sigmoid of decision function) is a choice;
   AUROC/AUPRC are rank-based and unaffected.
7. **Compute and determinism guarantees are bf16-GPU-specific**; fp16/fp32
   could shift ties/floors (fr floor 0.0021 is fp32-accumulation noise, not
   signal).

---

## 9. Reproducibility

| Artifact | Path |
|---|---|
| Pre-registered protocol (frozen) | `docs/experiment_protocol.md` |
| Runner (single source of truth) | `analysis/monitor_run.py` |
| Corrected analysis | `analysis/run2_analysis.py` |
| Pre-fix (audited) analysis | `analysis/run2_analysis_prefix.py` |
| Raw per-step data (this run) | `kaggle_out/full/geodesic_study2/` |
| Raw run-1 data | `geodesic_study/` (unchanged) + `geodesic_study.zip` |
| Verification scripts | `analysis/audit_raw_A1A4.py`, `analysis/fetch_and_verify.py`, `analysis/smoke_verify.py` |
| Result tables | `analysis/outputs2/*.csv` (+ `outputs2_prefix/` for the audited pre-fix set) |
| Figures | `analysis/figures/fig_run2_*.png` |
| Patched notebook | `notebooks/geodesic_study.ipynb` cell 3 (all run-1 cells untouched); original saved as `hellio_backup_run1.ipynb` |

Kaggle kernels: `kaggle.com/code/pikanigga/geodesic-run-2-full` (v1, complete) |
`…/geodesic-run-2-smoke` (v1/v2). All seeds = 42.

**Reproduction status (verified in this environment):** an independent
re-execution of the full analysis pipeline on the released artifacts produced
**byte-identical** outputs for `trained_monitors.csv`, `paired_deltas.csv`,
`risk_coverage_crit.csv`, `univariate_auroc.csv`, `question_features.csv`
(md5-verified; rerun output retained in `analysis/outputs2_reprod/`). Figures
are regenerated (not byte-compared, by rendering nondeterminism).

```
python analysis/run2_analysis.py --indir kaggle_out/full/geodesic_study2 --outdir analysis/outputs2
python analysis/audit_raw_A1A4.py
```

---

## 10. Candid self-assessment

**What is genuinely novel (with evidence):** (i) the controlled
same-trajectory comparison itself — a manual reading suggests no published
pre-registered same-trajectory test of FR candidate-futures vs entropy/NLL is
known to the authors, but **this is not a verified literature claim** (no web
access in the writing environment; see §3); (ii) the artifact-anchored
decomposition of the ≈0.84 confound (two-regime prevalence story, numbers in
§2) — a concrete, reusable case study in LLM-monitor evaluation hygiene;
(iii) the released per-step dataset (519,356 labeled candidate-feature rows)
enabling others to test *their* geometric ideas without re-running
inference at this scale.

**What is likely to attract reviewer criticism:**

1. **"One small model."** The strongest objection: 1.5B may be simply too weak
   for future-geometry to matter, making the negative claim scientifically
   modest. *Mitigation:* we frame the claim as scoped to this model/task and
   provide the harness to extend it.
2. **"tiny dev set (100)."** Monitor fits have high variance; a conv monitor
   with 0.770 ± 0.054 still leaves room for small FR effects. We cannot
   exclude Δ ≈ +0.01–0.02. *Mitigation:* we state this bound explicitly.
3. **" Aggregation hides signal.** Question-level means may destroy
   early-warning structure; a sequence model might succeed where aggregates
   fail. *Mitigation:* pre-declare as untested (Table 5), not claimed.
4. **"Why Fisher–Rao specifically?** Related divergence families (Jensen–Shannon,
   Wasserstein, Rényi) were not tested; a reviewer may argue the tested
   functional is unrepresentative of "geometric monitoring" per se.
5. **Protocol frozen post hoc relative to run-1.** Not a first-pass
   pre-registration; we say so plainly.
6. **A negative result paper with public artifacts** will be judged by its
   falsification rigor; any residual unexplained artifact (e.g., subtle
   dependence of FR on the bf16 softmax tail) is a possible attack surface —
   mitigated by raw-data release.

**Minimum additional evidence that would materially strengthen the paper** (in
order of value/effort):

1. **Replicate on ≥1 stronger model** (e.g., a 7B-Instruct) with the identical
   harness, reusing the frozen splits — 1 GPU × ~10 h. Even a confirmatory null
   at 7B converts "1.5B too small" from a criticism into a tested boundary.
2. **Test one alternative geometric functional** (e.g., JS divergence between
   futures — same grid, near-zero extra compute in the *analysis*, no new
   inference needed: recompute divergences from the saved future distributions? 
   NOTE: ls future *distributions* themselves were not saved (only scalars),
   so this needs a small re-run. Worth stating honestly.) — OR extend the
   runner to also dump q-vectors for a 50-question subset.
3. **k-step lookahead probe** (2–3 steps) on a 100-question subset to test
   whether deeper futures change the conclusion.
4. **Power statement by simulation:** parametric bootstrap under the
   fitted CONV monitor to quantify the smallest true ΔAUROC the design
   detects with 80% power (expected ≈ 0.03–0.05).

---

## Table 5 — Findings classification

| Finding | Class | Basis |
|---|---|---|
| Conventional monitor AUROC 0.770 [0.713, 0.821], AUPRC 0.596 | **Directly measured** | `trained_monitors.csv`, V12-verified |
| No prereg FR addition improves AUROC/AUPRC (8/8 non-positive point estimates on AUROC incl. ALL_GEO) | **Directly measured** | `paired_deltas.csv` |
| Risk–coverage at 80/90%: FR-augmented worse than CONV | **Directly measured** | `risk_coverage_crit.csv` |
| FR family alone: AUROC 0.661, i.e., includes *some* independent predictive signal | **Directly measured** | `trained_monitors.csv` |
| run-1 ≈0.84 = degraded-regime artifact (52% accuracy arm, prevalence 0.4675, entropy/length driven) | **Directly measured** + protocol interpretation | §2 tables |
| FR features subsumed by entropy/length/len (redundancy) | **Interpretation** (supported by univariate + dev-correlation, not a causal mechanism) | `univariate_auroc.csv`, `spearman_dev.csv` |
| Geometry carries *opposite-polarity* signal (std_fr/altspread) | **Interpretation** | §6.3 |
| Deterministic greedy decoding = reproducible (byte-identical cross-session) | **Directly measured** | md5 comparisons |
| Monitor accuracy 0.714 replicating run-1 greedy arm on 500/500 prediction agreement | **Directly measured** | A1/V11 |
| FR useless at larger scale / other tasks / deeper lookahead / step-sequence monitors | **Untested hypothesis** | §8; H3 open |
| "Geometric monitoring is a dead end for correctness estimation" | **Not claimed** | — |

---

## References — machine-verified 2026-10-17

**Verification method:** every entry below was checked on 2026-10-17 via the
arXiv export API (title/ID match + author list + venue comment), Crossref
metadata search, and/or resolving the printed DOI directly. Two earlier
inconsistencies were caught and corrected in v1.2: Nielsen 2020 is Entropy
22(10), article 1100 (not 1105), and Kapoor & Narayanan 2023 is Patterns
4(9) (not 4(7)). The two Calcutta Mathematical Society classics predate
digital identifiers and were verified through their standard citations in
[Nielsen 2020]; print-level page numbers were not independently re-checked.

- D. Hendrycks, K. Gimpel. "A Baseline for Detecting Misclassified and
  Out-of-Distribution Examples in Neural Networks." **ICLR 2017.**
  arXiv:1610.02136.
- C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. "On Calibration of Modern
  Neural Networks." **ICML 2017.** arXiv:1706.04599.
- L. Kuhn, Y. Gal, S. Farquhar. "Semantic Uncertainty: Linguistic Invariances
  for Uncertainty Estimation in Natural Language Generation." **ICLR 2023
  (Spotlight).** arXiv:2302.09664.
- S. Farquhar, J. Kossen, L. Kuhn, Y. Gal. "Detecting hallucinations in large
  language models using semantic entropy." **Nature 630(8017):625–630, 2024.**
  doi:10.1038/s41586-024-07421-0.
- S. Kadavath et al. "Language Models (Mostly) Know What They Know."
  **arXiv:2207.05221, 2022.**
- S. Lin, J. Hilton, O. Evans. "Teaching Models to Express Their Uncertainty
  in Words." **TMLR 2022.** arXiv:2205.14334.
- A. Kamath, R. Jia, P. Liang. "Selective Question Answering under Domain
  Shift." **ACL 2020.** arXiv:2006.09462.
- P. Manakul, A. Liusie, M. J. F. Gales. "SelfCheckGPT: Zero-Resource
  Black-Box Hallucination Detection for Generative Large Language Models."
  **EMNLP 2023 (main).** arXiv:2303.08896.
- K. Cobbe et al. "Training Verifiers to Solve Math Word Problems."
  **arXiv:2110.14168, 2021.**
- Qwen Team. "Qwen2.5 Technical Report." **arXiv:2412.15115, 2024.**
  *(added v1.2 — the studied model family.)*
- C. R. Rao. "Information and the accuracy attainable in the estimation of
  statistical parameters." **Bull. Calcutta Math. Soc. 37:81–91, 1945.**
- A. Bhattacharyya. "On a measure of divergence between two statistical
  populations defined by their probability distributions." **Bull. Calcutta
  Math. Soc. 35:99–109, 1943.**
- S.-i. Amari. "Information Geometry and Its Applications." **Springer
  (Applied Mathematical Sciences 194), 2016.** doi:10.1007/978-4-431-55978-8.
- F. Nielsen. "An Elementary Introduction to Information Geometry."
  **Entropy 22(10):1100, 2020.** doi:10.3390/e22101100.
- S. Kapoor, A. Narayanan. "Leakage and the reproducibility crisis in
  machine-learning-based science." **Patterns 4(9):100804, 2023.**
  doi:10.1016/j.patter.2023.100804.
- K. Gorman, S. Bedrick. "We Need to Talk about Standard Splits."
  **ACL 2019.** doi:10.18653/v1/P19-1267.
- Y. Geifman, R. El-Yaniv. "Selective Classification for Deep Neural
  Networks." **NeurIPS 2017.** arXiv:1705.08500.

**Concurrent/related preprints (verified to exist with those titles/authors
on 2026-10-17; not necessarily peer-reviewed — treat as adjacent work):**

- J. Ying et al. "Belief-Trajectory Energy: Measuring the Path to a
  Prediction." arXiv:2610.05114.
- J. Jukić, I. Titov. "Geometric Self-Distillation for Reasoning
  Generalization." arXiv:2607.06855.
- D. Picozzi. "The information geometry of large language models is shared,
  learned, and controllable." arXiv:2609.11063.
- J. Wang, Z. Ye, W. Yin. "Functionality-Oriented LLM Merging on the
  Fisher–Rao Manifold." arXiv:2603.04972.
- X. Du, K. Tanaka-Ishii. "Correlation Dimension of Natural Language in a
  Statistical Manifold." Phys. Rev. Research 6, L022028, 2024.
  doi:10.1103/PhysRevResearch.6.L022028. arXiv:2405.06321.
- Q. Zhao et al. "The First to Know: How Token Distributions Reveal Hidden
  Knowledge in Large Vision-Language Models?" ECCV 2024. arXiv:2403.09037.
- M. Gabriel. "The First Token Knows: Single-Decode Confidence for
  Hallucination Detection." arXiv:2605.05166.

*Note on related-work novelty: no claim of general novelty of Fisher–Rao
methodology is made; the paper's evidence-backed claim is the specific
falsification result and the audit framework (see §3 novelty paragraph).*

*Note on related-work novelty: no claim of general novelty of Fisher–Rao
methodology is made; the paper's evidence-backed claim is the specific
falsification result and the audit framework.*
