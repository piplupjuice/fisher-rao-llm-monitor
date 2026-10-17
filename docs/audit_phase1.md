# Audit Report — Geodesic Decoding Study (Phase 1)

Study: 6-method controlled decoding comparison, Qwen2.5-1.5B-Instruct, 500 GSM8K questions,
frozen 100 dev / 400 test split (seed 42). Code: `notebooks/geodesic_study.ipynb` (2 cells). Raw data:
`geodesic_study/` (verified byte-list identical to `geodesic_study.zip`).
Audit code: `analysis/audit_analysis.py` (+ probes `probe_*.py`). All randomness seeded (42).
**No raw file was modified.** Corrected/derived tables live in `analysis/outputs/`.

## 1. Data inventory & structural verification (all PASS)

| Check | Result |
|---|---|
| predictions.csv rows | 3,000 = 6 methods × 500 questions ✔ |
| Every cell: dev=100 / test=400 predictions | exact ✔ |
| Same question set (ids 0–499) across all 6 methods | identical ✔ |
| Duplicate (question_id, method) rows | 0 ✔ |
| Merged token_traces.csv == de-duplicated part0+part1 | identical ✔ |
| Trace rows == generated token counts for every (q, method) | 0 mismatches ✔ |
| summary.csv / summary_by_split.csv accuracies recomputed from raw | 0 diffs ✔ |
| paired_comparison.csv vs predictions pivot | identical ✔ |
| Unparseable final answers (empty `prediction`) | 0 of 3,000 ✔ |
| NaN golds | 0 ✔ |

Truncation (stopped_on = token_limit, not EOS): greedy 5, nll_only 5,
full_geodesic 7, conservative 5, entropy_only 10, fr_only 36 of 500. fr_only's
36 truncations co-occur with its divergent decoding (see §3.6).

## 2. What is valid in the existing experiment

- The design is genuinely controlled: one frozen seed-42 split saved **before**
  test evaluation; identical questions, prompt, parser, token budget, stopping
  and evaluation for all six methods; inferential stats confined to test.
- Accuracy numbers are correct as computed at runtime (see caveat §3.1).
- The M0↔M1 sanity check did its job — it exposed a real property of the
  pipeline (§3.5) that a naive run would have hidden.
- Length/cost bookkeeping (forward calls, seconds, GPU MB) is consistently recorded.
- Paired comparisons (same question both arms) vs greedy, with bootstrap CI +
  exact McNemar, are the right statistics; verified recomputation matches:
  full_geodesic +1.25 pp, CI ≈ [−1.5, +4.0], McNemar p = 0.4869; conservative
  +1.00 pp (p = 0.557); nll_only +1.00 (p = 0.541); entropy_only −5.75 pp
  (p = 0.022); fr_only −19.25 pp (p < 0.0001).

## 3. Bugs, leakage risks, and statistical issues

3.1 **predictions.csv serializes predictions as floats** (`18` → `18.0`) because
`read_parts` re-reads the part CSVs with pandas type inference before the final
merge. The stored `correct` column was computed on true strings at runtime, so
**accuracy values are immune**, but any downstream re-audit doing naive string
exact-match will score ~40% of rows "wrong". No answer-precision loss was found
(all prediction values survived round-trip). Corrected in the audit by
re-normalizing both sides (`analysis/audit_analysis.py::norm_num`).

3.2 **The reported "Fisher–Rao AUROC 0.844" is not a Fisher–Rao monitor.**
The notebook's AUROC feature list contains no FR feature. The quoted numbers
(0.844 / 0.737 / 0.716 / 0.718) match the **COMBINED monitor AUROC per decoding
method** — recomputed here: fr_only 0.842, entropy_only 0.731, nll_only 0.715,
full_geodesic 0.714, conservative 0.712, greedy 0.712. The 0.844 is `fr_only`'s
combined monitor — and its dominance comes from entropy-family features
(mean_H 0.837, mean_nll 0.826, max_H 0.815) measured **on fr_only's own
degenerated generations** (test accuracy 53.25%, prevalence of wrong answers
0.47 vs ~0.27 for greedy-class methods). A monitor whose input distribution is
shifted by decoding damage is not evidence that FR geometry carries
correctness information. Transfer test (`monitor_transfer.csv`): monitor
trained on fr_only generations achieves only 0.733 on greedy generations —
worse than a plain conventional monitor trained and tested on greedy (0.755).

3.3 **Geometric features are recorded only on gate-open steps.** In all traces,
`fr / d_centroid / next_entropy` are populated exactly on steps where
`H(p) > tau` (mean gate rates from summary.csv: full_geodesic 9.0%,
entropy_only 10.6%, fr_only 22.6%, conservative 0.65% — and 164/400
conservative test questions received **zero** lookahead rows). Any "FR score" therefore conditions on the subpopulation
of steps where the model already hesitates — a selection bias; and on the two
methods one would actually deploy (greedy, conservative) geometric features are
essentially absent. `n_FRsteps`-style availability itself correlates with
wrongness, so averaging FR over those steps cannot be compared across methods.

3.4 **FR-feature AUROCs are weak on their own terms** (test split, wrong=positive:
mean_fr 0.42–0.64; mean_dcent 0.34–0.66 depending on method; the notebook-style
COMBINED of full_geodesic 0.714 is carried by mean/max entropy, not geometry).

3.5 **Tie-breaking nondeterminism breaks "nll_only ≡ greedy".** At flat
distributions the fp32 probability of two tokens is exactly equal
(e.g. q0/step228: NLL 0.918636 for both 10657 and 20517); `torch.topk` returns
tied tokens in non-deterministic order across separate runs, so the nll_only
run and the greedy run — identical algorithms — produce different outputs:
258/500 questions diverge token-wise (median first divergence step 50), and the
changed prefix changes 49 final answers. greedy vs nll_only accuracy differs by
0.2 pp on 500 questions. Consequence: single-run method deltas of ≈ ±1 pp are
the same order as pure tie-flip noise, and each method's number is one sample
from a tie-stochastic process. The notebook's +1.25 pp must be read against
this floor.

3.6 **Conservative method almost never engages geometry**: gate rate 0.65%
(τ = 2.0 nats vs typical conditional entropy ≪ 2). Conservative ≈ greedy plus
rare forced deviations — it does not meaningfully test the geometric theory.

3.7 **Calibration statistics are not calibration.** AUROC/Brier were computed on
a percentile-rank mixture score (rank-scaled [0,1] with NaN → 0.5); Brier values
on such an arbitrary monotone transform are not proper scoring (the notebook's
`brier()` even re-normalizes by observed min/max). Treat reliability claims as
invalid until probabilities from a real model are used.

3.8 **No label leakage found.** All monitor features derive from model logits
during generation only; gold answers enter solely as the outcome; the split was
frozen before test evaluation; dev/test modeling in the audit itself used dev
(100 q) only for fitting. The COMBINED score's use of `gate_frac` as a
"feature" does, however, make it partially a function of which questions the
gate never triggered on — another reason it is not "Fisher–Rao".

3.9 Minor: prompt uses a right single quote `’` (Janet’s) which appears
mojibake-corrupted (`Janet?Ts`) in the saved UTF-8 CSV on Windows readers —
haven-of-gold unaffected, but questions text contains encoding artifacts.

## 4. Recomputed result tables (Phase 1 tasks 6–9)

All tables in `analysis/outputs/`:
- `monitor_features.csv` — per (method, question) aggregates incl. dev split.
- `auroc_table.csv` — AUROC/AUPRC/Brier + 95% bootstrap CIs (2000 resamples,
  seed 42) for all 13 candidate scores × 6 methods, test n=400. Headline subset
  (AUROC, wrong = positive class):

| Score | greedy | full_geodesic | conservative | fr_only |
|---|---|---|---|---|
| mean chosen-token NLL | 0.643 | 0.644 | 0.635 | 0.826 |
| mean entropy H(p) | 0.661 | 0.661 | 0.652 | 0.837 |
| max entropy | 0.711 | 0.712 | 0.726 | 0.815 |
| mean FR dist → greedy future | — | 0.536 | 0.490 | 0.417 |
| mean FR dist → consensus | — | 0.410 | 0.432 | 0.342 |
| conventional percentile combo | 0.691 | 0.701 | 0.699 | 0.840 |
| notebook COMBINED combo | 0.712 | 0.714 | 0.712 | 0.842 |
| *trained* logistic (CONV features, dev-fit) | **0.755** [0.700, 0.806] | 0.756 [0.698, 0.809] | 0.760 [0.704, 0.813] | 0.822 [0.777, 0.863] |
| *trained* logistic (CONV + FR) | — (no FR) | 0.769 [0.710, 0.820] | 0.754 [0.696, 0.810] | 0.820 [0.774, 0.861] |
| *trained* logistic (FR only) | — (no FR) | 0.765 [0.708, 0.817] | 0.720 [0.660, 0.779] | 0.863 [0.823, 0.899] |

- `trained_monitors.csv` — dev(100 q)-fitted logistic monitors, test-evaluated;
  paired AUROC deltas CONV+GEOM − CONVENTIONAL: full_geodesic +0.013
  [−0.015, +0.043]; conservative −0.006 [−0.027, +0.012]; fr_only −0.002
  [−0.010, +0.005] → **no detectable incremental information from FR geometry**.
- `monitor_transfer.csv` — fr_only-trained monitor → greedy test AUROC 0.733
  (< 0.755 conventional-on-greedy) → the 0.84 does not transfer.
- `incremental_percentile.csv` — mixing FR into the percentile combo *hurts*
  (−0.06 to −0.18 AUROC) everywhere except conservative (+0.002, CI includes 0).
- `spearman_redundancy.csv` — image of correlations; FR columns correlate
  r ≈ 0.2–0.5 with entropy-family features; far from orthogonal.
- `prevalence_wrong.csv` — class base rates per method (0.2625–0.4675), the
  AUPRC random-ranking baseline.
- `decoding_paired_stats.csv` — verified deltas (§2), 5000-re-sample bootstrap.
- `failure_category_counts.csv` — C: +19/+15/+13/+43/+33 improvements,
  B: −18/−14/−15/−66/−130 regressions for full/conservative/nll/entropy/fr resp.
- `verified_summary.csv`, `audit_lines.txt`.
- Figures: `fig_auroc_forest.png`, `fig_reliability.png`, `fig_risk_coverage.png`.

## 5. Reproduce

```
cd C:\Users\nehad\Desktop\geodesics
python -m pip install pandas numpy scipy scikit-learn matplotlib
python analysis\audit_analysis.py        # main audit; ~2–3 min CPU
python analysis\probe_auroc.py           # FR-variant scoreboard
python analysis\probe_nlldiff.py         # tie-divergence quantification
python analysis\probe_zoom.py            # q0/step228 tie case + parts≡merged
```

## 6. Verdict on the Round-1 claims

| Claim | Status |
|---|---|
| 6-method decoding comparison, frozen split, identical eval | reproduced exactly |
| full_geodesic +1.25 pp over greedy | correct number, but CI includes 0 → unresolved |
| Fisher–Rao monitor AUROC 0.844 | **not reproducible as FR**; reproducible only as fr_only's combined monitor (0.842) — artifact |
| Entropy / NLL monitors ≈ 0.72–0.74 | supported (COMBINED 0.71–0.73; single best honest trained monitor 0.755–0.758) |
| FR adds information beyond entropy/NLL | **negated** on current data (Phase 4: Δ ≤ +0.013, CIs include 0) |
