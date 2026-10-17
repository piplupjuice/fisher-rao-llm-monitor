# Audit Checklist — Run 2 → Paper v1.1

Auditor: agent session 2026-10-17. Every check below cites its evidence file.
Status: **PASS** / **PASS-with-correction** / **OPEN (unverifiable here)**.

## 1. Raw artifacts (read-only, untouched)

| # | Check | Evidence | Status |
|---|---|---|---|
| 1.1 | 500 unique qids, split labels row-exact vs frozen `geodesic_study/split_map.csv` | `analysis/outputs2/raw_audit.csv` (A1) | PASS |
| 1.2 | Splits: 100 dev / 400 test; dev-fit only, test untouched | A1 + code path inspection (`fit_score` standardizes/median-fills with dev stats only) | PASS |
| 1.3 | Test error prevalence 0.275 (110 wrong / 400); dev acc 0.670; test acc 0.725 | A1 | PASS |
| 1.4 | 519,356 candidate rows = 4 per step everywhere | raw_audit (A2) | PASS |
| 1.5 | rank-0 candidate token == decoded chosen token at all 129,839 steps | raw_audit (A2) | PASS |
| 1.6 | NLL monotone non-decreasing in candidate rank (every step group) | raw_audit (A2) | PASS |
| 1.7 | Zero NaN in candidate features; FR ∈ [0, π]; rank-0 FR floor 0.0021 (fp32) | raw_audit (A2) | PASS |
| 1.8 | forward_calls == 1 + 5·steps − 1[eos] exactly per question (495 eos / 5 limit) | raw_audit (A1) | PASS |
| 1.9 | Ties: 969/129,839 steps (0.75%), 0 tie/p-gap inconsistencies | raw_audit (A3) | PASS |
| 1.10 | Config row matches claims (2×T4, bf16, torch 2.11.0+cu128, wall 13,134.9 s) | `monitor_config.csv` (A4) | PASS |
| 1.11 | Determinism: byte-identical candidate tables across 2 independent Kaggle GPU sessions + 3 local CPU runs | smoke md5 `d7db02eb…` | PASS |
| 1.12 | No per-question failures; no silent past-KV corruption (in-run crop assertion per step) | absent `failures*.csv`; kernel log | PASS |
| 1.13 | Run-1 raw data unchanged (checksums, zip intact) | `geodesic_study/` untouched; `hellio_backup_run1.ipynb` | PASS |

## 2. Analysis artifacts (corrected pipeline, `run2_analysis.py`)

| # | Check | Evidence | Status |
|---|---|---|---|
| 2.1 | Monitor table reproduced byte-identically on an independent re-run | md5 equality of 5 CSVs vs `analysis/outputs2_reprod/` | PASS |
| 2.2 | Every AUROC and AUPRC verified against `sklearn` (hand-impl diff = 0.0) | `trained_monitors.csv` columns `*_maxdiff` | PASS |
| 2.3 | Every CI contains its own point estimate (post bug-fix) | `trained_monitors.csv`, `paired_deltas.csv` | PASS |
| 2.4 | Paired bootstrap uses a single shared resample index for y/sᴬ/sᴮ | `boot_paired` source inspection | PASS |
| 2.5 | Two-sided bootstrap p-values computed by reflection; reported for every delta | `paired_deltas.csv` | PASS |
| 2.6 | Pre-registered CONV (8 features) and G1–G6 implemented per written definitions | §4.3 definition table vs code | PASS (after corrections) |
| 2.7 | Labels (`correct`/`wrong`) never used as features; features only from decoded trajectory + logits | `question_features.csv` column audit | PASS |
| 2.8 | Risk–coverage at exactly 80%/90% exported with retained-set accuracy | `risk_coverage_crit.csv` | PASS |
| 2.9 | Figures regenerate from the same pipeline (no hand-tuned visuals) | `analysis/figures/fig_run2_*.png` | PASS |

## 3. Hypothesis-label discipline

| # | Check | Status |
|---|---|---|
| 3.1 | H1/H2 pass criteria frozen pre-data in `docs/experiment_protocol.md` addendum | PASS (documented) |
| 3.2 | Confirmatory set = 6 prereg single-feature additions + ALL_GEO + GEO-only | PASS |
| 3.3 | `max_frcons` labeled **exploratory** (implementation extra beyond prereg) | PASS |
| 3.4 | `tie_frac` never in CONV (prereg) — reported univariate-only | PASS |
| 3.5 | No multiplicity-correction survival claimed for the two nominally significant *harmful* deltas (p = 0.042/0.043 > Bonferroni 0.00625) | PASS |
| 3.6 | Protocol written after run-1 (not annotation-free) — disclosed in paper §1.2/§8.5 | PASS (disclosed) |

## 4. Four analysis defects — disclosure audit

| Defect | Disclosed in paper? | Impact quantified? | Preserved originals? |
|---|---|---|---|
| D1 late-quarter-only aggregation of mean_H/max_H/mean_nll/gate_frac/all FR means (material) | §4.6(1) | Yes: CONV 0.704→0.770 on fix; all deltas re-tabulated | `run2_analysis_prefix.py`, `analysis/outputs2_prefix/`, `docs/run2_report.md` body |
| D2 `mean_hgap` ≡ duplicate of `mean_next_H` (G5 mis-implementation) | §4.6(2) | Yes (identical-model rows pre-fix) | same |
| D3 `std_fr` computed over rank-0 rows (≈0); prereg G6 requires candidates (j>0) | §4.6(2) | Yes (univariate AUROC 0.507→0.334 with correct def) | same |
| D4 non-paired CI helper (independent y/s resampling) — caught because CI excluded own point estimate | §4.6(3) | No conclusions were ever drawn from it; fixed before inference | same |
| D5 `max_frcons` extra beyond prereg | §4.6(3) + Table 2 status column | Yes (now exploratory) | same |

All five (D4 counted) → **disclosure audit PASS**.

## 5. Paper-claims vs artifacts cross-check

| Paper number | Artifact | Status |
|---|---|---|
| CONV AUROC 0.770 [0.713, 0.821], AUPRC 0.596, Brier 0.159 | trained_monitors.csv | PASS |
| FR-only 0.661 [0.599, 0.719], AUPRC 0.397 | trained_monitors.csv | PASS |
| Table 2 deltas (8 rows) + p-values | paired_deltas.csv | PASS |
| Table 3 risk–coverage 81.3/77.2 (CONV), 78.8/75.8 (ALL_GEO), 75.9/74.4 (GEO) | risk_coverage_crit.csv | PASS |
| Table 4 univariate + dev Spearman | univariate_auroc.csv, spearman_dev.csv | PASS |
| Redundancy magnitudes (ρ = 0.51 mean_altdist↔mean_H; −0.63 std_fr↔mean_next_H) | spearman_dev.csv | PASS (v1.1 fix: draft said vaguely "high"; corrected to exact values) |
| §2 confound table (52.0% acc, prevalence 0.4675, mean_H 0.837, COMBINED 0.8417 …) | `analysis/outputs/*` run-1 audit CSVs | PASS |
| Run-1 gate rates 9.0% / 0.65% | `geodesic_study/summary.csv` | PASS |
| Reproduction agreement (run-2 greedy ≡ run-1 greedy arm 500/500, acc 0.714) | A1 + `predictions.csv` | PASS |

## 6. OPEN items

**Resolved 2026-10-17 (paper v1.2):**

1. **Literature / novelty — RESOLVED (partial by design).** A targeted arXiv
   search was performed on 2026-10-17 via the arXiv export API, restricted to
   cs.CL (with cs.LG overlap), using these query families:
   - `abs:"Fisher-Rao" AND cat:cs.CL` (10 hits reviewed in full)
   - `abs:"first-token" AND abs:"uncertainty" AND cat:cs.CL` (7 hits reviewed in full)
   Result: no paper found applying candidate-future divergence features to a
   supervised correctness monitor under frozen greedy decoding with a
   pre-registered falsification protocol. Closest works recorded in the paper
   (§References "Concurrent/related preprints"), most notably
   [Ying et al., arXiv:2610.05114] (Fisher–Rao geometry over layer-wise
   predictive revisions + detection AUROC). *Remaining limit, stated in the
   paper:* targeted, not exhaustive; non-arXiv venues and different
   vocabulary not covered; no general first-ness claim is made.
2. **Citations/DOIs — RESOLVED.** All 16 original references plus the Qwen2.5
   Technical Report were machine-verified on 2026-10-17 (arXiv export API,
   Crossref, doi.org). Two metadata errors found and fixed: Nielsen 2020 =
   Entropy 22(10):1100 (not 1105); Kapoor & Narayanan 2023 = Patterns 4(9):
   100804 (not 4(7)). Hendrycks & Gimpel arXiv ID confirmed as 1610.02136;
   Gorman & Bedrick as ACL 2019 P19-1267; Kamath et al. corrected to
   Kamath, Jia & Liang (arXiv:2006.09462). The two 1940s Calcutta Math. Soc.
   classics have no digital identifiers; they are cited via their standard
   form with a note. No reference required removal.

**Still open:**

3. **Figures byte-reproducibility** intentionally not asserted (matplotlib
   rendering nondeterminism); numbers behind them are assert-reproducible.
4. **Brier calibration choice** (sigmoid of decision function) is a defensible
   convention, not a truth claim — flagged, not an error.
5. **Power bound** (excludes FR gains ≳ +0.04 AUROC; cannot exclude ≲0.01) is
   stated from CI widths, not from a formal power simulation — flagged as
   future work.

## 7. Version ledger

- `run2_analysis.py` — v1.0 pre-fix (preserved as `run2_analysis_prefix.py`), v1.1 corrected (authoritative).
- `analysis/outputs2/` — corrected; `analysis/outputs2_prefix/` — audited pre-fix; `analysis/outputs2_reprod/` — independent rerun.
- `docs/run2_report.md` — original body preserved verbatim; addendum appended (append-only).
- `paper/paper_draft.md` — v1.0 → v1.1 (novelty hedging, redundancy values, reproduction statement, reference header).
- `paper/paper_draft.md` — v1.2 (2026-10-17): references machine-verified (2 metadata fixes), Related-Work search performed and documented, concurrent 2026 FR-geometry preprints added as adjacent work.
- `paper/audit_checklist.md` — OPEN items 1–2 resolved with documented evidence (see §6).
- Raw data: never modified (A1–A4 read-only scripts).
- No git commit was made by the agent (repo state left to the author).
