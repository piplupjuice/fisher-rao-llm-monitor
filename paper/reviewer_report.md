# Reviewer-style report on `paper/paper_draft.md` (v1.1)

Perspective: a competent-but-skeptical ICML reviewer (R2 profile: selective
prediction / uncertainty / evaluation methodology). No claims about accept or
reject decisions.

## Three strongest contributions

**C1 — A matched-trajectory falsification with unusually complete artifact
hygiene.** The experimental question (does FR geometry between candidate
futures add information *when everything is measured on the same trajectory
and no gating exists*?) is precisely posed, pre-registered in writing with
numeric pass criteria, executed with zero failures, and every claim is
traceable to a released CSV (sklearn-cross-checked, byte-reproducible on
rerun). Most LLM-monitoring papers do not freeze feature definitions or
release the per-candidate table; this one does both. The negative answer is
therefore *costly to dispute* — a reviewer can reproduce every row.

**C2 — Mechanistic decomposition of the ≈0.84 confound.** The paper does not
just report its own null; it explains an earlier positive-looking number from
its own lab as a two-regime artifact (degraded controller → 52% accuracy /
46.75% prevalence; entropy/length features carry the regime's error signal;
FR features themselves 0.34–0.42 univariate; gate-censoring structurally
selects high-uncertainty steps). The matched clean-regime replication
(settling at 0.770 equal to run-1's independent conventional estimate) is the
causal seal. This is a genuinely reusable evaluation-hygiene case study for
the monitoring/uncertainty subfield.

**C3 — Useful released asset for the geometric-monitoring research program.**
519,356 candidate rows with per-step candidate futures' scalar geometry + a
deterministic decode harness (tie-stabilized greedy; byte-identical
cross-session) lets others test *their* geometry cheaply. Null results are
most valuable when they enable the next experiment at near-zero cost; this
asset does.

## Five most serious weaknesses

**W1 — Single small model (severity: high).** A 1.5B model under bf16 may not
have well-calibrated/segmented candidate-future geometry at all; the null may
reflect model immaturity, not a property of the feature family. A 7–8B
replication is the single change that would most alter the paper's weight.
*Flagged remedy:* pre-registered 3B replication (§`experiment/replication_protocol.md`).

**W2 — Statistical power and dev-set size (severity: high).** Dev n = 100
drives monitor variance; test n = 400 excludes FR gains ≳ +0.04 AUROC but
cannot exclude ≲0.02, and the two "significant" harmful deltas
(p = 0.042/0.043, 8 comparisons) would not survive multiplicity correction —
the paper says so, but a hostile reader can still frame the design as
under-powered both ways. *Remedy:* formal power analysis; larger replication
if resources allow; pre-registered hierarchies.

**W3 — Question-level aggregation may be the real bottleneck (severity:
medium-high).** Mean/max/range aggregates discard trajectory structure
(timing, escalation patterns). An attacker: "geometry's value, if any, is
in *when* divergence occurs, not its average." The paper classifies this as
untested (Table 5), which is honest but leaves the core question only
half-covered. *Remedy:* sequence-level monitor arm on a future dev set ≥ 4×
larger, pre-registered.

**W4 — Representativeness of the tested geometry (severity: medium).** One
divergence (FR over one-step futures over the top-4 grid). JS/Wasserstein/Rényi
variants and k-step futures are untested, so "geometric monitoring is
uninformative" cannot be generalized — the paper correctly scopes its claim
narrowly, which some reviewers will read as a limited-substance paper.
*Remedy:* the piggyback JSD arm in the replication (zero extra inference) plus
a k-step probe decision rule (see
`experiment/next_experiment_recommendation.md`).

**W5 — Protocol frozen post-audit; internal-authority provenance (severity:
medium).** Definitions were frozen after the team saw run-1 test behavior;
the confirmatory status is replication-with-frozen-definitions, weaker than
first-pass preregistration. Additionally all "define before measuring" claims
rest on files (not an external registry) — a skeptic may discount the timeline.
*Remedy:* state provenance honestly (done); for the replication, push the
preregistration to a timestamped external artifact or commit hash before any
3B outcome is observed.

**Additional minor flags** (not top-5): Brier defined on sigmoid of decision
function is nonstandard → report a temperature-scaled Brier in the appendix;
bf16 numerics and fp32 floor on rank-0 FR (0.0021) should be a footnote;
"folklore claim" is not attributed → either attribute or neutralize the
framing; abstract states "0.84 our prior experiment" — fine, but make sure
submission does not imply it is an external community claim.
