# Next-Experiment Recommendation — Decision After the 3B Replication

**Status: recommendation only. Nothing has been launched or approved.**
This file exists to prevent drifting into an uncontrolled feature search: it
states what each candidate follow-up would test, what we can already predict,
what it costs, and the multiple-testing rules it must satisfy.

## Candidates evaluated

### Candidate A — Jensen–Shannon divergence between candidate futures (piggyback)

- **Distinct hypothesis tested:** whether *any* symmetric, bounded divergence
  between one-step candidate futures adds correctness signal beyond CONV —
  i.e., whether run-2's null is specific to Fisher–Rao geometry
  (arccos/Bhattacharyya affinity) or generic to "dissimilarity of futures."
  JSD is a genuinely different functional (symmetric, bounded in [0, log 2],
  no square-root-affinity), though still one-step and still top-4.
- **Does the current null predict it?** Partially. Run-2's FR-only AUROC
  (0.661) and the moderate redundancy structure (ρ ≈ 0.4–0.6 with entropy
  block) suggest *any* same-regime one-step future geometry is plausibly
  redundant — but JSD is not mathematically monotone in FR, so it is a
  legitimate distinct test, not derivable from run-2's data.
- **Cost if piggybacked:** **zero extra GPU inference** *if* JSD is computed
  inside the 3B replication runner at each step from the same softmax futures
  (one extra vectorized reduction per step — negligible). If computed in a
  separate post-hoc pass, it requires re-running inference, because raw
  future *distributions* were not saved in run 2 (only scalars) — a real
  cost (~7–9.5 GPU-h) that piggybacking avoids. **This is the strongest
  argument for A: the runner must be extended once, before replication
  launch, to also record JSD and (optionally) the raw future logits' top-q
  vector; that extension is a protocol change and must be added to
  `replication_protocol.md` BEFORE approval and timestamping.**
- **New insight or feature expansion?** Insight *if* JSD also nulls
  (generic-white-noise conclusion for the whole one-step geometry class) or
  *if one of the two divergences wins* (geometry-specific mechanism, a
  publishable dissection). Mere feature expansion *if* both measures correlate
  ρ > 0.9 with FR and behave identically — reporting plan handles that by
  declaring the JSD arm exploratory-secondary and reporting its Spearman
  correlation with FR first.
- **Multiple-testing containment:** JSD arm labeled **exploratory-secondary**
  in the replication prereg; primary family remains the 8 prereg FR tests;
  JSD excluded from the confirmatory count; any positive JSD finding must be
  re-tested on the SAME frozen test set with Holm correction across the
  explicitly enumerated extra divergences (only JSD is enumerated — no
  post-hoc additions permitted).

### Candidate B — Multi-step lookahead (k = 2–3 future divergence geometry)

- **Distinct hypothesis tested:** does the geometry of *trajectories of
  futures* (corridor divergence: how the candidate-future manifolds separate
  over k tokens) carry information that one-step futures provably do not?
  This is a genuinely different measurement: FR at one step is a single-token
  boundary quantity; run-2's null says nothing strict about k-step separations
  (the paper says so explicitly in §8.2 — "deep futures untested").
- **Does the current null predict it?** No — this is the one candidate whose
  outcome run-2 cannot bound. (We note one suggestive but non-conclusive
  observation: run-2's opposite-polarity FR features show geometry carries
  *some* correct-signaling information, so deeper futures doing more of the
  same is plausible either way.)
- **Cost:** ×3× the per-step lookahead work (k = 3 → 5-forward-lookahead →
  ~9 forwards/token: 1 main + (1+2+3) lookahead forwards) ≈ **run-2's cost
  again (~7–9 GPU-h at 1.5B; ~15+ at 3B)**. Also requires its own prereg and
  smoke verification (crop semantics at depth 3, resume logic).
- **New insight or feature expansion?** The most *scientifically* interesting
  of the two (it tests a structural limitation of run 2 rather than swapping
  functionals). But it is exactly the kind of experiment that becomes a
  positive-mining exercise if run ad hoc. Therefore:
- **Recommendation:** **Do NOT launch now.** Decision rule: run only if
  (i) the 3B replication lands (either sign), AND (ii) the paper's confirmatory
  record is otherwise closed, AND (iii) B is preregistered separately — with
  its own MDE (suggest ΔAUROC ≥ +0.05 again), single primary metric (the
  k-step FR-only monitor's Δ vs CONV), fixed n = 500, fixed k = 3, and a
  written commitment that neither k, nor divergence, nor step-aggregation
  will be revisited regardless of outcome. Prepared as a draft (not yet
  written as prereg — only after the replication result exists, because the
  replication's outcome legitimately informs B's framing, and pretending
  otherwise would be fake preregistration).

## Explicit anti-selection rules (both candidates)

1. One primary metric per experiment; everything else is reported secondary,
   regardless of sign.
2. No experimental phase may be run twice with tweaked parameters; failures
   are reported as failures.
3. Every prereg file gets a timestamped hash (git commit) BEFORE launch; any
   post-outcome edit requires a new version plus a deviation note.
4. Test set (400 qs, frozen) may not be accessed for design decisions after
   the first outcome is observed; all subsequent-derived analyses are labeled
   "post-hoc, unadjusted."

## Bottom line

- **Approve and piggyback:** A (JSD) inside the pending 3B replication —
  near-zero cost, legitimate distinct hypothesis, contained.
- **Defer:** B (k-step) until the replication closes, then preregister
  separately if the research question still needs it.
- 7B: re-scope to whenever > 24 GB compute is available; do not quantize to
  force it (would break the frozen-numerics comparability that gives this
  study its evidential strength).
