# Do geometry features help catch a language model's wrong answers?

**One-sentence answer from this study: on my test, no.**

I tested a fancy idea from math ("Fisher–Rao geometry") as a way to tell when
the model Qwen2.5-1.5B-Instruct gets a GSM8K math question wrong. The simple
features (how surprised the model is, how long its answer is) worked.
The geometry features added **zero** extra help.

I also show why an earlier version of this study looked much better
(AUROC ≈ 0.84) and why that number was a trap.

Everything here is open. Every number comes from a file you can check.

---

## The full story — how this project got here

This is the whole project, told in order, the way it actually happened.
Five acts. Every number is real and comes from a file in this repo.

### Act I — Experiment 1: everything I did, everything I got

**The idea I wanted to test.** Bigger language models catch their own
mistakes better. Small models can't afford much self-checking, so I wanted a
cheap trick: instead of always taking the top token, pause at some steps,
look at the **top-4 candidate tokens**, peek **one token ahead** at what the
model *would* do under each candidate, and compare those candidate "futures"
using **Fisher–Rao geometry** — a mathematical distance between probability
distributions. Think of it as checking around a corner before committing to
a direction. The question: does this foresight produce *better answers*, and
do the geometric quantities also help a monitor *detect wrong answers*?

**What I did — the pipeline, step by step (fully deterministic):**

1. **Data.** 500 GSM8K math questions, frozen split: 100 dev / 400 test.
   Model: Qwen2.5-1.5B-Instruct, bf16.
2. **Decoding.** I decoded all 500 questions **six different ways** — one
   plain baseline and five lookahead-style methods (table below).
3. **Recording.** At every generation step I saved **all 4 top candidates**
   with **13 scores per candidate**: token NLL, token probability, next-step
   entropy, plus the Fisher–Rao distances (candidate future → greedy future,
   candidate future → the candidates' consensus) and combination scores.
4. **Traces.** Full token-level traces for every question — **744,951 rows**
   — so every measurement can be re-done forever without touching a GPU.
5. **Features.** Per question, I collapsed the trace into simple features:
   mean entropy, max entropy, mean NLL, the four FR features, answer length.
6. **Monitors.** I trained a small logistic-regression monitor on the 100 dev
   questions, tested on the 400 untouched test questions: *which features
   tell wrong answers from right ones?*

**The six decoding arms — what each one does and what it produced:**

| Arm | What it does | Test accuracy | Time/question | Forward passes/question |
|---|---|---|---|---|
| `greedy` | Baseline: always take the top token. | 71.4% | 9.1 s | 260 |
| `nll_only` | Lookahead always on; pick candidate by least-surprising future. | 71.6% | 9.1 s | 257 |
| `entropy_only` | Same gate, scored by future entropy (most certain future). | 66.8% | 10.7 s | 311 |
| `full_geodesic` | Pick by Fisher–Rao distance between futures, moderate gate (9% of steps). | **71.8%** | 12.1 s | 349 |
| `conservative` | FR gating only in the rarest, safest cases (0.65% of steps). | 71.6% | 9.3 s | 265 |
| `fr_only` | Aggressive FR reranking (fires on 22.6% of steps). | **52.0%** | 16.6 s | 488 |

*(All values from `geodesic_study/summary.csv` — reproducible.)*

**Everything the run produced (this is what "I did the experiment" means in
files):**

| Artifact | Size / rows | What is inside |
|---|---|---|
| `geodesic_study/predictions.csv` | 3,000 rows | Final answer per question × 6 arms + pairwise comparison fields |
| `geodesic_study/summary.csv` | 6 rows | Accuracy, timing, forward calls, gate rates per arm |
| `geodesic_study/token_traces*.csv` | 744,951 rows (~160 MB, off GitHub) | Every token step, every candidate, all 13 scores |
| `analysis/outputs/` | 11 tables | Run-1 audit tables, monitor scores, all AUROC/CIs |
| `geodesic_study/accuracy_overview.png` | figure | The 6-arm accuracy chart used in this README |

**What came out of the arms — the raw findings:**

1. **Foresight did not pay.** Best arm (`full_geodesic`) = 71.8% vs greedy
   71.4%: +0.4 points, not statistically significant (McNemar p = 0.49).
2. **Entropy-scored reranking actively hurt:** −5.7 points (p = 0.02).
3. **The aggressive FR arm broke the model:** 52.0% accuracy (−19.4 points,
   p < 0.0001), 1.9× the compute, and it stopped early on 36/500 questions
   (vs ~5 for the healthy arms) — the reranking was derailing generations.
4. **Cost curve:** more foresight = more forward passes (488 vs 260 per
   question), with zero accuracy to show for it.

**The monitors, and the moment everything changed.** Because each arm had
its own 500 generations, I could train a monitor on any arm. Accuracy-level
results were boring (0.71–0.72 everywhere except the broken arm). Then the
monitor table landed:

| Monitor trained on | COMBINED AUROC (test) |
|---|---|
| `fr_only` generations | **0.842** |
| `entropy_only` generations | 0.731 |
| `greedy` / `nll_only` / `full_geodesic` / `conservative` | 0.712 – 0.715 |

**0.842.** At that moment I believed I had found something: geometry
features looked like a huge correctness signal. This is the high point of
the story — and the exact moment a good researcher should get suspicious.

### Act II — The audit: I attack my own best result

I decided to assume my 0.842 was wrong until proven right. I wrote
independent checks (`analysis/audit_raw_A1A4.py`) that only re-measured —
no new modeling, just verification. What followed is the most important
part of this project:

- **The 0.842 was about the broken arm, not about geometry.** The `fr_only`
  arm had a 46.75% wrong-answer rate (vs 28% for greedy). In that wrecked
  regime the *simple* signals carry it all: mean entropy 0.837 AUROC,
  mean NLL 0.826 — the monitor was essentially reading "model is very
  confused ⇒ answer wrong", which is trivially true when the decoder is
  derailing itself.
- **The score did not transfer.** The monitor that scored 0.842 on the
  broken arm's data scored only **0.733** on healthy greedy generations.
  A real signal would travel. An artifact doesn't.
- **Geometry added nothing even there.** Paired deltas (geometry-augmented
  vs conventional monitor) on normal arms: +0.013 for `full_geodesic`
  (CI includes 0), −0.006, −0.002. Nothing significant in the good regime.
- **I also caught six bugs in my own analysis code** (feature duplications,
  aggregation errors, a tie-breaking nondeterminism that made two arms
  diverge, and later a plotting normalization bug). All fixed and versioned;
  the before/after artifacts are kept side by side.

The verdict on my best number: **0.842 = a regime artifact, not a finding.**
The honest arm-level story: geometry foresight doesn't beat greedy decoding,
and geometry features don't beat entropy/NLL in a monitor.

### Act III — Freeze the rules before the next shot

I wrote a pre-registration (`docs/experiment_protocol.md`): the same frozen
split, a rewritten deterministic runner, fixed monitor definitions, and
pass/fail criteria — including the exact condition under which the null
result stands. All written **before** any new GPU run.

### Act IV — The clean run (run 2)

Re-executed end-to-end on free Kaggle GPUs (2× T4): 519,356 candidate rows,
500/500 questions, zero failures, hardness-gated and seed-fixed. Decoding
was byte-identical across two independent GPU sessions (md5-verified). Wall
time: about 3.7 hours.

### Act V — The verdict

Under the pre-registered rules: **no Fisher–Rao feature adds information.**
Every pre-registered geometry arm scored ΔAUROC ≤ 0; one unregistered,
exploratory arm gained +0.005 (not significant). The honest monitor:
**0.770** (CI 0.713–0.821) with plain entropy/NLL/length features. FR alone:
0.661 — some signal, but redundant with entropy (dev Spearman ρ = 0.51)
once simple features are known. Both the failed version and the corrected
version of this study live in this repo, side by side.

---

## What I got out of this project

**Scientific findings:**

1. **Fisher–Rao candidate-future features add nothing** to entropy/NLL/length
   monitors on this model and task. Every pre-registered arm scored ≤ 0
   ΔAUROC; the only positive point estimate (+0.005) was an unregistered
   exploratory arm and is not significant.
2. **The honest monitor level is AUROC ≈ 0.77** on 400 frozen test questions.
3. **The famous-looking 0.84 was a regime confound** — I decomposed it fully
   (52% decoding arm, higher wrong-answer rate, simple features doing all the
   work) and can now tell that story with numbers.
4. **Fisher–Rao features are not useless**, but they are **mostly redundant**:
   alone 0.661 AUROC, yet overlapping with entropy (dev-set Spearman ρ = 0.51
   with mean next-entropy; −0.63 for the FR spread feature).

**Engineering outcomes:**

1. **A fully deterministic pipeline** — same inputs, same bytes out, verified
   across independent GPU sessions. Rare in LLM research and worth copying.
2. **A reusable audit framework** — raw-data checks (A1–A4), metric
   cross-checks vs sklearn, a written mapping of every claim to the CSV that
   backs it, and a line-by-line audit checklist.
3. **Six self-found defects** (D1–D6), each fixed, disclosed, and versioned —
   including one plotting bug found only while rebuilding this README.
4. **A pre-registered protocol** and a ready-to-run replication plan on a
   bigger model (Qwen2.5-3B), waiting for approval.
5. All artifacts: ~35 result tables (CSV), 6 figures, raw logs, kernel
   sources - everything regenerable from the commands in this README.

**What I still don't know (my open bets):** bigger model? different look-ahead?
step-sequence monitors? The written-but-unrun 3B plan (`experiment/`) is my
next test.

---

## New to this? Read this first (30 seconds)

- **A "monitor"** is a small helper model. It watches the big model and
  guesses: *right answer or wrong answer?*
- **AUROC** is a score from 0.0 to 1.0.
  - 0.5 = coin flip (useless).
  - 1.0 = perfect.
  - 0.77 = clearly useful but not perfect.
- **Entropy** = how unsure the model is about its next word. High = unsure.
- **NLL** (negative log-likelihood) = another word for "how surprised the
  model is". Low = confident.
- **Fisher–Rao (FR) distance** = a smart mathematical distance between two
  probability distributions. My question: does this smart distance see
  things that plain entropy cannot see?
- **Answer: no.** On this model and task: no.

---

## The results in one minute

I took 500 GSM8K questions. 100 for training the monitor, 400 kept away for
testing (untouched, frozen, never peeked at while building).

| Monitor | AUROC on 400 test questions | What it means |
|---|---|---|
| Simple: entropy + NLL + answer length | **0.770** (CI 0.713 – 0.821) | Best. Simple features win. |
| Simple + all 6 geometry features | 0.728 | Geometry made it **worse**. |
| Geometry features only | 0.661 | Some signal, but weaker than simple. |

More results, with all 8 statistical tests: one exploratory (not
pre-registered) arm, CONV+max_frcons, scored 0.775 — that is only +0.005 and
it is not statistically significant. Nothing else got even +0.005.

### The main chart

*Every monitor tested. The dashed line at 0.5 = coin flip. Higher = better.
The best simple setup sits near 0.77; the geometry-only setups sit lower.*

![Monitor AUROC chart with 95% confidence intervals](analysis/figures/fig_auroc_forest.png)

### All 500 questions in one 3D picture

This is the whole study in one image, drawn from the real per-question data (no
statistics, just the raw points). Every dot is one GSM8K question. Blue =
the model answered it correctly, red = wrong. If geometry "saw" something
entropy cannot see, red and blue would separate in some rotation of this
cloud. They don't. That is the null result, drawn.

![3D feature space of all 500 questions](analysis/figures/fig_3d_feature_space.png)

### What "skipping hard questions" looks like

If the monitor is good, you can skip the questions it marks as risky and the
questions you do answer become more accurate. Here is that trade-off.
At the 80% line (answer 80% of questions), the simple monitor and the
geometry monitor are close to each other or the simple one is better.

![Risk coverage curves](analysis/figures/fig_run2_risk_coverage.png)

### The trap: what the old ≈0.84 number really was

In my first run, one decoding arm accidentally made the model much weaker
(52% accuracy instead of 72%). In that broken regime, simple features looked
amazing (AUROC ≈ 0.84). But that was a feature of the broken regime, not a
feature of the geometry method. I show the pieces of this mistake
step-by-step in the docs. This chart shows how much worse the weak arm was:

![Accuracy of 6 decoding methods, run 1](geodesic_study/accuracy_overview.png)

*The red bar (`fr_only`, 52%) is the accidentally weak decoding arm.
Everything else is ~72%.*

### Honesty chart: my monitor scores are ranks, not probabilities

The monitors output a risk score. I scale it to 0–1 for plots. It is a
*ranking*, not a calibrated probability. This is stated everywhere.

![Reliability curves](analysis/figures/fig_reliability.png)

---

## Why you can trust these numbers

- Frozen test set. I never tuned anything on it.
- I wrote the pass/fail rules **before** looking at the run-2 results.
- Decoding is fully deterministic. Re-running gives byte-identical outputs
  (I checked, via md5, across two separate GPU sessions).
- I found **six bugs in my own analysis**. I fixed them, and I published
  the wrong version *and* the fixed version, so you can see the difference.
- Independent checks with sklearn matched my own code exactly (0.0
  difference) on every monitor.
- Two previous headline claims from this project were themselves wrong — the
  README you are reading now is the corrected story.

---

## Run it yourself

You need Python 3.11+ and about 16 GB of RAM.

```bash
pip install -r requirements.txt
```

**Important:** two raw data folders are too big for GitHub, so first
(optional, only if you want the full raw data):
```bash
kaggle kernels output pikanigga/geodesic-run-2-full -p kaggle_out/full
```

Then rebuild every table and figure of the main analysis:
```bash
python analysis/run2_analysis.py --indir kaggle_out/full/geodesic_study2 --outdir analysis/outputs2
python analysis/make_3d_figure.py
```

You should get the exact numbers above (seed 42, deterministic). The small
score tables (`analysis/outputs2/*.csv`) **are** in this repo, so you can
verify against them without downloading anything.

To redo the *GPU generation* part, use the Kaggle notebooks in `kernels/`
(cost: about 15-19 T4-hours; a 2-question smoke costs ~2 minutes).

---

## What is in this repo

| Folder / file | What it is |
|---|---|
| `notebooks/geodesic_study.ipynb` | The main notebook (3 cells: run-1 study, run-1 fix, run-2 monitor). |
| `analysis/` | All analysis code + small result CSVs + figures. |
| `kernels/` | Kaggle notebook sources for the 2x-T4 GPU runs. |
| `geodesic_study/` | Run-1 raw results (big CSVs excluded, see above). |
| `kaggle_out/` | Run-2 raw results (excluded from git; download command above). |
| `docs/` | The reports: audit, run-2 report, frozen protocol. |
| `experiment/` | A written plan (not yet approved) to repeat on Qwen2.5-3B. |

## Suggested reading order

1. This README.
2. `docs/run2_report.md` — the full run-2 story.
3. `docs/audit_phase1.md` — how the first experiment was audited, line by line,
   with the version ledger of all six defects.
4. `docs/experiment_protocol.md` — the frozen rules.
5. `experiment/replication_protocol.md` — the not-yet-run 3B plan.

## Known limits (please read)

- One model (1.5B), one dataset (GSM8K), bf16. Results do not automatically
  extend to other models or tasks.
- Lookahead is one token. Longer lookahead: untested.
- With 400 test questions, my confidence intervals can rule out geometry
  gains bigger than about +0.04 AUROC. Gains smaller than that: possible,
  I cannot see them.
- The monitors use one score per question. Step-level sequence models might
  behave differently. Untested.
- The replication plan is written but **not run**.

## Credits & license

- Model: Qwen2.5-1.5B-Instruct (Apache-2.0, Alibaba Qwen team).
- Dataset: GSM8K (Cobbe et al., 2021).
- Compute: free Kaggle notebook GPUs (2x T4).
- License: MIT (see `LICENSE`). Cite via `CITATION.cff`.

If this repo saved you from believing a *confounded* AUROC — that was the
point. Negative results, done carefully, are still science.
