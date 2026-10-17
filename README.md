# Fisher-Rao Monitors for LLM Correctness — A Controlled Study

This project tests one clear question: **do Fisher-Rao geometry features help
predict when a language model gives a wrong answer, beyond simple features
like entropy and NLL?**

The short answer: **on our experiment, no.** We ran a controlled test. We
report the full result honestly, including what went wrong first.

This repo is for GSM8K math questions. We used the **Qwen2.5-1.5B-Instruct**
model. If you reuse our method, cite this repo and check the limits in Section
"Known limits".

---

## What this repo contains

| Folder / file | What it is |
|---|---|
| `notebooks/geodesic_study.ipynb` | The main notebook. Cell 1 is the first experiment. Cell 2 fixes the crowd-sourced top-k bug and adds clean decoding. Cell 3 is the run-2 monitor experiment. |
| `analysis/` | All analysis code. Each script runs on its own. |
| `kernels/` | Kaggle kernel notebooks we used on 2x T4 GPUs. |
| `geodesic_study/` | Raw results of the first experiment (run 1). Split files, gold answers, small CSVs. |
| `kaggle_out/` | Raw results of the second experiment (run 2). Large CSVs live here. Some are too big for GitHub. |
| `docs/` | All reports. Start here if you just want the story. |
| `docs/audit_phase1.md` | The audit of run 1. It found four real problems. |
| `docs/run2_report.md` | The full report of run 2 (the controlled test). |
| `docs/experiment_protocol.md` | The plan we froze before running. It states the rules and pass criteria. |
| `paper/` | Draft research paper, an audit checklist, and a review-style report. |
| `experiment/` | One plan to repeat the test on a bigger model. Not run yet. |

---

## The results in one minute

We decoded 500 GSM8K questions with plain greedy decoding (no sampling).
At every step we saved all 4 top tokens. For each token we saved:
negative log probability, entropy of its next-step distribution, and
Fisher-Rao distances between candidate futures.

Then we built small classifiers ("monitors") to predict *wrong vs. right*
final answers. We trained on 100 questions. We tested on 400 untouched
questions. Here is what we found.

| Monitor | AUROC on test data | Notes |
|---|---|---|
| Entropy + NLL + length only | **0.770** (CI 0.713 - 0.821) | Best monitor. Simple features win. |
| Same + all 6 Fisher-Rao features | 0.728 | Fisher-Rao did not help. |
| Fisher-Rao features only | 0.661 | Weaker than simple features. |

The confidence intervals come from a paired bootstrap with 2000 resamples.
Seed 42 everywhere. Code and data are in `analysis/` and `kaggle_out/`.

**Main finding.** Fisher-Rao geometry did not add useful information here
once entropy and length features were already used. Some Fisher-Rao features
even hurt the monitor.

**A second finding.** An earlier version of our experiment showed an AUROC
near 0.84. That number was misleading. The reason: bad sampling made the
model fail more often, and that regime made simple features look very
strong. The paper and `docs/run2_report.md` explain this step by step.

---

## Why this experiment is trustworthy

- Split is frozen. We never tune anything on the test set.
- We wrote the pass/fail rules before we looked at run-2 results.
- Decoding is fully deterministic. Ties go to the smaller token id.
  Re-running gives byte-identical outputs.
- Every number in the reports comes from a released CSV file.
- Independent metric checks (sklearn) matched our own code on every row.
- We found four bugs in our own first analysis. We fixed them. We published
  both the wrong and the corrected versions so you can check what changed.
- We report negative results. There is no cherry-picking here.

---

## How to run this yourself

You need Python 3.11+ and about 16 GB RAM for the analysis. For the GPU runs
you can use free Kaggle 2x T4 sessions, like we did.

```bash
pip install -r requirements.txt

# Raw-data integrity checks for run 2 (fast, CPU only)
python analysis/audit_raw_A1A4.py

# Rebuild every number, table, and figure of the run-2 analysis
python analysis/run2_analysis.py --indir kaggle_out/full/geodesic_study2 --outdir analysis/outputs2
```

Output tables go to `analysis/outputs2/`. Figures go to `analysis/figures/`.

To repeat the *generation* (the GPU part), open `kernels/` and push a kernel
to Kaggle, or run `notebooks/geodesic_study.ipynb` cell 3 on Kaggle. Each
full run costs about 15-19 T4-hours. A small test of 2 questions costs about
2 minutes, so start with that first.

---

## Data notes (what is NOT on GitHub)

Some raw files are too big for GitHub (50 MB+):

- `geodesic_study/token_traces*.csv` (run-1 per-token traces, about 160 MB)
- `geodesic_study2/monitor_candidates*.csv` (run-2 candidate table, about 54 MB)

Two ways to get them:

1. Download from Kaggle: `kaggle kernels output pikanigga/geodesic-run-2-full -p kaggle_out/full`
2. Re-run the kernels yourself.

Small key files *are* in this repo: `predictions.csv`, `split_map.csv`,
`summary.csv`, `monitor_answers.csv`, `monitor_steps.csv` (per part), and all
test scores in `analysis/outputs2/`.

---

## Known limits (please read)

- One model (1.5B), one dataset (GSM8K), bf16. Do not assume results extend to
  other models or tasks.
- Lookahead is one token. Longer lookahead was never tested.
- Test set has 400 questions. Our paired CIs can rule out Fisher-Rao gains
  bigger than about +0.04 AUROC, but nothing smaller.
- Monitors use per-question averages. Step-level sequence models may behave
  differently. This is untested.
- The protocol was written after we audited run 1. It is a documented replication,
  not a blind pre-registration. We say this openly in the paper.

---

## Suggested reading order

1. `docs/run2_report.md` — the whole run-2 story, short version.
2. `paper/paper_draft.md` — the submission draft with all numbers.
3. `paper/audit_checklist.md` — every check, evidence, and open item.
4. `docs/audit_phase1.md` — how we found and fixed our own problems.
5. `experiment/replication_protocol.md` — the plan for a 3B replication
   (pending approval, not run).

---

## Citation / reuse

Model: Qwen2.5-1.5B-Instruct (Apache-2.0, provided by Alibaba's Qwen team).
Dataset: GSM8K (Cobbe et al., 2021). Kaggle notebook GPUs (2x T4).

License: MIT (see `LICENSE`).

If this repo saved you from believing a *confounded* AUROC, that was the
point — negative and corrected results are still science.
