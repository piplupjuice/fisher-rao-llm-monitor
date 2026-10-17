"""Fetch a finished kernel's outputs and run the full verification gate on them.

Usage: python analysis/fetch_and_verify.py smoke
       python analysis/fetch_and_verify.py full
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
SLUG = f"pikanigga/geodesic-run-2-{MODE}"
DEST = os.path.join(ROOT, "kaggle_out", MODE)
os.makedirs(DEST, exist_ok=True)
os.environ["Path"] = os.environ["Path"] + \
    ";C:\\Users\\nehad\\AppData\\Roaming\\Python\\Python313\\Scripts"

print(f"== fetching outputs of {SLUG} -> {DEST} ==")
r = subprocess.run(["kaggle", "kernels", "output", SLUG, "-p", DEST],
                   capture_output=True, text=True)
print(r.stdout, r.stderr)

steps = os.path.join(DEST, "geodesic_study2", "monitor_steps_part0.csv")
if not os.path.exists(steps):
    # outputs may be dumped flat; look for part files anywhere in DEST
    for dirpath, _, files in os.walk(DEST):
        if "monitor_steps_part0.csv" in files:
            steps = dirpath
            break
    else:
        raise SystemExit("no monitor_steps_part0.csv found in output")

data_dir = steps if os.path.isdir(steps) else os.path.dirname(steps)
print("data dir:", data_dir)

import hashlib
import pandas as pd

c = pd.read_csv(os.path.join(data_dir, "monitor_candidates_part0.csv"))
s = pd.read_csv(os.path.join(data_dir, "monitor_steps_part0.csv"))
a = pd.read_csv(os.path.join(data_dir, "monitor_answers_part0.csv"))

ok = True


def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name)


gen = a.set_index("question_id").generated_tokens
check("steps == generated_tokens", (s.groupby("question_id").size() == gen).all())
check("4 candidates per step",
      (c.groupby(["question_id", "step"]).size() == 4).all())
mg = c[c.is_greedy == 1].set_index(["question_id", "step"])
ch = s.set_index(["question_id", "step"]).chosen_token
check("greedy candidate == chosen token",
      mg.token_id.reindex(ch.index).eq(ch).all())
check("fr_greedy rank0 ~ 0 (GPU fp32 floor)", mg.fr_greedy.abs().max() < 3e-3)
check("no NaN", c[["q_entropy", "fr_greedy", "fr_consensus"]].isna().sum().sum() == 0)
check("fr in [0, pi]", ((c.fr_greedy >= -1e-6) & (c.fr_greedy <= 3.1416)).all()
      and ((c.fr_consensus >= -1e-6) & (c.fr_consensus <= 3.1416)).all())
check("nll monotone with rank",
      (c.sort_values(["question_id", "step", "rank"])
       .groupby(["question_id", "step"]).nll
       .apply(lambda nl: bool((pd.Series(nl.values).diff().dropna() >= -1e-9).all()))).all())
check("answers parsed",
      a.prediction.astype(str).str.len().gt(0).all())
check("forward_calls == 1 + steps*5",
      (a.forward_calls == 1 + a.generated_tokens * 5).all())

ties = int(s.tie_flag.sum())
print(f"tie events: {ties} / {len(s)} steps")
print("cand md5:", hashlib.md5(c.to_csv(index=False).encode()).hexdigest())
print("answers md5 (except seconds):",
      hashlib.md5(a.drop(columns=["seconds", "response"]).to_csv(index=False).encode()).hexdigest())

print("\nGATE VERDICT:", "ALL PASS" if ok else "PROBLEMS FOUND")
print("correct rate on this sample:", a.correct.mean())
