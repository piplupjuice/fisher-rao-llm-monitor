# Smoke validation gate: alignment, completeness, determinism (RQ 7, 10).
import hashlib
import math
import numpy as np
import pandas as pd

D1 = "geodesic_study2_smoke"
D2 = "geodesic_study2_smoke2"  # second identical run written by determinism re-run

c1 = pd.read_csv(f"{D1}/monitor_candidates_part0.csv")
s1 = pd.read_csv(f"{D1}/monitor_steps_part0.csv")
a1 = pd.read_csv(f"{D1}/monitor_answers_part0.csv")

checks = []
def check(name, ok):
    checks.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL "), name)

# ---- alignment / completeness ------------------------------------------
gen = a1.set_index("question_id").generated_tokens
check("steps rows == generated_tokens per question",
      (s1.groupby("question_id").size() == gen).all())
check("4 candidate rows per step",
      (c1.groupby(["question_id", "step"]).size() == 4).all())
mg = c1[c1.is_greedy == 1].set_index(["question_id", "step"])
chosen = s1.set_index(["question_id", "step"]).chosen_token
check("greedy-voted candidate == decoded chosen token exactly",
      mg.token_id.reindex(chosen.index).eq(chosen).all())
check("rank ids 0..3 present", (c1.groupby(["question_id", "step"])["rank"]
      .apply(lambda r: set(r) == {0, 1, 2, 3})).all())
check("no NaN in candidate features",
      c1[["q_entropy", "fr_greedy", "fr_consensus", "nll", "p_value"]].isna().sum().sum() == 0)
check("fr_greedy of rank0 ~ 0 (fp32 tol < 1e-3)", mg.fr_greedy.abs().max() < 1e-3)
check("fr ranges sane", ((c1.fr_greedy >= 0) & (c1.fr_greedy <= math.pi)
      & (c1.fr_consensus >= 0) & (c1.fr_consensus <= math.pi)).all())
check("q_entropy >= 0", (c1.q_entropy >= 0).all())
check("nll monotone with p (rank ordering)", 
      (c1.sort_values(["question_id", "step", "rank"])
          .groupby(["question_id", "step"]).nll.apply(
              lambda nl: bool(np.all(np.diff(nl.values) >= -1e-9)))).all())
check("all answers parsed", a1.prediction.notna().all() and (a1.prediction.astype(str) != "").all())

# ---- determinism: second full run must be byte-identical ------------------
try:
    c2 = pd.read_csv(f"{D2}/monitor_candidates_part0.csv")
    s2 = pd.read_csv(f"{D2}/monitor_steps_part0.csv")
    same_c = hashlib.md5(c1.to_csv(index=False).encode()).hexdigest() == \
             hashlib.md5(c2.to_csv(index=False).encode()).hexdigest()
    same_s = hashlib.md5(s1.to_csv(index=False).encode()).hexdigest() == \
             hashlib.md5(s2.to_csv(index=False).encode()).hexdigest()
    check("tie determinism: two identical runs byte-equal (candidates)", same_c)
    check("tie determinism: two identical runs byte-equal (steps)", same_s)
except Exception as e:
    print("determinism re-run not available:", e)

ties = int(s1.tie_flag.sum())
print(f"tie events at k=1..2 top pair: {ties} of {len(s1)} steps "
      f"({ties / max(len(s1), 1):.1%})")
print("\nALL GATE CHECKS:", ["OK" if v else "PROBLEM" for _, v in checks])
import math  # noqa
