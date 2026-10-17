# Zoom into question 0 around the divergence: is the nll_only run choosing
# a non-cand0 token? And are part files consistent with the merged traces?
import pandas as pd

KEEP = ["method", "question_id", "step", "chosen_token", "greedy_token",
        "nll", "current_entropy", "gate", "topk_ids"]

tr = pd.read_csv("geodesic_study/token_traces.csv", usecols=KEEP)
p0 = pd.read_csv("geodesic_study/token_traces_part0.csv", usecols=KEEP)
p1 = pd.read_csv("geodesic_study/token_traces_part1.csv", usecols=KEEP)

for name, df in [("merged", tr), ("part0", p0), ("part1", p1)]:
    rows = df[(df.question_id == 0) & (df.method == "nll_only") &
              (df.step >= 226) & (df.step <= 230)]
    print(f"--- {name}: q0 nll_only steps 226-230 ---")
    print(rows.to_string(index=False))
    rows_g = df[(df.question_id == 0) & (df.method == "greedy") &
                (df.step >= 226) & (df.step <= 230)]
    print(f"--- {name}: q0 greedy  steps 226-230 ---")
    print(rows_g.to_string(index=False))
    print()

# Which (question, method, step) rows differ between parts and merged?
full = pd.concat([p0, p1], ignore_index=True)
full = full.drop_duplicates(subset=["question_id", "method", "step"], keep="last")
key = ["question_id", "method", "step"]
mm = tr.merge(full, on=key, suffixes=("_merged", "_part"),
              how="outer", indicator=True)
print("rows in merged but not in parts:", (mm._merge == "left_only").sum())
print("rows in parts but not in merged:", (mm._merge == "right_only").sum())
tk = mm[(mm._merge == "both") & (mm.chosen_token_merged != mm.chosen_token_part)]
print("rows where chosen token differs between merged and parts:", len(tk))
tk2 = mm[(mm._merge == "both") & (mm.greedy_token_merged != mm.greedy_token_part)]
print("rows where greedy differs between merged and parts:", len(tk2))
