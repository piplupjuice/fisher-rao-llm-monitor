# Audit: why does nll_only diverge from greedy token-by-token?
import pandas as pd

tr = pd.read_csv("geodesic_study/token_traces.csv",
                 usecols=["method", "question_id", "step", "chosen_token",
                          "greedy_token", "nll", "current_entropy", "topk_ids"])
pred = pd.read_csv("geodesic_study/predictions.csv")

g = tr[tr.method == "greedy"].set_index(["question_id", "step"])
n = tr[tr.method == "nll_only"].set_index(["question_id", "step"])
j = g.join(n, rsuffix="_nll", how="inner")
diff = j[j.chosen_token != j.chosen_token_nll]

fd = diff.reset_index().groupby("question_id").step.min()
cnt = diff.groupby("question_id").size()
print("questions with any token divergence:", len(cnt), "of 500")
print("first-divergence step: mean=%.1f median=%.0f max=%.0f" % (fd.mean(), fd.median(), fd.max()))
print("diff steps per question: mean=%.1f median=%.0f max=%d" % (cnt.mean(), cnt.median(), cnt.max()))

gg = pred[pred.method == "greedy"].set_index("question_id")
nn = pred[pred.method == "nll_only"].set_index("question_id")
diff_q = cnt.index
both_diff = gg.loc[diff_q, "prediction"] != nn.loc[diff_q, "prediction"]
print("final answer differs among diverged questions:",
      int(both_diff.sum()), "of", len(diff_q))
print("accuracy: greedy=%.4f nll_only=%.4f" % (gg.correct.mean(), nn.correct.mean()))

q = diff.index.get_level_values(0).unique()[0]
sub = diff.loc[q].head(3)
print()
print("=== example divergences, question", q, "===")
print(sub[["current_entropy", "nll", "nll_nll",
           "chosen_token", "chosen_token_nll", "greedy_token",
           "topk_ids"]].to_string())
