"""A1-A4 raw-artifact audit of the run-2 full-coverage monitor outputs.
Read-only; writes a machine-readable verdict to analysis/outputs2/raw_audit.csv.
"""
import pandas as pd

D = "kaggle_out/full/geodesic_study2"
C = pd.concat([pd.read_csv(f"{D}/monitor_candidates_part{i}.csv") for i in (0, 1)],
              ignore_index=True)
S = pd.concat([pd.read_csv(f"{D}/monitor_steps_part{i}.csv") for i in (0, 1)],
              ignore_index=True)
A = pd.concat([pd.read_csv(f"{D}/monitor_answers_part{i}.csv") for i in (0, 1)],
              ignore_index=True)
out = []


def rec(name, result):
    out.append({"check": name, "result": result})
    print(f"{name}: {result}")


# ---- A1 answers ---------------------------------------------------------
sp = pd.read_csv("geodesic_study/split_map.csv")
rec("A1 split labels == frozen run-1 map (row-exact)",
    bool(A.sort_values("question_id").split.tolist()
         == sp.sort_values("question_id").split.tolist()))
rec("A1 n questions / unique qids", f"{len(A)} / {A.question_id.nunique()}")
rec("A1 prevalence wrong (test)",
    round((1 - A[A.split == "test"].correct).mean(), 4))
rec("A1 dev accuracy / test accuracy",
    (round(A[A.split == "dev"].correct.mean(), 4),
     round(A[A.split == "test"].correct.mean(), 4)))
rec("A1 stopped_on counts", A.stopped_on.value_counts().to_dict())
eos = (A.stopped_on == "eos")
exp = 1 + 5 * A.generated_tokens - eos.astype(int)
rec("A1 forward_calls == 1 + 5*steps - eos_stops (exact)",
    bool((A.forward_calls == exp).all()))

# ---- A2 candidates ------------------------------------------------------
g = C.groupby(["question_id", "step"]).size()
rec("A2 exactly 4 candidates per step", bool((g == 4).all()))
rec("A2 candidate rows total", len(C))
mg = C[C.is_greedy == 1].set_index(["question_id", "step"])
ch = S.set_index(["question_id", "step"]).chosen_token
rec("A2 greedy candidate token == chosen token (every step)",
    bool(mg.token_id.reindex(ch.index).eq(ch).all()))
Cs = C.sort_values(["question_id", "step", "rank"])
mono = Cs.groupby(["question_id", "step"]).nll.apply(
    lambda x: bool((pd.Series(x.values).diff().dropna() >= -1e-9).all()))
rec("A2 NLL monotone non-decreasing in rank", mono)
rec("A2 NaN count in feature columns",
    int(C[["q_entropy", "fr_greedy", "fr_consensus", "nll", "p_value"]]
        .isna().sum().sum()))
rec("A2 fr_greedy range", (round(C.fr_greedy.min(), 4), round(C.fr_greedy.max(), 4)))
rec("A2 rank0 |fr_greedy| max (fp32 floor)", round(mg.fr_greedy.abs().max(), 6))

# ---- A3 steps -----------------------------------------------------------
js = S.groupby("question_id").size()
gt = A.set_index("question_id").generated_tokens.reindex(js.index)
rec("A3 steps == generated_tokens (every question)", bool((js == gt).all()))
rec("A3 total steps", int(js.sum()))
rec("A3 tie events / inconsistent tie rows",
    (int(S.tie_flag.sum()), int(((S.tie_flag == 1) & (S.p_gap01 > 0)).sum())))
rec("A3 h_p in [0, log V] (V=151936)", bool(S.h_p.between(0, 151936 * 0 + 12).all()))
rec("A3 p_gap01 >= 0", bool((S.p_gap01 >= -1e-9).all()))

# ---- A4 config ----------------------------------------------------------
cfg = pd.read_csv(f"{D}/monitor_config.csv").to_dict("records")[0]
rec("A4 config row", {k: cfg[k] for k in
                      ("model", "seed", "top_k", "max_new_tokens",
                       "n_gpus", "torch", "wall_seconds")})

pd.DataFrame(out).to_csv("analysis/outputs2/raw_audit.csv", index=False)
print("\nwrote analysis/outputs2/raw_audit.csv")
