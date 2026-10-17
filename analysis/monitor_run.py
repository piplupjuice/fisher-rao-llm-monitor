# =========================================================================
# RUN 2 — FULL-COVERAGE GREEDY-ANCHORED MONITOR (geodesic Phase-2 asset)
# Single source of truth. Used as: (a) local CPU smoke test, (b) the new
# appended cell of hellio.ipynb, (c) the Kaggle kernel body for the full run.
#
# Protocol (pre-registered in experiment_protocol.md, run-2 addendum):
#   * Decoding = greedy with DETERMINISTIC tie-breaking:
#       highest p; if fp32 tie, smaller token id. (Fixes audited §3.5.)
#   * At EVERY step, for ALL k=4 candidates (no entropy gate — removes
#     audited gate-censoring §3.3): record NLL, q_j entropy,
#     FR(q_j -> q_greedy), FR(q_j -> consensus), p_value.
#   * Chosen token == candidate rank 0 always (the trajectory IS greedy).
#   * Frozen run-1 split/golds (never re-derived -> no new contamination).
# Env knobs: GEODESIC2_MODEL, GEODESIC2_LIMIT, GEODESIC2_MAX_NEW_TOKENS,
#            GEODESIC2_OUT_DIR, GEODESIC2_DATA, GEODESIC2_EMBED, GEODESIC2_SEQ
# ========================================================================
import json
import math
import os
import random
import re
import time
import traceback

import numpy as np
import pandas as pd
import torch

try:  # rarely needed; glob used for pristine check
    from glob import glob
except Exception:  # pragma: no cover
    glob = None

CONFIG = {
    "model": os.environ.get("GEODESIC2_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
    "seed": 42,
    "top_k": 4,
    "max_new_tokens": int(os.environ.get("GEODESIC2_MAX_NEW_TOKENS", "512")),
    "n_questions_limit": int(os.environ.get("GEODESIC2_LIMIT", "500")),
    "split_source": "run-1 frozen split (geodesic_study/split_map.csv / embedded)",
    "gold_source": "run-1 golds (geodesic_study/predictions.csv / embedded)",
    "torch": torch.__version__,
}
OUT_DIR = os.environ.get("GEODESIC2_OUT_DIR", "geodesic_study2")
os.makedirs(OUT_DIR, exist_ok=True)

if glob is None:
    def glob(pat):
        return [pat]

# ------------------------------------------------------------ helpers ----
def normalize_number(s):
    s = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s


def predicted_answer(text):
    t = text.replace("*", "").replace("`", "")
    m = re.findall(r"[Ff]inal\s+answer\s*[:：=]\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)", t)
    if m:
        return normalize_number(m[-1])
    m = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", t)
    if m:
        return normalize_number(m[-1])
    nums = re.findall(r"([-+]?\d[\d,]*(?:\.\d+)?)", t)
    return normalize_number(nums[-1]) if nums else ""


def entropy(p):
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)


def fisher_rao(p, q):
    aff = torch.sqrt(p.clamp_min(0) * q.clamp_min(0)).sum(-1)
    return 2.0 * torch.acos(aff.clamp(0.0, 1.0))


def topk_deterministic(p_row, k):
    """Highest-prob k tokens; ties (exact fp32 equality) -> smaller token id."""
    cand = torch.topk(p_row, k=k).indices.tolist()
    cand.sort(key=lambda i: (-float(p_row[i]), i))
    return cand


# ------------------------------------------------------------- engine ----
@torch.inference_mode()
def generate_monitor(prompt, tokenizer, model, eos_id):
    device = next(model.parameters()).device
    k = CONFIG["top_k"]
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    out = model(input_ids=ids, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :]
    forward_calls = 1
    seq_len = ids.shape[1]
    crop_ok = []
    steps, cands = [], []
    stopped = "token_limit"
    t0 = time.perf_counter()
    for step in range(CONFIG["max_new_tokens"]):
        p = torch.softmax(logits.float(), dim=-1)[0]
        cand = torch.as_tensor(topk_deterministic(p, k), device=device)
        p_top = p[cand]
        nll = (-torch.log(p.clamp_min(1e-12)))[cand]
        H_p = float(entropy(p).item())
        p_gap = float(p_top[0].item() - p_top[1].item())

        # ---- lookahead: all k candidates, no gate ----
        qs = []
        for j in range(k):
            o = model(input_ids=cand[j].view(1, 1),
                      past_key_values=past, use_cache=True)
            qs.append(torch.softmax(o.logits[:, -1, :].float(), dim=-1)[0])
            forward_calls += 1
            before = int(past.get_seq_length())
            past.crop(-1)
            if j == 0:
                crop_ok.append(int(past.get_seq_length()) == seq_len)
        crop_semantics = "OK" if all(crop_ok) else "FAILED"
        assert crop_ok and crop_semantics == "OK", (
            f"KV crop failed at step {step}: after={past.get_seq_length()} expect={seq_len}")
        Q = torch.stack(qs)                                   # [k, V]
        q_ent = entropy(Q)                                    # [k]
        fr_greedy = fisher_rao(Q, Q[0:1].expand(k, -1))       # [k], q0 ~ 0
        centroid = Q.mean(dim=0, keepdim=True)
        fr_cons = fisher_rao(Q, centroid.expand(k, -1))       # [k]
        chosen = int(cand[0].item())
        for j in range(k):
            cands.append({
                "step": step, "rank": j,
                "token_id": int(cand[j].item()),
                "p_value": float(p_top[j].item()),
                "nll": float(nll[j].item()),
                "q_entropy": float(q_ent[j].item()),
                "fr_greedy": float(fr_greedy[j].item()),
                "fr_consensus": float(fr_cons[j].item()),
                "is_greedy": int(j == 0),
            })
        steps.append({
            "step": step, "chosen_token": chosen,
            "h_p": H_p, "nll_greedy": float(nll[0].item()),
            "p_gap01": p_gap, "tie_flag": int(p_top[0].item() == p_top[1].item()),
        })
        if chosen == eos_id:
            stopped = "eos"
            break
        ids = torch.cat([ids, cand[0:1].view(-1, 1)], dim=1)
        try:
            out = model(input_ids=ids[:, -1:],
                        past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]
            forward_calls += 1
            seq_len += 1
        except Exception:
            out = model(input_ids=ids, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]
            forward_calls += 1
    new_tokens = ids[0, enc["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return (text, pd.DataFrame(steps), pd.DataFrame(cands), stopped,
            time.perf_counter() - t0, forward_calls)


# --------------------------------------------------------------- main ----
def load_items():
    """Frozen split-from-run-1 questions + golds. Locally: read the study CSVs;
    on Kaggle: an embedded JSON file (GEODESIC2_EMBED) so no run-1 artifacts."""
    embed = os.environ.get("GEODESIC2_EMBED")
    if embed:
        with open(embed, encoding="utf-8") as f:
            payload = json.load(f)
        src = pd.DataFrame(payload["rows"])
    else:
        src = pd.read_csv(os.environ.get(
            "GEODESIC2_DATA", "geodesic_study/predictions.csv"),
            usecols=["question_id", "method", "question", "gold"])
        src = src[src.method == "greedy"]
    src = src.drop_duplicates("question_id").sort_values("question_id")
    src["gold"] = src.gold.map(normalize_number)
    if "split" not in src.columns:
        split_map = pd.read_csv("geodesic_study/split_map.csv")
        src = src.merge(split_map, on="question_id", how="left")
    return src.reset_index(drop=True)


def run_worker(g, num_workers, src):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = f"cuda:{g}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    try:
        tok = AutoTokenizer.from_pretrained(CONFIG["model"])
        try:
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG["model"], dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG["model"], torch_dtype=dtype)
        model = model.to(device)
    except Exception as e:
        pd.DataFrame([{"worker": g, "error": f"{type(e).__name__}: {e}"}]
                     ).to_csv(f"{OUT_DIR}/failures_worker{g}.csv", index=False)
        raise
    model.eval()
    eos = tok.eos_token_id
    print(f"[worker{g}] device={device} dtype={dtype}", flush=True)

    part_ans = f"{OUT_DIR}/monitor_answers_part{g}.csv"
    part_stp = f"{OUT_DIR}/monitor_steps_part{g}.csv"
    part_cnd = f"{OUT_DIR}/monitor_candidates_part{g}.csv"

    done = set()
    if os.path.exists(part_ans):
        done |= set(pd.read_csv(part_ans, usecols=["question_id"])
                    .question_id.astype(int))
    todo = [i for i in range(len(src))
            if int(src.question_id[i]) not in done and i % num_workers == g]
    for n_i, i in enumerate(todo):
        qid, qtext, gold = (int(src.question_id[i]), src.question[i],
                            src.gold[i])
        splitlab = (src.split[i] if "split" in src.columns
                    else ("dev" if i < CONFIG.get("DEV_SIZE", 100) else "test"))
        msgs = [{"role": "system", "content":
                 "Solve the math problem carefully. Show your work step by step. "
                 "End your reply with exactly: Final answer: <number>."},
                {"role": "user", "content": qtext}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
        try:
            text, steps, cands, stopped, secs, fwd = generate_monitor(
                prompt, tok, model, eos)
            row = {"question_id": qid, "split": splitlab,
                   "prediction": predicted_answer(text), "gold": gold,
                   "correct": int(predicted_answer(text) == gold),
                   "stopped_on": stopped, "seconds": secs,
                   "forward_calls": fwd, "generated_tokens": len(steps),
                   "model": CONFIG["model"], "seed": 42, "response": text}
            for path, df in ((part_ans, pd.DataFrame([row])),
                             (part_stp, steps.assign(question_id=qid,
                                                     split=splitlab)),
                             (part_cnd, cands.assign(question_id=qid))):
                df.to_csv(path, mode="a",
                          header=not os.path.exists(path), index=False)
        except Exception as e:
            pd.DataFrame([{"question_id": qid,
                           "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc(limit=3)}]
                         ).to_csv(f"{OUT_DIR}/failures.csv", mode="a",
                                  header=not os.path.exists(
                                      f"{OUT_DIR}/failures.csv"), index=False)
            print(f"[fail] q{qid}: {type(e).__name__}: {e}", flush=True)
            continue
        if (n_i + 1) % 10 == 0:
            print(f"[worker{g}] {n_i + 1}/{len(todo)} done", flush=True)
        if device.startswith("cuda") and (n_i + 1) % 25 == 0:
            torch.cuda.empty_cache()


def out_dir_pristine():
    return not os.path.exists(f"{OUT_DIR}/monitor_answers_part1.csv")


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    src = load_items().head(CONFIG["n_questions_limit"]).reset_index(drop=True)
    print(f"[setup] model={CONFIG['model']} questions={len(src)} "
          f"max_new={CONFIG['max_new_tokens']}", flush=True)

    import multiprocessing as mp
    n_gpu = torch.cuda.device_count()
    t0 = time.time()
    if n_gpu >= 2 and out_dir_pristine() \
            and os.environ.get("GEODESIC2_SEQ", "0") != "1":
        ctx = mp.get_context("fork")
        procs = [ctx.Process(target=run_worker, args=(g, 2, src))
                 for g in range(2)]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        if not os.path.exists(f"{OUT_DIR}/monitor_answers_part1.csv"):
            print("[guard] worker1 produced nothing -> single-GPU fill", flush=True)
            run_worker(0, 1, src)
    else:
        run_worker(0, 1, src)
    pd.DataFrame([{**CONFIG, "n_gpus": n_gpu,
                   "transformers": __import__("transformers").__version__,
                   "wall_seconds": time.time() - t0}]
                 ).to_csv(f"{OUT_DIR}/monitor_config.csv", index=False)
    print(f"[done] wall={(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
