"""Build the two Kaggle kernels (smoke + full) for run 2.

Embeds the frozen run-1 questions/splits/golds as base64 in a data cell so
the kernel needs no external dataset. Monitor code is pasted verbatim from
analysis/monitor_run.py (single source of truth).
"""
import base64
import json
import os

import pandas as pd

ROOT = "C:/Users/nehad/Desktop/geodesics"
SRC = f"{ROOT}/analysis/monitor_run.py"


def main():
    with open(SRC, encoding="utf-8") as f:
        runner = f.read()

    pred = pd.read_csv(f"{ROOT}/geodesic_study/predictions.csv",
                       usecols=["question_id", "method", "question", "gold"])
    pred = pred[pred.method == "greedy"]
    split = pd.read_csv(f"{ROOT}/geodesic_study/split_map.csv")
    items = (pred.merge(split, on="question_id").sort_values("question_id")
             [["question_id", "split", "question", "gold"]])
    payload = json.dumps({"rows": items.to_dict("records")},
                         ensure_ascii=False)
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    print(f"embedded payload: {len(payload)} bytes -> {len(payload_b64)} b64")

    header = """\
# =========================================================
# RUN 2 {MODE} KERNEL - geodesic greedy-anchored monitor
# Frozen run-1 questions/splits/golds embedded verbatim.
# =========================================================
import json, os
LIMIT = {limit}
MAX_NEW = {max_new}
SEQ = '{seq}'
"""

    data_cell = """\
import base64
_payload_b64 = "{b64}"
_ITEMS = json.loads(base64.b64decode(_payload_b64).decode("utf-8"))
with open("/kaggle/working/items.json", "w", encoding="utf-8") as f:
    json.dump(_ITEMS, f, ensure_ascii=False)
os.environ["GEODESIC2_EMBED"] = "/kaggle/working/items.json"
os.environ["GEODESIC2_OUT_DIR"] = "/kaggle/working/geodesic_study2"
os.environ["GEODESIC2_LIMIT"] = str(LIMIT)
os.environ["GEODESIC2_MAX_NEW_TOKENS"] = str(MAX_NEW)
os.environ["GEODESIC2_SEQ"] = SEQ
print("[data] embedded questions:", len(_ITEMS["rows"]))
""".replace("{b64}", payload_b64)

    gate_cell = """\
# =========================================================
# SMOKE GATE - alignment + determinism passport
# =========================================================
import hashlib
import pandas as _pd
_s1 = _pd.read_csv("/kaggle/working/geodesic_study2/monitor_steps_part0.csv")
_c1 = _pd.read_csv("/kaggle/working/geodesic_study2/monitor_candidates_part0.csv")
_a1 = _pd.read_csv("/kaggle/working/geodesic_study2/monitor_answers_part0.csv")
_gen = _a1.set_index("question_id").generated_tokens
assert (_s1.groupby("question_id").size() == _gen).all(), "step count mismatch"
assert (_c1.groupby(["question_id", "step"]).size() == 4).all(), "candidates != 4/step"
_mg = _c1[_c1.is_greedy == 1].set_index(["question_id", "step"])
_ch = _s1.set_index(["question_id", "step"]).chosen_token
assert _mg.token_id.reindex(_ch.index).eq(_ch).all(), "greedy != chosen"
assert _c1[["q_entropy", "fr_greedy", "fr_consensus"]].isna().sum().sum() == 0
assert _mg.fr_greedy.abs().max() < 3e-3, "rank0 fr_greedy nonzero"
assert (_a1.prediction.astype(str).str.len() > 0).all(), "unparsed answers"
assert _a1.correct.isin([0, 1]).all()
print("[GATE] alignment OK | tie events:", int(_s1.tie_flag.sum()),
      "of", len(_s1), "| forward_calls", _a1.forward_calls.tolist())
print("[GATE] candidates md5:",
      hashlib.md5(_c1.to_csv(index=False).encode()).hexdigest())
"""

    for mode, limit, max_new, seq in (("smoke", 2, 64, "1"),
                                      ("full", 500, 512, "0")):
        d = f"{ROOT}/kernels/geodesic-run2-{mode}"
        os.makedirs(d, exist_ok=True)
        nb = {
            "nbformat": 4, "nbformat_minor": 0,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python",
                                        "name": "python3"}},
            "cells": [
                {"cell_type": "code", "execution_count": None,
                 "metadata": {}, "outputs": [],
                 "source": header.format(MODE=mode.upper(), limit=limit,
                                         max_new=max_new, seq=seq)},
                {"cell_type": "code", "execution_count": None,
                 "metadata": {}, "outputs": [], "source": data_cell},
                {"cell_type": "code", "execution_count": None,
                 "metadata": {}, "outputs": [],
                 "source": runner},
            ],
        }
        if mode == "smoke":
            nb["cells"].append({"cell_type": "code", "execution_count": None,
                                "metadata": {}, "outputs": [],
                                "source": gate_cell})
        name = f"geodesic-run2-{mode}.ipynb"
        with open(f"{d}/{name}", "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        meta = {
            "id": f"pikanigga/geodesic-run-2-{mode}",
            "title": f"geodesic run 2 {mode}",
            "code_file": name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        with open(f"{d}/kernel-metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        size = os.path.getsize(f"{d}/{name}")
        print(f"built {d} ({size} bytes)")
        assert size < 900_000, "kernel source too big"


if __name__ == "__main__":
    main()
