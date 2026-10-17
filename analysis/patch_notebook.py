"""Append the run-2 monitor cell to hellio.ipynb without touching anything else."""
import json

NB = r"C:\Users\nehad\Desktop\geodesics\hellio.ipynb"
SRC = r"C:\Users\nehad\Desktop\geodesics\analysis\monitor_run.py"

with open(SRC, encoding="utf-8") as f:
    src = f.read()

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)
n_before = len(nb["cells"])

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# =========================================================\n"
        "# RUN 2 CELL - FULL-COVERAGE GREEDY-ANCHORED MONITOR\n"
        "# Same code as analysis/monitor_run.py (single source of truth).\n"
        "# Deterministic greedy (tie -> smaller token id); all k=4 candidate\n"
        "# features at EVERY step, no entropy gate; frozen run-1 split.\n"
        "# Full run (~3-4 h, 2x T4):  just run this cell as-is.\n"
        "# Smoke test:  import os; os.environ['GEODESIC2_LIMIT']='8'\n"
        "#              os.environ['GEODESIC2_MAX_NEW_TOKENS']='64'\n"
        "#              then run this cell.\n"
        "# Outputs -> /kaggle/working/geodesic_study2/\n"
        "#   monitor_steps.csv / monitor_candidates.csv / monitor_answers.csv\n"
        "# =========================================================\n",
        src,
    ],
}

nb["cells"].append(new_cell)
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# verify round-trip
with open(NB, encoding="utf-8") as f:
    nb2 = json.load(f)
print("cells:", n_before, "->", len(nb2["cells"]))
print("last cell is monitor cell:", "RUN 2 CELL" in (nb2["cells"][-1]["source"][0]))
