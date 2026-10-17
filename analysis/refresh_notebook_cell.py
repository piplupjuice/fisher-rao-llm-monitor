"""Re-apply the final monitor_run.py into hellio.ipynb as cell 3, and build
the Kaggle kernels (smoke + full) with embedded run-1 questions/golds."""
import json

SRC = r"C:\Users\nehad\Desktop\geodesics\analysis\monitor_run.py"
NB = r"C:\Users\nehad\Desktop\geodesics\hellio.ipynb"
with open(SRC, encoding="utf-8") as f:
    src = f.read()

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

if mb := ("RUN 2 CELL" in "".join(nb["cells"][-1]["source"])):
    pass
if nb["cells"] and "RUN 2 CELL" in "".join(nb["cells"][-1]["source"]):
    nb["cells"][-1]["source"] = [
        "# =========================================================\n"
        "# RUN 2 CELL - FULL-COVERAGE GREEDY-ANCHORED MONITOR (final)\n"
        "# Same code as analysis/monitor_run.py (single source of truth).\n"
        "# Full run (~3-4 h, 2x T4):  just run this cell as-is.\n"
        "# Smoke test:  import os\n"
        "#              os.environ['GEODESIC2_LIMIT']='2'\n"
        "#              os.environ['GEODESIC2_MAX_NEW_TOKENS']='48'\n"
        "#              os.environ['GEODESIC2_SEQ']='1'\n"
        "#              then run this cell.\n"
        "# Outputs -> /kaggle/working/geodesic_study2/monitor_*.csv\n"
        "# =========================================================\n",
        src,
    ]
else:
    raise SystemExit("cell 3 not found in expected state")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("notebook cell 3 refreshed with final runner")
