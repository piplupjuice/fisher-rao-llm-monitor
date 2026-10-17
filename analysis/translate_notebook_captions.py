"""Translate the Bangla figure-caption prints in hellio.ipynb cell 1 to plain
English, and clear execution outputs for a clean professional notebook state.

The untouched original is preserved as hellio_backup_run1.ipynb (local only,
excluded from the git push).
"""
import json
import re

NB = "hellio.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

captions = {
    1: ('FIG 1 -> fig1_accuracy_ci.png | Accuracy per method, dev vs test. '
        'The test panel is the final comparison.'),
    2: ('FIG 2 -> fig2_paired_forest.png | Distance from greedy, per method. '
        'If a CI touches zero, that difference is not proven.'),
    3: ('FIG 3 -> fig3_outcome_matrix.png | Red (B) = greedy was right and the '
        'method broke it; green (C) = a broken answer fixed to right.'),
    4: ('FIG 4 -> fig4_risk_coverage.png | Further left = more correct answers '
        'with fewer questions taken. Compare methods at the 80 percent line.'),
    5: ('FIG 5 -> fig5_auroc_bars.png | 0.5 = coin flip; higher is a better '
        'monitor. Focus on the COMBINED group.'),
    6: ('FIG 6 -> fig6_geometry_vs_correctness.png | If the red (wrong) box '
        'stays on top, uncertainty flags errors early.'),
    7: ("FIG 7 -> fig7_efficiency.png | Methods in the red zone cost more time "
        "for less accuracy - the 'not worth it' region."),
    8: ('FIG 8 -> fig8_divergence.png | Which step the paths split at; more '
        'mass on the right means late divergence.'),
}

src = "".join(nb["cells"][1]["source"])
pattern = re.compile(r'print\("FIG (\d) -> .*?"\)\n', re.DOTALL)
count = [0]


def repl(m):
    fig = int(m.group(1))
    count[0] += 1
    indented = captions[fig].replace(". ", ".\n")
    # keep it as one clean single-line print for simplicity
    return 'print("%s")\n' % captions[fig]


new_src = pattern.sub(repl, src)
assert count[0] == 8, f"expected 8 caption prints, replaced {count[0]}"
assert not re.search(r"[\u0980-\u09FF]", new_src), "Bangla still present"

nb["cells"][1]["source"] = [new_src]

# clean execution state across all cells
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# verify
nb2 = json.load(open(NB, encoding="utf-8"))
src012 = "".join("".join(c["source"]) for c in nb2["cells"])
assert not re.search(r"[\u0980-\u09FF]", src012), "Bangla remains in notebook"
import nbformat
nbformat.validate(nbformat.read(NB, as_version=4))
print("notebook captions translated, outputs cleared, nbformat valid")
print("caption example:", new_src[new_src.find('print("FIG 2'):][:140])
