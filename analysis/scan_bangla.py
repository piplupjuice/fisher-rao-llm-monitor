import json
import re

OUT = open("nb_bangla_report.txt", "w", encoding="utf-8")


def has_bangla(s):
    return re.search(r"[\u0980-\u09FF]", s) is not None


for nbname in ["hellio.ipynb", "hellio_backup_run1.ipynb"]:
    nb = json.load(open(nbname, encoding="utf-8"))
    OUT.write("=== %s ===\n" % nbname)
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if has_bangla(src):
            OUT.write("cell %d type=%s\n" % (i, c["cell_type"]))
            for line in src.split("\n"):
                if has_bangla(line):
                    OUT.write("   | %s\n" % line)
        for o in c.get("outputs", []):
            txt = "".join(o.get("text", []))
            if has_bangla(txt):
                OUT.write("cell %d OUTPUT:\n" % i)
                for line in txt[:400].split("\n"):
                    if has_bangla(line):
                        OUT.write("   out| %s\n" % line)
OUT.close()
print("report written")
