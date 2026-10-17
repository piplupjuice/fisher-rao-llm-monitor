"""Autonomous supervisor for the Kaggle run-2 pipeline.

Polls kernel status -> fetches outputs when complete -> runs the local
verification gate -> when the FULL run is verified, runs the pre-registered
analysis (run2_analysis.py). Logs everything; never re-litigates the protocol.
"""
import os
import subprocess
import time

SCRIPTS = "C:\\Users\\nehad\\AppData\\Roaming\\Python\\Python313\\Scripts"
ROOT = "C:\\Users\\nehad\\Desktop\\geodesics"
os.environ["Path"] = os.environ["Path"] + ";" + SCRIPTS


def status(slug):
    r = subprocess.run(["kaggle", "kernels", "status", slug],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    return out


def fetch(slug, dest):
    os.makedirs(dest, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", slug, "-p", dest],
                   capture_output=True, text=True)


def wait_for(slug, deadline_hours, label):
    t_end = time.time() + deadline_hours * 3600
    while time.time() < t_end:
        s = status(slug)
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {label}: {s}", flush=True)
        if "complete" in s.lower():
            return True
        if "error" in s.lower() or "cancel" in s.lower():
            return False
        time.sleep(180)
    return False


def data_dir(mode):
    base = os.path.join(ROOT, "kaggle_out", mode, "geodesic_study2")
    return base if os.path.isdir(base) else os.path.join(ROOT, "kaggle_out", mode)


def main():
    log = open(os.path.join(ROOT, "analysis", "supervise_run2.log"),
               "a", buffering=1, encoding="utf-8")
    print("=== supervisor start", time.strftime("%F %T"), flush=True)

    # ---- stage 1: smoke --------------------------------------------------
    ok = wait_for("pikanigga/geodesic-run-2-smoke", 0.8, "smoke")
    fetch("pikanigga/geodesic-run-2-smoke", os.path.join(ROOT, "kaggle_out", "smoke"))
    r = subprocess.run(["python", os.path.join(ROOT, "analysis", "fetch_and_verify.py"),
                        "smoke"], capture_output=True, text=True,
                       cwd=ROOT, env=os.environ)
    print("--- smoke verify ---\n", r.stdout, r.stderr, flush=True)
    if not ok:
        print("!!! smoke kernel not complete; continuing to monitor anyway", flush=True)

    # ---- stage 2: full ---------------------------------------------------
    ok = wait_for("pikanigga/geodesic-run-2-full", 12.0, "FULL")
    if not ok:
        print("!!! FULL kernel ended abnormally — fetching whatever exists", flush=True)
    fetch("pikanigga/geodesic-run-2-full", os.path.join(ROOT, "kaggle_out", "full"))
    r = subprocess.run(["python", os.path.join(ROOT, "analysis", "fetch_and_verify.py"),
                        "full"], capture_output=True, text=True,
                       cwd=ROOT, env=os.environ)
    print("--- FULL verify ---\n", r.stdout, r.stderr, flush=True)

    # ---- stage 3: pre-registered analysis (only if gate passed) ----------
    ddir = data_dir("full")
    answ = os.path.join(ddir, "monitor_answers_part0.csv")
    if os.path.exists(answ):
        a = open(answ, encoding="utf-8").readline()
        n_rows = sum(1 for _ in open(answ, encoding="utf-8")) - 1
        print(f"FULL answers rows: {n_rows}", flush=True)
        if "GATE VERDICT: ALL PASS" in open(os.path.join(
                ROOT, "analysis", "supervise_run2.log"),
                encoding="utf-8").read()[-8000:]:
            r = subprocess.run(
                ["python", os.path.join(ROOT, "analysis", "run2_analysis.py"),
                 "--indir", ddir, "--outdir", os.path.join(ROOT, "analysis", "outputs2")],
                capture_output=True, text=True, cwd=ROOT, env=os.environ)
            print("--- run2_analysis ---\n", r.stdout, r.stderr, flush=True)
        else:
            print("!!! verification gate did not pass: analysis NOT claimed automatic; "
                  "inspect manually", flush=True)
    print("=== supervisor end", time.strftime("%F %T"), flush=True)
    log.close()


if __name__ == "__main__":
    main()
