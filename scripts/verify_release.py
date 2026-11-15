"""Clean-room check of a hand-in package on a GPU machine.

Takes the package tree written by scripts/make_release.py --pack (without weights), builds
a fresh virtual environment from its requirements.txt, attaches the weight files after
checking their hashes against weights/MANIFEST.json, and then, using only that tree:
  1. runs the tests,
  2. predicts the full test set with scripts/predict.py and compares with submission.csv,
  3. executes the notebook.

Usage: python verify_release.py --version=v7|v11 [--tar=/data/release/<version>_code.tar.gz]
           [--weights=/data/weights_<version>] [--limit=0]
Writes $SOLAR_RESULTS/verify_release/<version>/{summary.json,pip_freeze.txt,pytest.txt,
solar_filament_pipeline.ipynb,predict/summary.json,predict/per_image.csv,submission_repro.csv}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v7", "v11"])
    ap.add_argument("--tar", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    v = a.version
    tar = Path(a.tar or f"/data/release/{v}_code.tar.gz")
    wsrc = Path(a.weights or f"/data/weights_{v}")
    test = Path(os.environ["SOLAR_TEST"])
    out = Path(os.environ.get("SOLAR_RESULTS", "results")) / "verify_release" / v
    shutil.rmtree(out, ignore_errors=True)
    (out / "predict").mkdir(parents=True)
    rel = Path("/tmp/release") / v
    shutil.rmtree(rel, ignore_errors=True)
    rel.mkdir(parents=True)
    with tarfile.open(tar) as tf:
        tf.extractall(rel)
    summary = {"version": v, "tar": str(tar), "tar_sha256": sha256(tar), "steps": {}}

    manifest = json.loads((rel / "weights" / "MANIFEST.json").read_text())
    for name, meta in manifest.items():
        dst = rel / "weights" / name
        src = dst if dst.is_file() else wsrc / name   # small files travel inside the tarball
        assert src.stat().st_size == meta["bytes"] and sha256(src) == meta["sha256"], f"weight mismatch {name}"
        if src is not dst:
            dst.symlink_to(src)
    summary["weights"] = {"files": len(manifest), "bytes": sum(m["bytes"] for m in manifest.values()), "hashes_ok": True}

    # Environment from the package's own requirements.txt, nothing inherited. The variables
    # of the hosting job that redirect inputs and outputs are removed.
    env = {k: val for k, val in os.environ.items() if k not in ("PYTHONPATH", "SOLAR_RESULTS", "SOLAR_TEST", "SOLAR_DATA")}
    venv = Path("/tmp/venv") / v
    shutil.rmtree(venv, ignore_errors=True)
    t1 = time.time()
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    py = str(venv / "bin" / "python")
    subprocess.run([py, "-m", "pip", "install", "-q", "-r", str(rel / "requirements.txt")], check=True, env=env)
    # This container has no libGL, which the OpenCV wheel of requirements.txt needs (the README
    # names the system libraries). The headless build of the same OpenCV version is put on top.
    if subprocess.run([py, "-c", "import cv2"], capture_output=True, env=env).returncode != 0:
        ocv = subprocess.run([py, "-c", "import importlib.metadata as m; print(m.version('opencv-python'))"],
                             capture_output=True, text=True, env=env).stdout.strip()
        subprocess.run([py, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", f"opencv-python-headless=={ocv}"], check=True, env=env)
        summary["container_workaround"] = f"no libGL in this container: opencv-python-headless=={ocv} installed over opencv-python=={ocv}"
    (out / "pip_freeze.txt").write_text(subprocess.run([py, "-m", "pip", "freeze"], capture_output=True, text=True, env=env).stdout)
    summary["steps"]["install"] = {"seconds": round(time.time() - t1, 1)}

    def run(name, cmd, log=None):
        t = time.time()
        r = subprocess.run(cmd, cwd=rel, env=env, capture_output=True, text=True)
        (out / (log or f"{name}.log")).write_text(r.stdout + r.stderr)
        print((r.stdout + r.stderr)[-3000:], flush=True)
        summary["steps"][name] = {"returncode": r.returncode, "seconds": round(time.time() - t, 1)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return r.returncode

    run("tests", [py, "-m", "pytest", "-q", "tests/test_rubric_dice.py", "tests/test_scorer_robust.py"], "pytest.txt")
    cmd = [py, "scripts/predict.py", f"--images={test}", "--weights=weights", "--out=submission_repro.csv",
           "--compare=submission.csv", "--tag=repro"] + ([f"--limit={a.limit}"] if a.limit else [])
    if run("predict", cmd) == 0:
        for n in ["summary.json", "per_image.csv"]:
            shutil.copy2(rel / "results" / "predict" / "repro" / n, out / "predict" / n)
        shutil.copy2(rel / "submission_repro.csv", out / "submission_repro.csv")
        summary["predict"] = json.loads((out / "predict" / "summary.json").read_text())
    if run("notebook", [py, "scripts/build_notebook.py", f"--version={v}", "--root=.", "--execute"]) == 0:
        shutil.copy2(rel / "notebooks" / "solar_filament_pipeline.ipynb", out / "solar_filament_pipeline.ipynb")
        summary["notebook"] = json.loads((rel / "results" / "build_notebook" / "summary.json").read_text())
    summary["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    summary["seconds"] = round(time.time() - t0, 1)
    summary["ok"] = all(s.get("returncode", 0) == 0 for s in summary["steps"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    assert summary["ok"], "a step failed"


if __name__ == "__main__":
    main()
