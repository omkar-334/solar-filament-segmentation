"""Check that every proposal dump has one refiner probability map per proposal row.

Usage: python check_dump.py --dump=a,b,...
Writes results/check_dump/<first dump name>/summary.json and prints it.
"""
import argparse, json, os
from pathlib import Path
import numpy as np

RESULTS = Path(os.environ.get("SOLAR_RESULTS", Path(__file__).resolve().parents[1] / "results"))
ap = argparse.ArgumentParser(); ap.add_argument("--dump", required=True); a = ap.parse_args()
out = {}
for d in a.dump.split(","):
    pr = json.loads((Path(d) / "proposals.json").read_text())["predictions"]
    bad, missing, shapes = [], [], set()
    for f, rows in pr.items():
        p = Path(d) / f"{Path(f).stem}.npz"
        if not p.is_file():
            missing.append(f); continue
        z = np.load(p)
        shapes.add(str(z["prob"].shape[1:]))
        if len(z["prob"]) != len(rows) or len(z["yolo"]) != len(rows):
            bad.append([f, len(rows), int(len(z["prob"])), int(len(z["yolo"]))])
    out[d] = {"images": len(pr), "missing_npz": len(missing), "mismatched": len(bad), "examples": bad[:5],
              "prob_shapes": sorted(shapes), "max_rows": max(len(r) for r in pr.values())}
o = RESULTS / "check_dump" / Path(a.dump.split(",")[0]).name
o.mkdir(parents=True, exist_ok=True)
(o / "summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
