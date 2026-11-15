"""Collect the weight files of the final pipeline into one directory with a manifest.

Usage (on the Modal volume): python collect_weights.py --out=/data/weights_v7
Writes <out>/<file> for every entry below and <out>/MANIFEST.json (source, bytes, sha256);
a copy of the manifest goes to results/collect_weights/summary.json.
"""
import argparse, hashlib, json, os, shutil
from pathlib import Path

RESULTS = Path(os.environ.get("SOLAR_RESULTS", Path(__file__).resolve().parents[1] / "results"))
SOURCES = {
    "detector_1024.pt": "/data/runs/full707_yolo11s_s43_lr002/train/weights/last.pt",
    "detector_1536.pt": "/data/runs/final707_1536_det/train/weights/last.pt",
    "refiner_r18_1024.pt": "/data/runs/final707_refiner/last.pt",
    "refiner_r34c384_1024.pt": "/data/runs/final707_refiner_r34c384/last.pt",
    "refiner_r18_1536.pt": "/data/runs/final707_1536_refiner/last.pt",
    "unet_dense.pt": "/data/runs/unet_full707/last.pt",
    "selector.txt": "/data/results/postproc_lab/dump_kf5b_fold0_x5_avg384_pool1536/rescorer_thr0.65.txt",
    **{f"stage2_fold{k}.pt": f"/data/runs/stage2_v6_fold{k}.pt" for k in range(5)},
}
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--version", default="v7")
a = ap.parse_args()
if a.version == "v11":
    SOURCES = {k: v for k, v in SOURCES.items() if not k.startswith("stage2_")}
    SOURCES.update({f"sam_{t}_fold{k}.pt": f"/data/runs/sam2_{t}_fold{k}.pt" for t in ("bplus", "large") for k in range(5)})
    SOURCES.update({f"sam_score_fold{k}.pt": f"/data/runs/sam2_bplus_sc_fold{k}.pt" for k in range(5)})
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
manifest = {}
for name, src in SOURCES.items():
    shutil.copyfile(src, out / name)
    h = hashlib.sha256((out / name).read_bytes()).hexdigest()
    manifest[name] = {"source": src, "bytes": (out / name).stat().st_size, "sha256": h}
(out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
(RESULTS / "collect_weights").mkdir(parents=True, exist_ok=True)
(RESULTS / "collect_weights" / ("summary.json" if a.version == "v7" else f"summary_{a.version}.json")).write_text(json.dumps(manifest, indent=2))
print(json.dumps({k: v["bytes"] for k, v in manifest.items()}, indent=2), "total MB", sum(v["bytes"] for v in manifest.values()) // 2**20)
