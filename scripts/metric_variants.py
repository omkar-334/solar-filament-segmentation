"""Score the selected predictions under alternative scorer behaviours, and test fixes.

Scorer variants (how the scorer might turn our RLE rows into instances):
  as_is        each submitted row is one instance (our local metric)
  split_cc     each row is split into 8-connected components (findContours-style)
  union_cc     the union of all rows is re-labelled by connected components
Prediction fixes:
  raw          current output
  largest_cc   keep only the largest component of each instance
  sep          largest_cc plus a 1 px gap between different instances

Usage (in the lab container): python metric_variants.py --dump=a,b --lab=<lab dir> --tau=0.2
Writes <lab dir>/metric_variants.json and prints it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_filament.postproc import paste_masks  # noqa: E402
import postproc_lab as L  # noqa: E402

EIGHT = ndimage.generate_binary_structure(2, 2)
G: dict = {}


def largest_cc(labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(labels)
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        m = labels[sl] == i
        lab, n = ndimage.label(m, structure=EIGHT)
        if n > 1:
            m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        out[sl][m] = i
    return out


def separate(labels: np.ndarray) -> np.ndarray:
    """Remove pixels of an instance that 8-touch a different instance."""
    out = labels.copy()
    mx = ndimage.maximum_filter(labels, size=3)
    lab_min = labels.astype(np.int32).copy()
    lab_min[lab_min == 0] = 1 << 30
    mn = ndimage.minimum_filter(lab_min, size=3)
    touch = (labels > 0) & ((mx != labels) | (mn != labels))
    out[touch] = 0
    return out


def split_cc(labels: np.ndarray) -> np.ndarray:
    out = np.zeros(labels.shape, dtype=np.int32)
    nxt = 1
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        lab, n = ndimage.label(labels[sl] == i, structure=EIGHT)
        for c in range(1, n + 1):
            out[sl][lab == c] = nxt
            nxt += 1
    return out


def union_cc(labels: np.ndarray) -> np.ndarray:
    return ndimage.label(labels > 0, structure=EIGHT)[0]


def compact(labels):
    ids = np.unique(labels)
    ids = ids[ids != 0]
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    return remap[labels], len(ids)


def process(f):
    rows = G["props"][f]
    z = np.load(G["file_dump"][f] / f"{Path(f).stem}.npz")
    labels, owner = paste_masks(rows, z["prob"], (2048, 2048), 0.65, 96)
    sel = G["pred"][f] > G["tau"]
    remap = np.zeros(len(owner) + 1, dtype=np.int32)
    remap[1:][sel] = np.arange(1, int(sel.sum()) + 1)
    labels = remap[labels]
    views = L.gt_masks_for(G["views"][f], G["anns"])
    fixes = {"raw": labels, "largest_cc": largest_cc(labels)}
    fixes["sep"] = separate(fixes["largest_cc"])
    out = {}
    multi = 0
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is not None and ndimage.label(labels[sl] == i, structure=EIGHT)[1] > 1:
            multi += 1
    out["n_instances"] = int(labels.max() and len(np.unique(labels)) - 1)
    out["n_multi_component"] = multi
    for fx, lab in fixes.items():
        for sv, fn in (("as_is", lambda x: x), ("split_cc", split_cc), ("union_cc", union_cc)):
            cl, n = compact(fn(lab))
            out[f"{fx}|{sv}"] = L.score_views(cl, n, views)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    props, file_dump = {}, {}
    for d in a.dump.split(","):
        p = json.loads((Path(d) / "proposals.json").read_text())["predictions"]
        props.update(p)
        file_dump.update({f: Path(d) for f in p})
    t = np.load(Path(a.lab) / "instances_thr0.65.npz", allow_pickle=True)
    files = list(t["files"])
    sizes = np.bincount(t["grp"], minlength=len(files))
    offs = np.concatenate([[0], np.cumsum(sizes)])
    pred = {f: t["pred"][offs[i]:offs[i + 1]] for i, f in enumerate(files)}
    views, anns = L.load_gt(files)
    G.update(props=props, file_dump=file_dump, pred=pred, tau=a.tau, views=views, anns=anns)
    with Pool(a.workers) as pool:
        res = pool.map(process, files)
    summary = {"images": len(files), "tau": a.tau,
               "instances": sum(r["n_instances"] for r in res),
               "multi_component_instances": sum(r["n_multi_component"] for r in res)}
    for key in [k for k in res[0] if "|" in k]:
        summary[key] = L.pq([v for r in res for v in r[key]])
    (Path(a.lab) / "metric_variants.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
