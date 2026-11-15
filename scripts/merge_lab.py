"""Fragment merging after selection: join kept instances separated by <= 2*r pixels.

Merged instances keep their original pixels (no gap filling). Scored on the OOF
dumps with the OOF selector predictions from a postproc_lab run.

Usage: python merge_lab.py --dump=a,b,... --lab=<lab dir> --tau=0.24
Writes <lab dir>/merge_lab.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_filament.postproc import EIGHT, paste_masks  # noqa: E402
import postproc_lab as L  # noqa: E402

RADII = [0, 2, 4, 6, 8, 12, 16]
MAX_AREA = [None, 4000, 1500]  # only merge if the smaller piece is below this area
G: dict = {}


def merge(labels: np.ndarray, r: int, max_small: int | None) -> tuple[np.ndarray, int]:
    n = int(labels.max())
    if r == 0 or n < 2:
        return labels, n
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys = slice(max(0, sl[0].start - 2 * r), sl[0].stop + 2 * r)
        xs = slice(max(0, sl[1].start - 2 * r), sl[1].stop + 2 * r)
        m = ndimage.binary_dilation(labels[ys, xs] == i, structure=EIGHT, iterations=2 * r)
        for j in np.unique(labels[ys, xs][m]):
            if j == 0 or j == i:
                continue
            if max_small is not None and min(areas[i], areas[j]) > max_small:
                continue
            parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n + 1)])
    _, inv = np.unique(roots[1:], return_inverse=True)
    remap = np.concatenate([[0], inv + 1])
    out = remap[labels]
    return out, int(out.max())


def process(f):
    rows = G["props"][f]
    z = np.load(G["file_dump"][f] / f"{Path(f).stem}.npz")
    labels, owner = paste_masks(rows, z["prob"], (2048, 2048), 0.65, 96)
    sel = G["pred"][f] > G["tau"]
    remap = np.zeros(len(owner) + 1, dtype=np.int32)
    remap[1:][sel] = np.arange(1, int(sel.sum()) + 1)
    labels = remap[labels]
    views = L.gt_masks_for(G["views"][f], G["anns"])
    out = {}
    for r in RADII:
        for ma in MAX_AREA if r else [None]:
            lab, n = merge(labels, r, ma)
            out[f"r{r}_max{ma}"] = L.score_views(lab, n, views)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--tau", type=float, default=0.24)
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    props, file_dump = {}, {}
    for d in a.dump.split(","):
        p = json.loads((Path(d) / "proposals.json").read_text())["predictions"]
        props.update(p)
        file_dump.update({f: Path(d) for f in p})
    t = np.load(Path(a.lab) / "instances_thr0.65.npz", allow_pickle=True)
    files = list(t["files"])
    offs = np.concatenate([[0], np.cumsum(np.bincount(t["grp"], minlength=len(files)))])
    pred = {f: t["pred"][offs[i]:offs[i + 1]] for i, f in enumerate(files)}
    views, anns = L.load_gt(files)
    G.update(props=props, file_dump=file_dump, pred=pred, tau=a.tau, views=views, anns=anns)
    with Pool(a.workers) as pool:
        res = pool.map(process, files)
    summary = {"images": len(files), "tau": a.tau}
    for key in res[0]:
        summary[key] = L.pq([v for r in res for v in r[key]])
    (Path(a.lab) / "merge_lab.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
