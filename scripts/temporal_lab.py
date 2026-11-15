"""Temporal-neighbour features for the selector.

For every pasted candidate of an image, look at the labels of other training images
taken within --max_days, moved to this image's time by solar rotation
(src/solar_filament/temporal.py). Features per candidate:
  cov_max, cov_wmean, cov_nearest   share of the candidate covered by moved labels (5 px tolerance)
  iou_max, iou_wmean                best IoU with a single moved label
  hit_share                         share of neighbour views in which cov > 0.3
  n_views, min_dt                   valid neighbour views and the smallest time gap
  nb_density                        mean number of labels per neighbour view
A neighbour view is valid for a candidate when the candidate's centre was on the visible
disk at the neighbour's time. The same physical image is never its own neighbour, which
is the situation of a test image (all training labels available, its own unknown).

The selector is then re-fitted with and without these features (same folds and model as
postproc_lab.py) and the exact pooled PQ sweep is reported.

Usage: python temporal_lab.py --dump=a,b,... --lab=<postproc_lab dir> [--avg_suffix=..] [--pool=..]
Writes results/temporal_lab/<lab name>/{summary.json,sweep.csv,features.npz}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_filament.postproc import paste_masks  # noqa: E402
from solar_filament.temporal import obs_days, rotate_points, rotated_polygon_mask  # noqa: E402
import postproc_lab as L  # noqa: E402

G: dict = {}
NAMES = ["cov_max", "cov_wmean", "cov_nearest", "iou_max", "iou_wmean", "hit_share", "n_views", "min_dt", "nb_density"]
K5 = np.ones((11, 11), np.uint8)
TAUS = [round(float(t), 3) for t in np.arange(0.12, 0.40, 0.02)]


def neighbour_maps(f: str, max_days: float, max_n: int):
    """[(dt, label map of moved labels, n labels)] for the nearest other training images."""
    t = G["days"][f]
    near = sorted((abs(G["days"][g] - t), g) for g in G["train_files"] if g != f and abs(G["days"][g] - t) <= max_days)
    out = []
    for _, g in near[:max_n]:
        dt = t - G["days"][g]  # move labels of g forward by dt (negative: backward)
        for view in G["train_views"][g]:
            lab = np.zeros((2048, 2048), np.uint16)
            k = 0
            for ann in G["anns"][view["id"]]:
                m = rotated_polygon_mask(ann.get("segmentation"), dt)
                if m.any():
                    k += 1
                    lab[m] = k
            out.append((dt, lab, len(G["anns"][view["id"]])))
    return out


def candidate_features(lab_c: np.ndarray, n: int, maps) -> np.ndarray:
    """Temporal features [n, len(NAMES)] of the instances 1..n of a label map."""
    feats = np.zeros((n, len(NAMES)), np.float32)
    feats[:, 7] = 9.0
    if n == 0:
        return feats
    area = np.bincount(lab_c.ravel(), minlength=n + 1)[1:].astype(np.float64)
    ys, xs = np.nonzero(lab_c)
    ids = lab_c[ys, xs].astype(np.int64) - 1
    cen = np.stack([np.bincount(ids, weights=xs, minlength=n), np.bincount(ids, weights=ys, minlength=n)], 1) / np.maximum(area, 1)[:, None]
    covs, ious, ws, dts = [], [], [], []
    for dt, lab_g, n_g in maps:
        _, vis = rotate_points(cen, -dt)  # where the candidate was at the neighbour's time
        union = cv2.dilate((lab_g > 0).astype(np.uint8), K5).astype(bool)
        cov = np.bincount(lab_c[union], minlength=n + 1)[1:] / np.maximum(area, 1)
        iou = np.zeros(n)
        both = (lab_c > 0) & (lab_g > 0)
        if both.any():
            ng = int(lab_g.max())
            ga = np.bincount(lab_g.ravel(), minlength=ng + 1).astype(np.float64)
            pair, cnt = np.unique(lab_c[both].astype(np.int64) * (ng + 1) + lab_g[both], return_counts=True)
            ci, gi = pair // (ng + 1) - 1, pair % (ng + 1)
            np.maximum.at(iou, ci, cnt / (area[ci] + ga[gi] - cnt))
        covs.append(np.where(vis, cov, np.nan)); ious.append(np.where(vis, iou, np.nan))
        ws.append(np.where(vis, 1 / (0.5 + abs(dt)), 0.0)); dts.append(np.where(vis, abs(dt), np.nan))
    if not covs:
        return feats
    C, I, Wt, D = np.array(covs), np.array(ious), np.array(ws), np.array(dts)
    valid = ~np.isnan(C)
    nv = valid.sum(0)
    C0, I0 = np.nan_to_num(C), np.nan_to_num(I)
    wsum = np.maximum(Wt.sum(0), 1e-9)
    nearest = np.where(valid, D, 99).argmin(0)
    feats[:, 0] = C0.max(0); feats[:, 1] = (C0 * Wt).sum(0) / wsum
    feats[:, 2] = np.where(nv > 0, C0[nearest, np.arange(n)], 0)
    feats[:, 3] = I0.max(0); feats[:, 4] = (I0 * Wt).sum(0) / wsum
    feats[:, 5] = ((C0 > 0.3) & valid).sum(0) / np.maximum(nv, 1)
    feats[:, 6] = nv
    feats[:, 7] = np.where(nv > 0, np.where(valid, D, 99).min(0), 9.0)
    dens = np.array([m[2] for m in maps], dtype=np.float64)
    feats[:, 8] = (valid * dens[:, None]).sum(0) / np.maximum(nv, 1)
    return feats


def process(f: str):
    rows, probs, _ = L.load_candidates(f)
    lab, owner = paste_masks(rows, probs, (2048, 2048), G["thr"], 96)
    return candidate_features(lab, len(owner), neighbour_maps(f, G["max_days"], G["max_n"]))


def sweep(pred, sizes, y_views, n_gt):
    out = []
    for tau in TAUS:
        tp = fp = fn = 0; s = 0.0; off = 0
        for n, yv, ng in zip(sizes, y_views, n_gt):
            sel = pred[off:off + n] > tau
            off += n
            for v in range(len(ng)):
                row = np.asarray(yv[v], dtype=np.float64)[sel] if n else np.zeros(0)
                t = int((row > 0).sum())
                tp += t; fp += int(sel.sum()) - t; fn += int(ng[v]) - t; s += float(row.sum())
        out.append({"tau": tau, "tp": tp, "fp": fp, "fn": fn, "pq": s / (tp + 0.5 * fp + 0.5 * fn)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--avg_suffix", default="")
    ap.add_argument("--pool", default="")
    ap.add_argument("--thr", type=float, default=0.65)
    ap.add_argument("--max_days", type=float, default=4.0)
    ap.add_argument("--max_n", type=int, default=8)
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    import lightgbm as lgb

    props, file_dump = {}, {}
    for d in a.dump.split(","):
        pr = json.loads((Path(d) / "proposals.json").read_text())["predictions"]
        props.update(pr); file_dump.update({f: Path(d) for f in pr})
    pools = []
    for group in [g for g in a.pool.split("+") if g]:
        pd_, pp_ = {}, {}
        for d in group.split(","):
            pp = json.loads((Path(d) / "proposals.json").read_text())["predictions"]
            pp_.update(pp); pd_.update({f: Path(d) for f in pp})
        pools.append((pd_, pp_))
    L.G.update(props=props, file_dump=file_dump, avg_suffix=a.avg_suffix, pools=pools)
    t = np.load(Path(a.lab) / f"instances_thr{a.thr}.npz", allow_pickle=True)
    X, y, grp, files = t["X"], t["y"], t["grp"], list(t["files"])
    y_views, n_gt = list(t["y_views"]), list(t["n_gt"])
    sizes = np.bincount(grp, minlength=len(files))
    views, anns = L.load_gt(files)
    G.update(train_files=files, train_views=views, anns=anns, days={f: obs_days(f) for f in files},
             thr=a.thr, max_days=a.max_days, max_n=a.max_n)
    with Pool(a.workers) as pool:
        T = pool.map(process, files, chunksize=4)
    assert [len(x) for x in T] == list(sizes), "candidate counts differ from the lab table"
    T = np.concatenate(T)
    out = L.RESULTS / "temporal_lab" / Path(a.lab).name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "features.npz", T=T, names=np.array(NAMES), files=np.array(files), grp=grp)
    folds = (np.arange(len(files)) % 5)[grp]
    kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    table, rows_out = {}, []
    sets = {"base": X, "base+temporal": np.hstack([X, T]), "base+coverage_only": np.hstack([X, T[:, :6]]),
            "temporal_only": T}
    for name, F in sets.items():
        pred = np.zeros(len(y))
        for k in range(5):
            tr = folds != k
            pred[~tr] = lgb.LGBMRegressor(random_state=k, **kw).fit(F[tr], y[tr]).predict(F[~tr])
        sw = sweep(pred, sizes, y_views, n_gt)
        best = max(sw, key=lambda r: r["pq"])
        table[name] = {**best, "corr": float(np.corrcoef(pred, y)[0, 1])}
        rows_out += [dict(r, features=name) for r in sw]
        np.save(out / f"pred_{name.replace('+', '_')}.npy", pred)
        print(name, table[name], flush=True)
    final = lgb.LGBMRegressor(random_state=0, **kw).fit(sets["base+temporal"], y)
    final.booster_.save_model(str(out / f"rescorer_temporal_thr{a.thr}.txt"))
    gain = dict(zip([f"x{i}" for i in range(X.shape[1])] + NAMES, final.booster_.feature_importance("gain").round(1).tolist()))
    with (out / "sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    matched = y > 0
    summary = {"args": vars(a), "images": len(files), "candidates": int(len(y)), "table": table,
               "feature_means": {n: {"matched": float(T[matched, i].mean()), "unmatched": float(T[~matched, i].mean())}
                                 for i, n in enumerate(NAMES)},
               "candidates_without_neighbour": float((T[:, 6] == 0).mean()),
               "temporal_gain": {k: v for k, v in sorted(gain.items(), key=lambda kv: -kv[1]) if k in NAMES},
               "top_gain": dict(sorted(gain.items(), key=lambda kv: -kv[1])[:8]), "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
