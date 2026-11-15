"""Overlap lab: score candidates WITHOUT pixel claiming (overlapping instances allowed).

The official scorer works on one RLE per row and never checks that rows are
disjoint, so a merged filament and its parts can both be submitted. This lab
measures what that is worth on the out-of-fold dumps, and separates an honest
gain from double counting:

  official   every (gt, pred) pair with IoU > 0.5 is a TP (the organisers' rule)
  strict     one-to-one: per ground truth only the best prediction is a TP,
             any further prediction hitting the same ground truth is a FP

Variants: tau sweep on a grouped 5-fold LightGBM selector, with greedy mask NMS
(drop a candidate whose mask IoU with an already kept one exceeds the limit).

Usage: python overlap_lab.py --dump=a,b,... [--avg_suffix=_r34c384] [--tag=x]
Writes results/overlap_lab/<name>/{summary.json,sweep.csv,instances.npz,rescorer.txt}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_filament.postproc import FEATURES, instance_features  # noqa: E402
import postproc_lab as L  # noqa: E402

THR, MIN_AREA = 0.65, 96
NMS = [None, 0.7, 0.5, 0.3]
TAUS = [round(float(t), 3) for t in np.arange(0.08, 0.42, 0.02)]
G: dict = {}


def load_probs(f: str):
    z = np.load(G["file_dump"][f] / f"{Path(f).stem}.npz")
    probs, yolo = z["prob"], z["yolo"]
    if G["avg_suffix"] and len(probs):
        p2 = np.load(Path(str(G["file_dump"][f]) + G["avg_suffix"]) / f"{Path(f).stem}.npz")["prob"]
        assert len(p2) == len(probs), f
        size = max(probs.shape[-1], p2.shape[-1])

        def up(a):
            if a.shape[-1] == size:
                return a
            return np.stack([cv2.resize(x, (size, size), interpolation=cv2.INTER_LINEAR) for x in a])

        probs = ((up(probs).astype(np.uint16) + up(p2)) // 2).astype(np.uint8)
        yolo = up(yolo)
    return probs, yolo


def process(f: str):
    rows = G["props"][f]
    probs, yolo = load_probs(f)
    gray = cv2.imread(str(L.DATA / "train_images" / f), cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    views = L.gt_masks_for(G["views"][f], G["anns"])
    owner, wins, masks = [], [], []
    for k, r in enumerate(rows):
        x0, y0, side = r["crop"]
        m = cv2.resize(probs[k].astype(np.float32) / 255.0, (side, side), interpolation=cv2.INTER_LINEAR) >= THR
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
        if xb <= xa or yb <= ya:
            continue
        m = m[ya - y0:yb - y0, xa - x0:xb - x0]
        if int(m.sum()) < MIN_AREA:
            continue
        owner.append(k); wins.append((ya, yb, xa, xb)); masks.append(m)
    n = len(owner)
    areas = np.array([m.sum() for m in masks], dtype=np.float64)
    # Features: one single-instance label map per candidate (no claiming).
    feats = []
    scores = np.array([rows[k]["score"] for k in owner])
    for i in range(n):
        lab = np.zeros((h, w), np.uint16)
        ya, yb, xa, xb = wins[i]
        lab[ya:yb, xa:xb][masks[i]] = 1
        x = instance_features(lab, [owner[i]], rows, probs, yolo, gray)[0]
        x[1] = float((scores > scores[i]).sum())
        x[15] = float(n)
        feats.append(x)
    X = np.asarray(feats, np.float32).reshape(n, len(FEATURES))
    # Candidate-candidate mask IoU.
    full = np.zeros((n, h, w), bool) if n else np.zeros((0, h, w), bool)
    for i in range(n):
        ya, yb, xa, xb = wins[i]
        full[i, ya:yb, xa:xb] = masks[i]
    cc = np.zeros((n, n), np.float32)
    for i in range(n):
        ya, yb, xa, xb = wins[i]
        inter = (full[:, ya:yb, xa:xb] & masks[i]).sum(axis=(1, 2)).astype(np.float64)
        cc[i] = inter / (areas + areas[i] - inter + 1e-9)
    # Per view: matched ground truth index (-1 if none) and its IoU, per candidate.
    hit_gt = np.full((len(views), n), -1, np.int32)
    hit_iou = np.zeros((len(views), n), np.float32)
    for vi, gts in enumerate(views):
        for gi, g in enumerate(gts):
            ga = float(g.sum())
            for i in range(n):
                ya, yb, xa, xb = wins[i]
                inter = float((g[ya:yb, xa:xb] & masks[i]).sum())
                iou = inter / (areas[i] + ga - inter + 1e-9)
                if iou > 0.5 and iou > hit_iou[vi, i]:
                    hit_gt[vi, i], hit_iou[vi, i] = gi, iou
    return {"file": f, "owner": owner, "X": X, "cc": cc, "hit_gt": hit_gt, "hit_iou": hit_iou,
            "n_gt": [len(g) for g in views]}


def score(results, preds, tau, nms):
    """Pooled PQ under the official (pairs) and strict (one-to-one) rules."""
    o = dict(tp=0, fp=0, fn=0, s=0.0)
    st = dict(tp=0, fp=0, fn=0, s=0.0)
    kept = 0
    for r, p in zip(results, preds):
        sel = np.nonzero(p > tau)[0]
        sel = sel[np.argsort(-p[sel], kind="stable")]
        if nms is not None:
            keep = []
            for i in sel:
                if all(r["cc"][i, j] <= nms for j in keep):
                    keep.append(i)
            sel = np.array(keep, dtype=int)
        kept += len(sel)
        for vi, ng in enumerate(r["n_gt"]):
            g, iou = r["hit_gt"][vi, sel], r["hit_iou"][vi, sel]
            hit = g >= 0
            o["tp"] += int(hit.sum()); o["fp"] += int((~hit).sum())
            o["fn"] += ng - len(set(g[hit].tolist())); o["s"] += float(iou[hit].sum())
            best = {}
            for gi, v in zip(g[hit], iou[hit]):
                best[gi] = max(best.get(gi, 0.0), float(v))
            st["tp"] += len(best); st["fp"] += len(sel) - len(best)
            st["fn"] += ng - len(best); st["s"] += sum(best.values())

    def fin(d):
        return d["s"] / max(1e-9, d["tp"] + 0.5 * d["fp"] + 0.5 * d["fn"])

    return {"tau": tau, "nms": "none" if nms is None else nms, "kept": kept,
            "pq_official": fin(o), "tp": o["tp"], "fp": o["fp"], "fn": o["fn"],
            "pq_strict": fin(st), "tp_strict": st["tp"], "fp_strict": st["fp"], "fn_strict": st["fn"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--avg_suffix", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    dumps = [Path(d) for d in a.dump.split(",")]
    out = L.RESULTS / "overlap_lab" / (dumps[0].name + f"_x{len(dumps)}" + a.tag)
    out.mkdir(parents=True, exist_ok=True)
    props, file_dump = {}, {}
    for d in dumps:
        p = json.loads((d / "proposals.json").read_text())["predictions"]
        props.update(p)
        file_dump.update({f: d for f in p})
    views, anns = L.load_gt(sorted(props))
    G.update(props=props, file_dump=file_dump, views=views, anns=anns, avg_suffix=a.avg_suffix)
    with Pool(a.workers) as pool:
        results = pool.map(process, sorted(views))

    import lightgbm as lgb
    X = np.concatenate([r["X"] for r in results])
    y = np.concatenate([r["hit_iou"].mean(axis=0) if len(r["n_gt"]) else np.zeros(len(r["owner"]))
                        for r in results])
    sizes = [len(r["owner"]) for r in results]
    grp = np.repeat(np.arange(len(results)), sizes)
    folds = np.arange(len(results)) % 5
    pred = np.zeros_like(y)
    kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    for k in range(5):
        tr = folds[grp] != k
        pred[~tr] = lgb.LGBMRegressor(random_state=k, **kw).fit(X[tr], y[tr]).predict(X[~tr])
    lgb.LGBMRegressor(random_state=0, **kw).fit(X, y).booster_.save_model(str(out / "rescorer.txt"))
    offs = np.concatenate([[0], np.cumsum(sizes)])
    preds = [pred[offs[i]:offs[i + 1]] for i in range(len(results))]
    sweep = [score(results, preds, tau, nms) for nms in NMS for tau in TAUS]
    with (out / "sweep.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(sweep[0])); wr.writeheader(); wr.writerows(sweep)
    np.savez_compressed(out / "instances.npz", X=X, y=y, pred=pred, grp=grp,
                        files=np.array([r["file"] for r in results]),
                        owner=np.concatenate([np.asarray(r["owner"], np.int32) for r in results]))
    best = {}
    for nms in NMS:
        rows = [s for s in sweep if s["nms"] == ("none" if nms is None else nms)]
        best[str(nms)] = {"best_official": max(rows, key=lambda s: s["pq_official"]),
                          "best_strict": max(rows, key=lambda s: s["pq_strict"])}
    summary = {"dumps": [str(d) for d in dumps], "avg_suffix": a.avg_suffix, "images": len(results),
               "candidates": int(len(y)), "refiner_thr": THR, "min_area": MIN_AREA,
               "selector_corr": float(np.corrcoef(pred, y)[0, 1]), "best": best,
               "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
