"""Selector lab: can the keep/drop model rank candidates better?

Reuses the candidate table of a postproc_lab run (features, per-view matched
IoU) and compares feature sets and model variants with the same grouped 5-fold
split and the exact pooled PQ sweep.

Feature sets
  base   the 17 instance features of the v5 selector
  ctx    + image context (candidate counts, score statistics, within-image ranks, neighbours)
  det2   + agreement with a second, independently trained detector (1536 px folds)
  all    ctx + det2
Models
  lgb        the v5 LightGBM regressor
  lgb_w      same, each candidate weighted by its number of annotator views (as the metric does)
  lgb_big    larger, slower model, 3 seeds averaged, view weighted
  hit_x_iou  P(match) x E[IoU | match], view weighted

Usage: python selector_lab.py --lab=<postproc_lab dir> --dump=a,b,... --aux=c,d,... [--pool=e,f,...+g,h,...]
Writes results/selector_lab/<lab name>/{summary.json,sweep.csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

RESULTS = Path(os.environ.get("SOLAR_RESULTS", Path(__file__).resolve().parents[1] / "results"))
TAUS = [round(float(t), 3) for t in np.arange(0.12, 0.40, 0.02)]


def box_iou(b, B):
    if len(B) == 0:
        return np.zeros(0)
    x0 = np.maximum(b[0], B[:, 0]); y0 = np.maximum(b[1], B[:, 1])
    x1 = np.minimum(b[2], B[:, 2]); y1 = np.minimum(b[3], B[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-9)


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
    ap.add_argument("--lab", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--pool", default="", help="pooled detectors of the lab run ('+' between detectors)")
    a = ap.parse_args()
    t0 = time.time()
    import lightgbm as lgb

    t = np.load(Path(a.lab) / "instances_thr0.65.npz", allow_pickle=True)
    X, y, grp, files, owner = t["X"], t["y"], t["grp"], list(t["files"]), t["owner"]
    y_views, n_gt = list(t["y_views"]), list(t["n_gt"])
    sizes = np.bincount(grp, minlength=len(files))
    offs = np.concatenate([[0], np.cumsum(sizes)])
    props, aux = {}, {}
    for d in a.dump.split(","):
        props.update(json.loads((Path(d) / "proposals.json").read_text())["predictions"])
    for group in [g for g in a.pool.split("+") if g]:  # same row order as postproc_lab.load_candidates
        pp = {}
        for d in group.split(","):
            pp.update(json.loads((Path(d) / "proposals.json").read_text())["predictions"])
        props = {f: list(r) + list(pp[f]) for f, r in props.items()}
    for d in a.aux.split(","):
        aux.update(json.loads((Path(d) / "proposals.json").read_text())["predictions"])
    assert set(files) <= set(aux), "second detector does not cover every image"

    ctx, det2, nviews = [], [], []
    for i, f in enumerate(files):
        sl = slice(offs[i], offs[i + 1])
        n = sizes[i]
        nviews += [len(n_gt[i])] * n
        if n == 0:
            continue
        x = X[sl]
        boxes = np.array([props[f][o]["bbox"] for o in owner[sl]], dtype=np.float64).reshape(-1, 4)
        det = x[:, 0]
        cen = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
        dist = np.hypot(cen[:, None, 0] - cen[None, :, 0], cen[:, None, 1] - cen[None, :, 1])
        np.fill_diagonal(dist, 1e6)
        nn = dist.argmin(axis=1)
        rank = lambda v: (np.argsort(np.argsort(-v)) / max(1, n - 1)) if n > 1 else np.zeros(n)  # noqa: E731
        z = lambda v: (v - v.mean()) / (v.std() + 1e-6)  # noqa: E731
        ctx.append(np.stack([
            np.full(n, n), np.full(n, det.mean()), np.full(n, det.max()), np.full(n, det.std()),
            np.full(n, (det > 0.5).sum()), np.full(n, (det > 0.25).sum()), rank(det), rank(x[:, 2]),
            z(x[:, 12]), z(x[:, 3]), rank(x[:, 3]), np.minimum(dist.min(axis=1), 2048.0) if n > 1 else np.full(n, 2048.0),
            det[nn] if n > 1 else np.zeros(n), (dist < 150).sum(axis=1), (dist < 400).sum(axis=1),
            np.full(n, x[:, 13].mean()),
        ], axis=1))
        B = np.array([r["bbox"] for r in aux[f]], dtype=np.float64).reshape(-1, 4)
        S = np.array([r["score"] for r in aux[f]], dtype=np.float64)
        rows = []
        for b in boxes:
            io = box_iou(b, B)
            if len(io):
                j = int(io.argmax())
                rows.append([io[j], S[j], float((io * S).max()), float((io > 0.5).sum()), len(B), float(S.max())])
            else:
                rows.append([0, 0, 0, 0, 0, 0])
        det2.append(np.asarray(rows))
    ctx = np.concatenate(ctx).astype(np.float32); det2 = np.concatenate(det2).astype(np.float32)
    w_views = np.asarray(nviews, np.float32)
    hit = np.concatenate([(np.stack([np.asarray(r, np.float64) for r in yv]) > 0).mean(axis=0) if s_ else np.zeros(0)
                          for yv, s_ in zip(y_views, sizes)]).astype(np.float32)
    iou_given_hit = np.where(hit > 0, y / np.maximum(hit, 1e-6), 0.0).astype(np.float32)
    sets = {"base": X, "ctx": np.hstack([X, ctx]), "det2": np.hstack([X, det2]), "all": np.hstack([X, ctx, det2])}
    folds = (np.arange(len(files)) % 5)[grp]
    kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    big = dict(n_estimators=900, learning_rate=0.015, num_leaves=31, min_child_samples=30, subsample=0.8,
               subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)

    def oof(F, model):
        pred = np.zeros(len(y), np.float64)
        for k in range(5):
            tr = folds != k
            if model == "lgb":
                pred[~tr] = lgb.LGBMRegressor(random_state=k, **kw).fit(F[tr], y[tr]).predict(F[~tr])
            elif model == "lgb_w":
                pred[~tr] = lgb.LGBMRegressor(random_state=k, **kw).fit(F[tr], y[tr], sample_weight=w_views[tr]).predict(F[~tr])
            elif model == "lgb_big":
                for seed in range(3):
                    pred[~tr] += lgb.LGBMRegressor(random_state=10 * seed + k, **big).fit(
                        F[tr], y[tr], sample_weight=w_views[tr]).predict(F[~tr]) / 3
            elif model == "hit_x_iou":
                ph = lgb.LGBMRegressor(objective="cross_entropy", random_state=k, **kw).fit(
                    F[tr], hit[tr], sample_weight=w_views[tr]).predict(F[~tr])
                m = tr & (hit > 0)
                qi = lgb.LGBMRegressor(random_state=k, **kw).fit(
                    F[m], iou_given_hit[m], sample_weight=(w_views * hit)[m]).predict(F[~tr])
                pred[~tr] = ph * qi
        return pred

    rows_out, table = [], {}
    for sname, F in sets.items():
        for model in ("lgb", "lgb_w", "lgb_big", "hit_x_iou"):
            pred = oof(F, model)
            sw = sweep(pred, sizes, y_views, n_gt)
            best = max(sw, key=lambda r: r["pq"])
            table[f"{sname}|{model}"] = {"best_pq": best["pq"], "tau": best["tau"], "tp": best["tp"], "fp": best["fp"],
                                         "fn": best["fn"], "corr": float(np.corrcoef(pred, y)[0, 1])}
            rows_out += [dict(r, features=sname, model=model) for r in sw]
            print(sname, model, round(best["pq"], 4), best["tau"], flush=True)
    out = RESULTS / "selector_lab" / Path(a.lab).name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    summary = {"lab": a.lab, "images": len(files), "candidates": int(len(y)), "table": table,
               "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
