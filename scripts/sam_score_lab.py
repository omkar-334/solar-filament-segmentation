"""Does SAM's fine-tuned mask-quality score improve the keep/drop decision?

Inputs (all out of fold, same 5,753 candidates in the same order):
  table.npz     of rescore2_lab.py (stage-1 features X1, stage-2 features F2, F2[:, 0] = stage-1 value)
  per_view.npy  of rescore2_lab.py (matched IoU per annotator view, labels per view)
  oof_score.npy of sam2_refiner.py --score --merge
Compares selecting by the stage-1 value, by the SAM score, by simple mixes, and by a
LightGBM selector (grouped 5 folds) with and without the SAM score, with the exact pooled PQ.

Usage: python sam_score_lab.py --table=... --per_view=... --score=... [--tag=bplus_sc]
Writes results/sam_score_lab/<tag>/{summary.json,sweep.csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
TAUS = [round(float(t), 3) for t in np.arange(0.14, 0.42, 0.01)]


def sweep(pred, pv):
    out = []
    for tau in TAUS:
        tp = fp = fn = 0; s = 0.0; off = 0
        for d in pv:
            n = len(d["pred"]); sel = pred[off:off + n] > tau; off += n
            for yv, ng in zip(np.asarray(d["y_views"]), d["n_gt"]):
                t = int((yv[sel] > 0).sum())
                tp += t; fp += int(sel.sum()) - t; fn += ng - t; s += float(yv[sel].sum())
        out.append({"tau": tau, "tp": tp, "fp": fp, "fn": fn, "pq": s / (tp + 0.5 * fp + 0.5 * fn)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--per_view", required=True)
    ap.add_argument("--score", required=True)
    ap.add_argument("--tag", default="bplus_sc")
    a = ap.parse_args()
    t0 = time.time()
    import lightgbm as lgb

    t = np.load(a.table, allow_pickle=True)
    pv = list(np.load(a.per_view, allow_pickle=True))
    sc = np.load(a.score).astype(np.float32)
    X1, F2, y, grp = t["X1"], t["F2"], t["y"], t["grp"]
    p1 = F2[:, 0]
    assert len(sc) == len(y) == sum(len(d["pred"]) for d in pv)
    assert np.allclose(np.concatenate([d["pred"] for d in pv]), p1, atol=1e-5)
    folds = grp % 5
    kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1)

    def oof(X):
        out = np.zeros(len(y))
        for k in range(5):
            for seed in range(3):
                m = lgb.LGBMRegressor(random_state=seed, **kw).fit(X[folds != k], y[folds != k])
                out[folds == k] += m.predict(X[folds == k]) / 3
        return out

    variants = {
        "stage1_value": p1,
        "sam_score": sc,
        "mean(stage1, sam)": (p1 + sc) / 2,
        "lgb(stage1 feats + value)": oof(np.column_stack([X1, p1])),
        "lgb(+ sam score)": oof(np.column_stack([X1, p1, sc])),
        "lgb(+ sam score + stage2 feats)": oof(np.column_stack([X1, F2, sc])),
    }
    rows, table = [], {}
    for name, pred in variants.items():
        sw = sweep(np.asarray(pred, dtype=np.float64), pv)
        best = max(sw, key=lambda r: r["pq"])
        table[name] = {"best_tau": best["tau"], "best_pq": best["pq"], "tp": best["tp"], "fp": best["fp"],
                       "corr_with_y": float(np.corrcoef(pred, y)[0, 1])}
        rows += [dict(r, variant=name) for r in sw]
    out = RESULTS / "sam_score_lab" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {"args": vars(a), "instances": int(len(y)), "table": table, "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for k, v in table.items():
        print(f"{k:34s} tau {v['best_tau']:.2f} pq {v['best_pq']:.4f} tp {v['tp']} fp {v['fp']} corr {v['corr_with_y']:.3f}")


if __name__ == "__main__":
    main()
