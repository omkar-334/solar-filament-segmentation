"""Second selection after stage 2.

Stage-2 masks are pasted for every candidate whose stage-1 selector value exceeds 0.12
(in value order, so the masks of instances above 0.24 are exactly those of v7). Each
instance gets the stage-1 features, the stage-1 value and stage-2 confidence features:
  s2_mean, s2_mean_seed   mean stage-2 probability inside the new mask / inside the old mask
  s2_frac70               share of the new mask with probability >= 0.7
  s2_soft                 pixels with probability 0.15 to 0.4 relative to the mask area (soft rim)
  s2_iou_seed, s2_area_ratio   agreement of new and old mask
  s2_used                 1 if the stage-2 mask was used
A LightGBM selector is fitted on the matched IoU of the stage-2 masks (grouped 5 folds)
and compared with selecting by the stage-1 value.

Usage: python rescore2_lab.py --dump=a,b,... --lab=<postproc_lab dir> --stage2=<oof_probs.npz> [--avg_suffix] [--pool]
Writes results/rescore2_lab/<lab name>/{summary.json,sweep.csv,table.npz}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_filament import stage2  # noqa: E402
from solar_filament.postproc import matched_iou_per_instance, paste_masks  # noqa: E402
import postproc_lab as L  # noqa: E402

G: dict = {}
S2 = ["s1_value", "s2_mean", "s2_mean_seed", "s2_frac70", "s2_soft", "s2_iou_seed", "s2_area_ratio", "s2_used"]
TAUS = [round(float(t), 3) for t in np.arange(0.14, 0.40, 0.02)]


def process(f: str):
    rows, probs, _ = L.load_candidates(f)
    lab, owner = paste_masks(rows, probs, (2048, 2048), G["thr"], 96)
    p = G["pred"][f]
    n = len(owner)
    cand = p > G["floor"]
    keep = np.concatenate([[False], cand])
    lab1 = np.where(keep[lab], lab, 0).astype(np.uint16)
    items = [(i, M, pr) for i, M, pr in G["s2"].get(f, []) if cand[i]]
    new, used = stage2.paste_refined(lab1, np.where(cand, p, -1.0), items, G["t"])
    wins = {i: stage2.window_prob(pr, M, lab.shape) for i, M, pr in items}
    idx = np.nonzero(cand)[0]
    feats = np.zeros((len(idx), len(S2)), np.float32)
    sl1 = ndimage.find_objects(lab1)
    for r, i in enumerate(idx):
        feats[r, 0] = p[i]
        if i not in wins or sl1[i] is None:
            continue
        x0, y0, x1, y1, pw = wins[i]
        m_new = new[y0:y1, x0:x1] == i + 1
        m_old = lab1[y0:y1, x0:x1] == i + 1
        a_new, a_old = float(m_new.sum()), float(m_old.sum())
        inter = float((m_new & m_old).sum())
        feats[r, 1:] = [pw[m_new].mean() if a_new else 0, pw[m_old].mean() if a_old else 0,
                        (pw[m_new] >= 0.7).mean() if a_new else 0,
                        float(((pw >= 0.15) & (pw < 0.4)).sum()) / max(1.0, a_new),
                        inter / max(1.0, a_new + a_old - inter), a_new / max(1.0, a_old), used[i] == "stage2"]
    views = L.gt_masks_for(G["views"][f], G["anns"])
    yv = np.stack([matched_iou_per_instance(new, n, g)[idx] for g in views]) if views else np.zeros((1, len(idx)))
    return {"idx": idx, "feats": feats, "y_views": yv, "n_gt": [len(g) for g in views]}


def sweep(pred, res):
    out = []
    for tau in TAUS:
        tp = fp = fn = 0; s = 0.0; off = 0
        for r in res:
            n = len(r["idx"])
            sel = pred[off:off + n] > tau
            off += n
            for yv, ng in zip(r["y_views"], r["n_gt"]):
                t = int((yv[sel] > 0).sum())
                tp += t; fp += int(sel.sum()) - t; fn += ng - t; s += float(yv[sel].sum())
        out.append({"tau": tau, "tp": tp, "fp": fp, "fn": fn, "pq": s / (tp + 0.5 * fp + 0.5 * fn)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--avg_suffix", default="")
    ap.add_argument("--pool", default="")
    ap.add_argument("--thr", type=float, default=0.65)
    ap.add_argument("--t", type=float, default=0.40)
    ap.add_argument("--floor", type=float, default=0.12)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--raster", default="coco", help="polygon rasteriser for the labels: coco, cv2, pil, pil_outline")
    a = ap.parse_args()
    import os
    os.environ["SOLAR_RASTER"] = a.raster
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
    files = list(t["files"])
    offs = np.concatenate([[0], np.cumsum(np.bincount(t["grp"], minlength=len(files)))])
    pred1 = {f: t["pred"][offs[i]:offs[i + 1]] for i, f in enumerate(files)}
    z = np.load(a.stage2, allow_pickle=True)
    s2 = {}
    for f, i, M, pr in zip(z["files"], z["idx"], z["M"], z["probs"]):
        s2.setdefault(str(f), []).append((int(i), M, pr))
    views, anns = L.load_gt(files)
    G.update(pred=pred1, s2=s2, views=views, anns=anns, thr=a.thr, t=a.t, floor=a.floor)
    with Pool(a.workers) as pool:
        res = pool.map(process, files, chunksize=4)
    X1 = np.concatenate([t["X"][offs[i]:offs[i + 1]][r["idx"]] for i, r in enumerate(res)])
    F2 = np.concatenate([r["feats"] for r in res])
    y = np.concatenate([r["y_views"].mean(axis=0) for r in res])
    grp = np.repeat(np.arange(len(files)), [len(r["idx"]) for r in res])
    folds = (np.arange(len(files)) % 5)[grp]
    kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    out = L.RESULTS / "rescore2_lab" / (Path(a.lab).name + ("" if a.raster == "coco" else f"_{a.raster}_t{a.t}"))
    out.mkdir(parents=True, exist_ok=True)
    table, rows_out = {}, []
    sets = {"stage1_value": None, "refit_stage1_features": X1, "stage1+value": np.hstack([X1, F2[:, :1]]),
            "stage1+stage2_features": np.hstack([X1, F2]), "stage2_features_only": F2}
    for name, F in sets.items():
        if F is None:
            pred = F2[:, 0].astype(np.float64)
        else:
            pred = np.zeros(len(y))
            for k in range(5):
                tr = folds != k
                pred[~tr] = lgb.LGBMRegressor(random_state=k, **kw).fit(F[tr], y[tr]).predict(F[~tr])
        sw = sweep(pred, res)
        best = max(sw, key=lambda r: r["pq"])
        table[name] = {**best, "corr": float(np.corrcoef(pred, y)[0, 1]),
                       "at_0.24": next(r for r in sw if r["tau"] == 0.24)}
        rows_out += [dict(r, features=name) for r in sw]
        print(name, table[name], flush=True)
    full = np.hstack([X1, F2])
    final = lgb.LGBMRegressor(random_state=0, **kw).fit(full, y)
    final.booster_.save_model(str(out / "rescorer2.txt"))
    gain = dict(zip([f"x{i}" for i in range(X1.shape[1])] + S2, final.booster_.feature_importance("gain").round(1).tolist()))
    np.savez_compressed(out / "table.npz", X1=X1, F2=F2, y=y, grp=grp, files=np.array(files))
    # PQ by number of annotator views of the image (selection by the stage-1 value).
    strata = {}
    offs2 = np.concatenate([[0], np.cumsum([len(r["idx"]) for r in res])])
    for nv in (1, 2, 3):
        sub = [i for i, r in enumerate(res) if len(r["n_gt"]) == nv]
        if sub:
            pr = np.concatenate([F2[offs2[i]:offs2[i + 1], 0] for i in sub])
            strata[str(nv)] = {"images": len(sub), "sweep": sweep(pr, [res[i] for i in sub])}
    (out / "view_strata.json").write_text(json.dumps(strata, indent=2, default=float))
    np.save(out / "per_view.npy", np.array([{"file": f, "y_views": r["y_views"], "n_gt": r["n_gt"],
                                             "pred": F2[offs2[i]:offs2[i + 1], 0]} for i, (f, r) in enumerate(zip(files, res))], dtype=object))
    with (out / "sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    summary = {"args": vars(a), "images": len(files), "instances": int(len(y)), "table": table,
               "gain": dict(sorted(gain.items(), key=lambda kv: -kv[1])[:10]), "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
