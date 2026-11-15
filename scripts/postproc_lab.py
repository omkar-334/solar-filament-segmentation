"""Local post-processing lab over a dumped validation proposal set.

Steps (all on CPU, grouped by physical image):
  baseline   reproduce the current submission rule (det conf / refiner thr / area)
  misses     classify every missed GT instance (no proposal, near miss, lost in paste)
  rescore    learn expected matched IoU per candidate instance (grouped 5-fold CV)
             and keep candidates whose prediction exceeds tau; sweep tau

Usage: python scripts/postproc_lab.py --dump=results/dumps/dump_val_yolo11s_oofref
Outputs: results/postproc_lab/<dump-name>/{summary.json,instances.csv,sweep.csv,misses.csv}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solar_filament.data import polygon_mask  # noqa: E402
from solar_filament.postproc import (  # noqa: E402
    FEATURES, UNET_FEATURES, instance_features, matched_iou_per_instance, paste_masks,
)

DATA = Path(os.environ.get("SOLAR_DATA", ROOT / "MAGFiLO_1.0_Kaggle_2026/train"))
RESULTS = Path(os.environ.get("SOLAR_RESULTS", ROOT / "results"))
G: dict = {}


def val_files() -> list[str]:
    coco = json.loads((DATA / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
    files = sorted({im["file_name"] for im in coco["images"]})
    rng = random.Random(42)
    rng.shuffle(files)
    return sorted(files[: round(len(files) * 0.2)])


def load_gt(files: list[str] | None = None) -> dict[str, list[list[np.ndarray]]]:
    coco = json.loads((DATA / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    views = defaultdict(list)
    for im in coco["images"]:
        views[im["file_name"]].append(im)
    return {f: views[f] for f in (files or val_files())}, anns


def gt_masks_for(file_views, anns) -> list[list[np.ndarray]]:
    return [
        [polygon_mask(a.get("segmentation"), v["height"], v["width"]).astype(bool) for a in anns[v["id"]]]
        for v in file_views
    ]


def score_views(labels: np.ndarray, n: int, views: list[list[np.ndarray]]):
    out = []
    for gts in views:
        miou = matched_iou_per_instance(labels, n, gts)
        tp = int((miou > 0).sum())
        out.append((tp, n - tp, len(gts) - tp, float(miou.sum())))
    return out


def pq(rows) -> dict:
    tp = sum(r[0] for r in rows); fp = sum(r[1] for r in rows); fn = sum(r[2] for r in rows)
    s = sum(r[3] for r in rows)
    return {"tp": tp, "fp": fp, "fn": fn, "sum_iou": round(s, 3), "sq": s / max(1, tp),
            "pq": s / (tp + 0.5 * fp + 0.5 * fn)}


def load_candidates(f: str):
    """Proposal rows, refiner probabilities and detector masks of one image (base dump,
    optional second-refiner average, optional pooled detectors)."""
    rows = G["props"][f]
    z = np.load(G["file_dump"][f] / f"{Path(f).stem}.npz")
    probs, yolo = z["prob"], z["yolo"]
    if G.get("avg_suffix"):
        # Average with a second refiner's probabilities for the same proposals.
        d2 = Path(str(G["file_dump"][f]) + G["avg_suffix"])
        p2 = np.load(d2 / f"{Path(f).stem}.npz")["prob"]
        assert len(p2) == len(probs), (f, len(p2), len(probs))
        size = max(probs.shape[-1], p2.shape[-1]) if len(probs) else probs.shape[-1]
        def up(a):
            return np.stack([cv2.resize(x, (size, size), interpolation=cv2.INTER_LINEAR) for x in a]) if len(a) and a.shape[-1] != size else a
        if len(probs):
            probs = ((up(probs).astype(np.uint16) + up(p2)) // 2).astype(np.uint8)
            yolo = up(yolo)
    for gi, (pool_dirs, pool_props) in enumerate(G.get("pools", [])):
        # Pool the candidates of a further detector (with its own refiner probabilities).
        z2 = np.load(pool_dirs[f] / f"{Path(f).stem}.npz")
        rows2 = pool_props[f]
        scale = G.get("pool_scale", [])
        if gi < len(scale) and scale[gi] != 1.0:  # lower scores paste later in the pixel claiming
            rows2 = [dict(r, score=r["score"] * scale[gi]) for r in rows2]
        if len(rows2):
            size = probs.shape[-1] if len(probs) else z2["prob"].shape[-1]
            def rs(a):
                return np.stack([cv2.resize(x, (size, size), interpolation=cv2.INTER_LINEAR) for x in a]) if a.shape[-1] != size else a
            p2, y2 = rs(z2["prob"]), rs(z2["yolo"])
            probs = np.concatenate([probs, p2]) if len(probs) else p2
            yolo = np.concatenate([yolo, y2]) if len(yolo) else y2
            rows = list(rows) + list(rows2)
    return rows, probs, yolo


def process_file(f: str):
    gtv, anns = G["views"][f], G["anns"]
    aux = {k: v.get(f, []) for k, v in G.get("aux", {}).items()}
    rows, probs, yolo = load_candidates(f)
    gray = cv2.imread(str(DATA / "train_images" / f), cv2.IMREAD_GRAYSCALE)
    unet = (cv2.imread(str(Path(G["unet"]) / f"{Path(f).stem}.png"), cv2.IMREAD_GRAYSCALE)
            if G.get("unet") else None)
    views = gt_masks_for(gtv, anns)
    res = {"file": f, "n_views": len(views)}
    # Baseline grid.
    grid = {}
    for conf in (0.25, 0.35):
        for thr in (0.55, 0.65):
            for lcc in (False, True):
                keep = np.array([r["score"] >= conf for r in rows], dtype=bool)
                lab, own = paste_masks(rows, probs, gray.shape, thr, 96, keep=keep, largest_cc=lcc)
                grid[(conf, thr, lcc)] = score_views(lab, len(own), views)
    res["grid"] = grid
    # Candidate pool for rescoring: every proposal >= 0.05, refiner thr 0.55 / 0.65.
    cands = {}
    for thr in (0.55, 0.65):
        lab, own = paste_masks(rows, probs, gray.shape, thr, 96)
        feats = instance_features(lab, own, rows, probs, yolo, gray, aux, unet=unet)
        targets = np.stack([matched_iou_per_instance(lab, len(own), g) for g in views]) if views else np.zeros((1, len(own)))
        cands[thr] = {"owner": own, "X": feats, "y": targets.mean(axis=0), "y_views": targets,
                      "n_gt": [len(g) for g in views]}
    res["cands"] = cands
    # Miss analysis at baseline (0.35/0.65): for each GT, best IoU with final preds
    # and with ANY proposal's own refiner mask (proposal-level oracle).
    keep = np.array([r["score"] >= 0.35 for r in rows], dtype=bool)
    lab, own = paste_masks(rows, probs, gray.shape, 0.65, 96, keep=keep)
    all_lab_ious = []
    misses = []
    for vi, gts in enumerate(views):
        for g in gts:
            ga = g.sum()
            inter = np.bincount(lab[g], minlength=len(own) + 1).astype(float)
            pa = np.bincount(lab.ravel(), minlength=len(own) + 1).astype(float)
            iou = inter / (pa + ga - inter + 1e-9); iou[0] = 0
            best_final = float(iou.max())
            # Oracle: each proposal individually pasted (no claiming), any conf.
            best_prop, best_prop_score = 0.0, 0.0
            ys, xs = np.nonzero(g)
            gy0, gy1, gx0, gx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            for k, r in enumerate(rows):
                x0, y0, side = r["crop"]
                if x0 > gx1 or y0 > gy1 or x0 + side < gx0 or y0 + side < gy0:
                    continue
                m = cv2.resize(probs[k].astype(np.float32) / 255, (side, side)) >= 0.5
                full = np.zeros(gray.shape, bool)
                xa, ya = max(0, x0), max(0, y0)
                xb, yb = min(gray.shape[1], x0 + side), min(gray.shape[0], y0 + side)
                full[ya:yb, xa:xb] = m[ya - y0:yb - y0, xa - x0:xb - x0]
                i = (full & g).sum(); u = (full | g).sum()
                if u and i / u > best_prop:
                    best_prop, best_prop_score = i / u, r["score"]
            misses.append({"file": f, "view": vi, "gt_area": int(ga), "best_final_iou": best_final,
                           "best_proposal_iou": best_prop, "best_proposal_score": best_prop_score})
    res["misses"] = misses
    return res


def repaste_file(args):
    """Second pass: re-paste only kept candidates, ordered by predicted value."""
    f, thr, owner, pred, taus = args
    rows = G["props"][f]
    z = np.load(G["file_dump"][f] / f"{Path(f).stem}.npz")
    gray_shape = (2048, 2048)
    views = gt_masks_for(G["views"][f], G["anns"])
    out = {}
    for tau in taus:
        keep = np.zeros(len(rows), bool)
        value = np.full(len(rows), -1.0)
        for o, p in zip(owner, pred):
            if p > tau:
                keep[o] = True
                value[o] = p
        order = np.argsort(-value, kind="stable")
        lab, own = paste_masks(rows, z["prob"], gray_shape, thr, 96, order=order, keep=keep)
        out[tau] = score_views(lab, len(own), views)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--unet", default="", help="dir of U-Net probability PNGs (adds features)")
    ap.add_argument("--tag", default="", help="suffix for the output directory name")
    ap.add_argument("--avg_suffix", default="", help="average refiner probs with <dump><suffix> dumps")
    ap.add_argument("--pool", default="", help="dumps of a second detector whose candidates are pooled in")
    ap.add_argument("--pool_scale", default="", help="score multiplier per pooled detector, e.g. 1,0.1")
    a = ap.parse_args()
    G["pool_scale"] = [float(x) for x in a.pool_scale.split(",") if x]
    t0 = time.time()
    dumps = [Path(d) for d in a.dump.split(",")]
    dump = dumps[0]
    out = RESULTS / "postproc_lab" / (dump.name + (f"_x{len(dumps)}" if len(dumps) > 1 else "") + a.tag)
    out.mkdir(parents=True, exist_ok=True)
    props, aux, file_dump = {}, defaultdict(dict), {}
    for d in dumps:
        payload = json.loads((d / "proposals.json").read_text())
        props.update(payload["predictions"])
        file_dump.update({f: d for f in payload["predictions"]})
        for k, v in payload.get("aux", {}).items():
            aux[k].update(v)
    views, anns = load_gt(sorted(props) if len(dumps) > 1 else None)
    G["aux"] = dict(aux)
    G["file_dump"] = file_dump
    G["unet"] = a.unet
    G["avg_suffix"] = a.avg_suffix
    G["pools"] = []
    for group in [g for g in a.pool.split("+") if g]:  # "+" separates detectors, "," separates their fold dumps
        pool_dirs, pool_props = {}, {}
        for d in group.split(","):
            pp = json.loads((Path(d) / "proposals.json").read_text())["predictions"]
            pool_props.update(pp)
            pool_dirs.update({f: Path(d) for f in pp})
        assert set(props) <= set(pool_props), "pool dumps do not cover every image"
        G["pools"].append((pool_dirs, pool_props))
    assert set(props) == set(views), (len(props), len(views))
    G.update(dump=dump, views=views, anns=anns, props=props)
    with Pool(a.workers) as pool:
        results = pool.map(process_file, sorted(views))
    summary = {"dump": str(dump), "images": len(results),
               "views": sum(r["n_views"] for r in results)}

    # Baseline grid.
    grid = {}
    for key in results[0]["grid"]:
        grid[str(key)] = pq([v for r in results for v in r["grid"][key]])
    summary["baseline_grid"] = grid

    # Misses.
    import csv
    misses = [m for r in results for m in r["misses"]]
    with (out / "misses.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(misses[0])); w.writeheader(); w.writerows(misses)
    fn = [m for m in misses if m["best_final_iou"] <= 0.5]
    cat = {
        "gt_total": len(misses),
        "fn_total": len(fn),
        "fn_no_proposal_overlap(prop_iou<0.1)": sum(m["best_proposal_iou"] < 0.1 for m in fn),
        "fn_poor_proposal(0.1-0.5)": sum(0.1 <= m["best_proposal_iou"] <= 0.5 for m in fn),
        "fn_good_proposal_lost(>0.5)": sum(m["best_proposal_iou"] > 0.5 for m in fn),
        "fn_good_proposal_lost_low_conf(<0.35)": sum(m["best_proposal_iou"] > 0.5 and m["best_proposal_score"] < 0.35 for m in fn),
        "fn_final_near_miss(0.3-0.5)": sum(0.3 <= m["best_final_iou"] <= 0.5 for m in fn),
    }
    summary["miss_breakdown"] = cat

    # Rescoring with grouped 5-fold CV over images.
    import lightgbm as lgb
    sweep = []
    for thr in (0.55, 0.65):
        X = np.concatenate([r["cands"][thr]["X"] for r in results])
        y = np.concatenate([r["cands"][thr]["y"] for r in results])
        grp = np.repeat(np.arange(len(results)), [len(r["cands"][thr]["y"]) for r in results])
        pred = np.zeros_like(y)
        folds = np.arange(len(results)) % 5
        for k in range(5):
            tr = folds[grp] != k
            model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                      min_child_samples=20, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, verbose=-1, random_state=k)
            model.fit(X[tr], y[tr])
            pred[~tr] = model.predict(X[~tr])
        # Exact pooled PQ when dropping instances from the full candidate paste
        # (dropped pixels are not re-claimed by other instances).
        for tau in [None] + [round(float(t), 3) for t in np.arange(0.06, 0.42, 0.02)]:
            tp = fp = fn = 0; si = 0.0; off = 0
            for r in results:
                c = r["cands"][thr]; n = len(c["y"])
                sel = np.ones(n, bool) if tau is None else pred[off:off + n] > tau
                off += n
                for yv, ng in zip(c["y_views"], c["n_gt"]):
                    t = int((yv[sel] > 0).sum())
                    tp += t; fp += int(sel.sum()) - t; fn += ng - t; si += float(yv[sel].sum())
            sweep.append({"refiner_thr": thr, "tau": "all" if tau is None else tau, "tp": tp, "fp": fp,
                          "fn": fn, "sq": si / max(1, tp), "pq": si / (tp + 0.5 * fp + 0.5 * fn)})
        final = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                  min_child_samples=20, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, verbose=-1, random_state=0).fit(X, y)
        final.booster_.save_model(str(out / f"rescorer_thr{thr}.txt"))
        extra = X.shape[1] - len(FEATURES)
        names = FEATURES + ([f"aux{i}" for i in range(extra - len(UNET_FEATURES))] + UNET_FEATURES
                            if a.unet else [f"aux{i}" for i in range(extra)])
        imp = dict(zip(names,
                       final.booster_.feature_importance("gain").round(1).tolist()))
        summary[f"feature_gain_thr{thr}"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
        corr = float(np.corrcoef(pred, y)[0, 1])
        summary[f"rescore_corr_thr{thr}"] = corr
        np.save(out / f"rescore_pred_thr{thr}.npy", pred)
        np.savez_compressed(
            out / f"instances_thr{thr}.npz", X=X, y=y, pred=pred, grp=grp,
            files=np.array([r["file"] for r in results]),
            owner=np.concatenate([np.asarray(r["cands"][thr]["owner"], dtype=np.int32) for r in results]),
            y_views=np.array([r["cands"][thr]["y_views"] for r in results], dtype=object),
            n_gt=np.array([r["cands"][thr]["n_gt"] for r in results], dtype=object))
        taus = [0.22, 0.24, 0.26, 0.28, 0.30, 0.32]
        jobs, off = [], 0
        for r in results:
            n = len(r["cands"][thr]["y"])
            jobs.append((r["file"], thr, r["cands"][thr]["owner"], pred[off:off + n], taus))
            off += n
        if not a.pool:  # the repaste pass reads a single dump, so it is skipped for pooled candidates
            with Pool(a.workers) as pool:
                rep = pool.map(repaste_file, jobs)
            summary[f"repaste_thr{thr}"] = {str(t): pq([v for r in rep for v in r[t]]) for t in taus}
    import csv as _csv
    with (out / "sweep.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(sweep[0])); w.writeheader(); w.writerows(sweep)
    summary["rescore_best"] = sorted(sweep, key=lambda r: -r["pq"])[:6]
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
