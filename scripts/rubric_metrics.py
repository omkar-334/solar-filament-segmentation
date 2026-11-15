"""Rubric metrics on the out-of-fold predictions (the organisers' 70% quantitative part).

For every (image, annotator view):
  semantic Dice and IoU   union of predicted instances against union of labelled filaments
  matched pairs           IoU and Dice of every IoU > 0.5 pair (the PQ true positives)
  relations               one-to-many: a labelled filament that holds >= 2 predictions
                          (more than half of each prediction lies inside it), i.e. fragmentation
                          many-to-one: a prediction that holds >= 2 labelled filaments
                          (more than half of each label lies inside it), i.e. merging

Reported for the selected keep threshold in full (per-view CSV, pair CSV, histograms)
and as a sweep over keep thresholds, to show the PQ against Dice trade-off.

Usage: python rubric_metrics.py --dump=a,b,... --lab=<postproc_lab dir> --tau=0.24 [--avg_suffix=_r34c384]
Writes results/rubric_metrics/<lab name>/{summary.json,per_view.csv,pairs.csv,sweep.csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_filament.postproc import paste_masks  # noqa: E402
from solar_filament import stage2  # noqa: E402
import postproc_lab as L  # noqa: E402

G: dict = {}
SWEEP = [0.06, 0.10, 0.14, 0.18, 0.22, 0.24, 0.26, 0.30, 0.34]


def semantic_dice(pred_px: float, gt_px: float, inter_px: float) -> float:
    """Foreground Dice of two binary masks from pixel counts (1.0 when both are empty)."""
    den = pred_px + gt_px
    return 2 * inter_px / den if den else 1.0


def label_map(f: str, lab_all: np.ndarray, p: np.ndarray, tau: float) -> np.ndarray:
    """Selected instances at ``tau``; with --stage2 their masks are the stage-2 masks."""
    if G["score"] is not None:
        # v11: stage-2 masks pasted for every candidate above the floor, then selection by
        # the mean of the selector value and the SAM mask-quality score.
        if "new" not in G["cache"].setdefault(f, {}):
            cand = p > G["floor"]
            lab1 = np.where(np.concatenate([[False], cand])[lab_all], lab_all, 0).astype(np.uint16)
            items = [(i, M, prob) for i, M, prob in G["s2"].get(f, []) if cand[i]]
            new, _ = stage2.paste_refined(lab1, np.where(cand, p, -1.0), items, G["s2_thr"])
            q = np.full(len(p), -1.0)
            q[cand] = 0.5 * (p[cand] + G["score"][f])
            G["cache"][f].update(new=new, q=q)
        c = G["cache"][f]
        return np.where(np.concatenate([[False], c["q"] > tau])[c["new"]], c["new"], 0).astype(np.uint16)
    keep = np.concatenate([[False], p > tau])
    lab = np.where(keep[lab_all], lab_all, 0).astype(np.uint16)
    if G["s2"] is not None:
        items = [(i, M, prob) for i, M, prob in G["s2"].get(f, []) if p[i] > tau]
        lab, _ = stage2.paste_refined(lab, np.where(p > tau, p, -1.0), items, G["s2_thr"])
    return lab


def process(f: str):
    rows, probs, _ = L.load_candidates(f)
    lab_all, owner = paste_masks(rows, probs, (2048, 2048), G["thr"], 96)
    p = G["pred"][f]
    N = len(owner)
    views = L.gt_masks_for(G["views"][f], G["anns"])
    out = {"file": f, "taus": {}}
    for tau in G["sweep"]:
        lab = label_map(f, lab_all, p, tau)
        pa_all = np.bincount(lab.ravel(), minlength=N + 1)[1:].astype(np.float64)
        sel = pa_all > 0
        P = pa_all[sel]
        per_view, pairs = [], []
        for v, gts in enumerate(views):
            M = np.zeros((len(gts), int(sel.sum()))); A = np.zeros(len(gts))
            gu = np.zeros(lab.shape, bool)
            for gi, g in enumerate(gts):
                M[gi] = np.bincount(lab[g], minlength=N + 1)[1:][sel]
                A[gi] = g.sum(); gu |= g
            pred_px = float(P.sum()); gt_px = float(gu.sum()); hit_px = float((lab[gu] > 0).sum())
            dice = semantic_dice(pred_px, gt_px, hit_px)
            uni = pred_px + gt_px - hit_px
            iou = M / (P[None, :] + A[:, None] - M + 1e-9) if M.size else np.zeros(M.shape)
            hit = iou > 0.5
            tp = int(hit.sum())
            one_to_many = (M / np.maximum(P[None, :], 1) > 0.5).sum(axis=1) if M.size else np.zeros(len(A), int)
            many_to_one = (M / np.maximum(A[:, None], 1) > 0.5).sum(axis=0) if M.size else np.zeros(int(sel.sum()), int)
            per_view.append({"file": f, "view": v, "n_gt": len(A), "n_pred": int(sel.sum()), "tp": tp,
                             "fp": int(sel.sum()) - tp, "fn": len(A) - tp, "sum_iou": float(iou[hit].sum()),
                             "dice": dice, "iou": hit_px / uni if uni else 1.0, "pred_px": pred_px,
                             "gt_px": gt_px, "inter_px": hit_px,
                             "gt_preds_inside": [int(x) for x in one_to_many],
                             "pred_gts_inside": [int(x) for x in many_to_one]})
            if tau == G["tau"]:
                for gi, pj in zip(*np.nonzero(hit)):
                    i_ = float(iou[gi, pj])
                    pairs.append({"file": f, "view": v, "iou": round(i_, 4), "dice": round(2 * i_ / (1 + i_), 4),
                                  "gt_area": int(A[gi]), "pred_area": int(P[pj])})
        out["taus"][tau] = {"per_view": per_view, "pairs": pairs}
    return out


def hist(values, edges):
    c, _ = np.histogram(values, bins=edges)
    return {f"{edges[i]:.2f}-{edges[i + 1]:.2f}": int(c[i]) for i in range(len(c))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--tau", type=float, default=0.24)
    ap.add_argument("--avg_suffix", default="")
    ap.add_argument("--thr", type=float, default=0.65, help="refiner mask threshold (0.55 or 0.65)")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--pool", default="", help="pooled detectors of the lab run ('+' between detectors)")
    ap.add_argument("--stage2", default="", help="oof_probs.npz of stage2_refiner.py: score its masks")
    ap.add_argument("--stage2_thr", type=float, default=0.40)
    ap.add_argument("--tag", default="")
    ap.add_argument("--score", default="", help="oof_score.npy of sam2_refiner.py --score --merge (v11 selection)")
    ap.add_argument("--floor", type=float, default=0.12, help="candidate floor used with --score")
    a = ap.parse_args()
    assert a.tau in SWEEP
    sweep_taus = [t for t in SWEEP if t >= 0.22] if a.stage2 else SWEEP  # stage-2 maps exist for value > 0.20
    t0 = time.time()
    props, file_dump = {}, {}
    for d in a.dump.split(","):
        pr = json.loads((Path(d) / "proposals.json").read_text())["predictions"]
        props.update(pr)
        file_dump.update({f: Path(d) for f in pr})
    t = np.load(Path(a.lab) / f"instances_thr{a.thr}.npz", allow_pickle=True)
    files = list(t["files"])
    offs = np.concatenate([[0], np.cumsum(np.bincount(t["grp"], minlength=len(files)))])
    pred = {f: t["pred"][offs[i]:offs[i + 1]] for i, f in enumerate(files)}
    views, anns = L.load_gt(files)
    pools = []
    for group in [g for g in a.pool.split("+") if g]:
        pd_, pp_ = {}, {}
        for d in group.split(","):
            pp = json.loads((Path(d) / "proposals.json").read_text())["predictions"]
            pp_.update(pp); pd_.update({f: Path(d) for f in pp})
        pools.append((pd_, pp_))
    L.G.update(props=props, file_dump=file_dump, avg_suffix=a.avg_suffix, pools=pools)
    s2 = None
    if a.stage2:
        z = np.load(a.stage2, allow_pickle=True)
        s2 = {}
        for f, i, M, pr in zip(z["files"], z["idx"], z["M"], z["probs"]):
            s2.setdefault(str(f), []).append((int(i), M, pr))
    score = None
    if a.score:
        sc = np.load(a.score).astype(np.float64)
        n_c = [int((pred[f] > a.floor).sum()) for f in files]
        assert sum(n_c) == len(sc), (sum(n_c), len(sc))
        o = np.concatenate([[0], np.cumsum(n_c)])
        score = {f: sc[o[i]:o[i + 1]] for i, f in enumerate(files)}
    G.update(props=props, pred=pred, tau=a.tau, views=views, anns=anns, thr=a.thr, sweep=sweep_taus, s2=s2,
             s2_thr=a.stage2_thr, score=score, floor=a.floor, cache={})
    with Pool(a.workers) as pool:
        res = pool.map(process, files)
    out = L.RESULTS / "rubric_metrics" / (Path(a.lab).name + (f"_thr{a.thr}" if a.thr != 0.65 else "") + a.tag)
    out.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0, 1, 11)
    sweep, summary = [], {"lab": a.lab, "images": len(files), "selected_tau": a.tau, "refiner_thr": a.thr}
    for tau in sweep_taus:
        pv = [r for x in res for r in x["taus"][tau]["per_view"]]
        tp = sum(r["tp"] for r in pv); fp = sum(r["fp"] for r in pv); fn = sum(r["fn"] for r in pv)
        s = sum(r["sum_iou"] for r in pv)
        dice = np.array([r["dice"] for r in pv]); iou = np.array([r["iou"] for r in pv])
        ipx = sum(r["inter_px"] for r in pv); ppx = sum(r["pred_px"] for r in pv); gpx = sum(r["gt_px"] for r in pv)
        g_in = np.array([c for r in pv for c in r["gt_preds_inside"]])
        p_in = np.array([c for r in pv for c in r["pred_gts_inside"]])
        row = {"tau": tau, "pq": s / (tp + 0.5 * fp + 0.5 * fn), "sq": s / max(1, tp),
               "rq": tp / (tp + 0.5 * fp + 0.5 * fn), "tp": tp, "fp": fp, "fn": fn,
               "mean_dice_per_view": float(dice.mean()), "median_dice_per_view": float(np.median(dice)),
               "global_dice": 2 * ipx / (ppx + gpx), "mean_iou_per_view": float(iou.mean()),
               "pixel_precision": ipx / max(1.0, ppx), "pixel_recall": ipx / max(1.0, gpx),
               "one_to_many_share_of_gt": float((g_in >= 2).mean()), "many_to_one_share_of_pred": float((p_in >= 2).mean()),
               "n_pred_per_view": float(np.mean([r["n_pred"] for r in pv]))}
        sweep.append(row)
        if tau == a.tau:
            pairs = [r for x in res for r in x["taus"][tau]["pairs"]]
            summary["selected"] = dict(
                row, views=len(pv),
                dice_per_view_hist=hist(dice, edges), iou_per_view_hist=hist(iou, edges),
                dice_per_view_quantiles={q: float(np.quantile(dice, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                matched_pair_iou_hist=hist([r["iou"] for r in pairs], np.linspace(0.5, 1, 11)),
                matched_pair_dice_hist=hist([r["dice"] for r in pairs], np.linspace(0.6, 1, 9)),
                matched_pair_iou_mean=float(np.mean([r["iou"] for r in pairs])),
                matched_pair_dice_mean=float(np.mean([r["dice"] for r in pairs])),
                gt_predictions_inside_counts=dict(sorted(Counter(np.minimum(g_in, 4).tolist()).items())),
                pred_labels_inside_counts=dict(sorted(Counter(np.minimum(p_in, 4).tolist()).items())))
            with (out / "per_view.csv").open("w", newline="") as fh:
                keys = [k for k in pv[0] if not k.endswith("_inside")]
                w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(pv)
            with (out / "pairs.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    with (out / "sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep[0])); w.writeheader(); w.writerows(sweep)
    summary["sweep"] = sweep
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
