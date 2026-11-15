"""Error analysis of the selected out-of-fold predictions (v5 recipe).

Every false positive and false negative is counted per annotator copy (view),
exactly like the scorer, and put in one category.

False positive (a prediction with no IoU > 0.5 ground truth in this view):
  fragment        localized (best IoU >= 0.1), >= 70% of the prediction lies inside one ground truth
  oversized       localized, the prediction covers >= 70% of the ground truth but is much larger
  boundary        localized, neither of the above (shift or thickness)
  other_annotator not localized here, but a true positive in another view of the image
  false_alarm     not localized in this view and never a true positive in any view

False negative (a ground truth with no IoU > 0.5 prediction in this view):
  split_by_us     localized, >= 70% covered by the union of >= 2 predictions
  merged_by_us    localized, >= 70% covered by one prediction that is much larger
  partial         localized, best prediction lies inside it but covers too little
  boundary        localized, none of the above
  dropped         not localized, but a candidate with IoU > 0.5 existed and the selector dropped it
  poor_candidate  not localized, best candidate IoU in 0.1 to 0.5
  no_candidate    not localized, no candidate

Each false negative also records whether another annotator drew the same filament.

Usage: python error_lab.py --dump=a,b,... --lab=<postproc_lab dir> --tau=0.24 [--avg_suffix=_r34c384]
Writes results/error_lab/<lab name>/{summary.json,pred_rows.csv,gt_rows.csv}.
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
import overlap_lab as O  # noqa: E402
import postproc_lab as L  # noqa: E402

G: dict = {}


def process(f: str):
    rows = G["props"][f]
    probs, _ = O.load_probs(f)
    lab_all, owner = paste_masks(rows, probs, (2048, 2048), 0.65, 96)
    assert list(owner) == list(G["owner"][f]), f
    p = G["pred"][f]
    N = len(owner)
    sel = np.nonzero(p > G["tau"])[0]
    pa = np.bincount(lab_all.ravel(), minlength=N + 1)[1:].astype(np.float64)
    views = L.gt_masks_for(G["views"][f], G["anns"])
    nv = len(views)
    # Per view: intersection of every ground truth with every candidate instance.
    inter, ga, gtmap = [], [], []
    for gts in views:
        m = np.zeros((len(gts), N))
        a = np.zeros(len(gts))
        gm = np.zeros(lab_all.shape, np.uint16)
        for gi, g in enumerate(gts):
            m[gi] = np.bincount(lab_all[g], minlength=N + 1)[1:]
            a[gi] = g.sum()
            gm[g] = gi + 1
        inter.append(m); ga.append(a); gtmap.append(gm)
    iou = [m / (pa[None, :] + a[:, None] - m + 1e-9) for m, a in zip(inter, ga)]
    tp_views = np.zeros(N, int)
    for v in range(nv):
        if iou[v].size:
            tp_views += (iou[v] > 0.5).any(axis=0)
    pred_rows, gt_rows = [], []
    for v in range(nv):
        I, M, A = iou[v][:, sel], inter[v][:, sel], ga[v]
        ng = len(A)
        for jj, k in enumerate(sel):
            base = dict(file=f, view=v, n_views=nv, area=int(pa[k]), det_score=round(rows[owner[k]]["score"], 4),
                        sel_pred=round(float(p[k]), 4), tp_views=int(tp_views[k]))
            if ng == 0:
                pred_rows.append(dict(base, status="FP", best_iou=0.0, cat="other_annotator" if tp_views[k] else "false_alarm"))
                continue
            bi = int(I[:, jj].argmax()); b = float(I[bi, jj])
            if b > 0.5:
                pred_rows.append(dict(base, status="TP", best_iou=round(b, 4), cat="tp"))
                continue
            in_gt = M[bi, jj] / pa[k]; cov = M[bi, jj] / A[bi]
            if b >= 0.1:
                cat = "fragment" if in_gt >= 0.7 else "oversized" if cov >= 0.7 else "boundary"
            else:
                cat = "other_annotator" if tp_views[k] else "false_alarm"
            pred_rows.append(dict(base, status="FP", best_iou=round(b, 4), cat=cat))
        for gi in range(ng):
            # Agreement with the other annotators: best IoU with one of their filaments.
            g = views[v][gi]
            agree = -1.0
            for u in range(nv):
                if u == v or len(ga[u]) == 0:
                    agree = max(agree, 0.0) if u != v else agree
                    continue
                c = np.bincount(gtmap[u][g], minlength=len(ga[u]) + 1)[1:].astype(np.float64)
                agree = max(agree, float((c / (ga[u] + A[gi] - c + 1e-9)).max()))
            base = dict(file=f, view=v, n_views=nv, area=int(A[gi]), other_annot_iou=round(agree, 4))
            b_all = float(iou[v][gi].max()) if N else 0.0
            k_all = int(iou[v][gi].argmax()) if N else -1
            base["best_cand_iou"] = round(b_all, 4)
            base["best_cand_sel_pred"] = round(float(p[k_all]), 4) if N else 0.0
            if len(sel) == 0:
                b, bj = 0.0, -1
            else:
                bj = int(I[gi].argmax()); b = float(I[gi, bj])
            if b > 0.5:
                gt_rows.append(dict(base, status="TP", best_iou=round(b, 4), cat="tp"))
                continue
            if b >= 0.1:
                cov_union = float(M[gi].sum() / A[gi])
                n_inside = int(((M[gi] / pa[sel]) >= 0.5).sum())
                cov = M[gi, bj] / A[gi]; in_gt = M[gi, bj] / pa[sel[bj]]
                if cov_union >= 0.7 and n_inside >= 2:
                    cat = "split_by_us"
                elif cov >= 0.7:
                    cat = "merged_by_us"
                elif in_gt >= 0.7:
                    cat = "partial"
                else:
                    cat = "boundary"
            else:
                cat = "dropped" if b_all > 0.5 else "poor_candidate" if b_all >= 0.1 else "no_candidate"
            gt_rows.append(dict(base, status="FN", best_iou=round(b, 4), cat=cat))
    return pred_rows, gt_rows


def pq(tp, fp, fn, s):
    return s / max(1e-9, tp + 0.5 * fp + 0.5 * fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--tau", type=float, default=0.24)
    ap.add_argument("--avg_suffix", default="")
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    props, file_dump = {}, {}
    for d in a.dump.split(","):
        pr = json.loads((Path(d) / "proposals.json").read_text())["predictions"]
        props.update(pr)
        file_dump.update({f: Path(d) for f in pr})
    t = np.load(Path(a.lab) / "instances_thr0.65.npz", allow_pickle=True)
    files = list(t["files"])
    offs = np.concatenate([[0], np.cumsum(np.bincount(t["grp"], minlength=len(files)))])
    pred = {f: t["pred"][offs[i]:offs[i + 1]] for i, f in enumerate(files)}
    owner = {f: t["owner"][offs[i]:offs[i + 1]] for i, f in enumerate(files)}
    views, anns = L.load_gt(files)
    O.G.update(file_dump=file_dump, avg_suffix=a.avg_suffix)
    G.update(props=props, pred=pred, owner=owner, tau=a.tau, views=views, anns=anns)
    with Pool(a.workers) as pool:
        res = pool.map(process, files)
    P = [r for x in res for r in x[0]]
    Q = [r for x in res for r in x[1]]
    out = L.RESULTS / "error_lab" / Path(a.lab).name
    out.mkdir(parents=True, exist_ok=True)
    for name, rows_ in (("pred_rows.csv", P), ("gt_rows.csv", Q)):
        keys = sorted({k for r in rows_ for k in r})
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows_)
    tp = sum(r["status"] == "TP" for r in P); fp = len(P) - tp
    fn = sum(r["status"] == "FN" for r in Q)
    s = sum(r["best_iou"] for r in P if r["status"] == "TP")
    fpc = Counter(r["cat"] for r in P if r["status"] == "FP")
    fnc = Counter(r["cat"] for r in Q if r["status"] == "FN")
    multi = lambda r: r["n_views"] > 1  # noqa: E731
    fn_rows = [r for r in Q if r["status"] == "FN"]
    tp_rows = [r for r in Q if r["status"] == "TP"]

    def agree_split(rows_):
        m = [r for r in rows_ if multi(r)]
        return {"single_view_images": len(rows_) - len(m),
                "multi_view_drawn_by_other(iou>0.5)": sum(r["other_annot_iou"] > 0.5 for r in m),
                "multi_view_partly_drawn(0.1-0.5)": sum(0.1 <= r["other_annot_iou"] <= 0.5 for r in m),
                "multi_view_only_this_annotator(<0.1)": sum(r["other_annot_iou"] < 0.1 for r in m)}

    bins = [0, 300, 1000, 3000, 10000, 10 ** 9]
    recall_by_area = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = [r for r in Q if lo <= r["area"] < hi]
        recall_by_area[f"{lo}-{hi}"] = {"gt": len(sub), "recall": sum(r["status"] == "TP" for r in sub) / max(1, len(sub))}
    prec_by_area = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = [r for r in P if lo <= r["area"] < hi]
        prec_by_area[f"{lo}-{hi}"] = {"pred": len(sub), "precision": sum(r["status"] == "TP" for r in sub) / max(1, len(sub))}
    by_views = {}
    for nvv in sorted({r["n_views"] for r in Q}):
        tpv = sum(r["status"] == "TP" for r in P if r["n_views"] == nvv)
        fpv = sum(r["status"] == "FP" for r in P if r["n_views"] == nvv)
        fnv = sum(r["status"] == "FN" for r in Q if r["n_views"] == nvv)
        sv = sum(r["best_iou"] for r in P if r["status"] == "TP" and r["n_views"] == nvv)
        by_views[str(nvv)] = {"tp": tpv, "fp": fpv, "fn": fnv, "pq": pq(tpv, fpv, fnv, sv)}
    near_fn = sum(r["best_iou"] >= 0.3 for r in fn_rows)
    near_fp = sum(r["status"] == "FP" and r["best_iou"] >= 0.3 for r in P)
    k = min(near_fn, near_fp)
    local_fn = sum(fnc[c] for c in ("split_by_us", "merged_by_us", "partial", "boundary"))
    local_fp = sum(fpc[c] for c in ("fragment", "oversized", "boundary"))
    k2 = min(local_fn, local_fp)
    summary = {
        "lab": a.lab, "tau": a.tau, "images": len(files), "tp": tp, "fp": fp, "fn": fn,
        "sq": s / max(1, tp), "pq": pq(tp, fp, fn, s),
        "fp_categories": dict(fpc.most_common()), "fn_categories": dict(fnc.most_common()),
        "fn_annotator_agreement": agree_split(fn_rows), "tp_annotator_agreement": agree_split(tp_rows),
        "fn_dropped_annotator_agreement": agree_split([r for r in fn_rows if r["cat"] == "dropped"]),
        "fp_false_alarm_by_views": dict(Counter(r["n_views"] for r in P if r["cat"] == "false_alarm")),
        "recall_by_gt_area": recall_by_area, "precision_by_pred_area": prec_by_area, "by_n_views": by_views,
        "what_if": {
            "remove_all_false_alarms": pq(tp, fp - fpc["false_alarm"], fn, s),
            "fix_near_misses_iou>=0.3_at_iou0.6": pq(tp + k, fp - k, fn - k, s + 0.6 * k),
            "fix_all_localized_errors_at_iou0.6": pq(tp + k2, fp - k2, fn - k2, s + 0.6 * k2),
            "raise_sq_by_0.03": pq(tp, fp, fn, s + 0.03 * tp),
        },
        "seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
