"""Variant lab: how much could a better per-instance mask shape be worth?

For every selected out-of-fold instance (v5 recipe) we build alternative masks
and score an ORACLE that picks, per instance, the variant with the highest
summed matched IoU over the annotator views. This is an upper bound for any
learned per-instance choice inside that family.

Families (all variants exclude pixels owned by other selected instances):
  thr      refiner threshold 0.30, 0.45, 0.55, 0.80 instead of 0.65
  morph    base mask dilated by 2 or 4 px, or eroded by 1 px
  lcc      largest connected component of the base mask
  merge    union with an adjacent selected instance (gap <= 8 px); the partner is removed
  all      best of everything above

Scoring is strict one-to-one (a second hit on the same ground truth is a false positive).

Usage: python variant_lab.py --dump=a,b,... --lab=<postproc_lab dir> --tau=0.24 [--avg_suffix=_r34c384]
Writes results/variant_lab/<lab name>/{summary.json,variants.csv}.
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

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_filament.postproc import EIGHT, paste_masks  # noqa: E402
import overlap_lab as O  # noqa: E402
import postproc_lab as L  # noqa: E402

G: dict = {}
PQ0 = 0.458
H = W = 2048
PAD = 48


def hits(mask_win, win, views, gas):
    """Per view: (gt index, IoU) of the IoU > 0.5 match, else (-1, 0)."""
    ya, yb, xa, xb = win
    a = float(mask_win.sum())
    out = []
    for gts, ga in zip(views, gas):
        best, bi = 0.0, -1
        if a > 0:
            for gi, g in enumerate(gts):
                inter = float((g[ya:yb, xa:xb] & mask_win).sum())
                if inter == 0:
                    continue
                iou = inter / (a + ga[gi] - inter)
                if iou > best:
                    best, bi = iou, gi
        out.append((bi, best) if best > 0.5 else (-1, 0.0))
    return out


def process(f: str):
    rows = G["props"][f]
    probs, _ = O.load_probs(f)
    lab_all, owner = paste_masks(rows, probs, (H, W), 0.65, 96)
    p = G["pred"][f]
    sel = np.nonzero(p > G["tau"])[0]
    remap = np.zeros(len(owner) + 1, np.int32)
    remap[sel + 1] = np.arange(1, len(sel) + 1)
    labels = remap[lab_all]
    views = L.gt_masks_for(G["views"][f], G["anns"])
    gas = [[float(g.sum()) for g in gts] for gts in views]
    n = len(sel)
    inst, diag = [], []
    gray = cv2.imread(str(L.DATA / "train_images" / f), cv2.IMREAD_GRAYSCALE)
    objs = ndimage.find_objects(labels, max_label=n)
    for i in range(n):
        idx = owner[sel[i]]
        x0, y0, side = rows[idx]["crop"]
        sl = objs[i]
        ya = max(0, min(y0, sl[0].start) - PAD); yb = min(H, max(y0 + side, sl[0].stop) + PAD)
        xa = max(0, min(x0, sl[1].start) - PAD); xb = min(W, max(x0 + side, sl[1].stop) + PAD)
        win = (ya, yb, xa, xb)
        lw = labels[ya:yb, xa:xb]
        base = lw == i + 1
        free = (lw == 0) | base
        prob = np.zeros(base.shape, np.float32)
        pr = cv2.resize(probs[idx].astype(np.float32) / 255.0, (side, side), interpolation=cv2.INTER_LINEAR)
        cya, cyb, cxa, cxb = max(ya, y0), min(yb, y0 + side), max(xa, x0), min(xb, x0 + side)
        prob[cya - ya:cyb - ya, cxa - xa:cxb - xa] = pr[cya - y0:cyb - y0, cxa - x0:cxb - x0]
        var = {"base": base}
        for t in (0.30, 0.45, 0.55, 0.80):
            m = (prob >= t) & free
            if t < 0.65:
                # keep only the part connected to the base mask
                lab, _ = ndimage.label(m | base, structure=EIGHT)
                ids = np.unique(lab[base]); ids = ids[ids > 0]
                m = np.isin(lab, ids)
            var[f"thr{t:.2f}"] = m
        var["dil2"] = ndimage.binary_dilation(base, structure=EIGHT, iterations=2) & free
        var["dil4"] = ndimage.binary_dilation(base, structure=EIGHT, iterations=4) & free
        var["ero1"] = ndimage.binary_erosion(base, structure=EIGHT, iterations=1)
        lab, k = ndimage.label(base, structure=EIGHT)
        var["lcc"] = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if k > 1 else base
        near = ndimage.binary_dilation(base, structure=EIGHT, iterations=8)
        # Classical image-based variants: grow into, or trim to, the dark filament body.
        gwin = gray[ya:yb, xa:xb]
        gf = gwin.astype(np.float32)
        ringb = near & ~base & (lw == 0)
        mu_in = float(gf[base].mean())
        bg = float(np.median(gf[ringb])) if ringb.any() else mu_in + 1.0
        for kfrac, r_px in ((0.5, 16), (0.3, 16), (0.5, 40)):
            dark = gf <= bg - kfrac * (bg - mu_in)
            reach = near if r_px == 8 else ndimage.binary_dilation(base, structure=EIGHT, iterations=r_px)
            var[f"grow{int(kfrac * 100)}r{r_px}"] = ndimage.binary_propagation(
                base, structure=EIGHT, mask=(dark | base) & free & reach)
        tr = base & (gf <= bg - 0.3 * (bg - mu_in))
        if tr.sum() >= 48:
            labt, kt = ndimage.label(tr, structure=EIGHT)
            tr = labt == (np.argmax(np.bincount(labt.ravel())[1:]) + 1) if kt > 1 else tr
        var["trim30"] = tr if tr.sum() >= 48 else base
        bh = cv2.morphologyEx(gwin, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        bh_in = float(bh[base].mean())
        reach24 = ndimage.binary_dilation(base, structure=EIGHT, iterations=24)
        var["bhgrow50"] = ndimage.binary_propagation(
            base, structure=EIGHT, mask=((bh >= 0.5 * bh_in) | base) & free & reach24)
        partners = [int(j) for j in np.unique(lw[near]) if j not in (0, i + 1)]
        for j in partners:
            var[f"merge{j - 1}"] = base | (lw == j)
        # Extend with adjacent candidates that the selector dropped.
        la = lab_all[ya:yb, xa:xb]
        dropped = (la > 0) & (lw == 0)
        for r_px in (8, 24):
            zone = near if r_px == 8 else ndimage.binary_dilation(base, structure=EIGHT, iterations=r_px)
            ids = [int(j) for j in np.unique(la[zone & dropped]) if j > 0]
            if not ids:
                continue
            for j in ids[:4]:
                var[f"ext{r_px}_{j}"] = base | (la == j)
            var[f"extall{r_px}"] = base | np.isin(la, ids)
        gw = gray[ya:yb, xa:xb].astype(np.float32)
        barea = max(1.0, float(base.sum()))
        d = {}
        for name, m in var.items():
            area = float(m.sum())
            added, removed = m & ~base, base & ~m
            ring = ndimage.binary_dilation(m, structure=EIGHT, iterations=4) & ~m
            pin = prob[m]
            feat = [np.log1p(area), np.log((area + 1.0) / barea),
                    float(pin.mean()) if area else 0.0, float(np.percentile(pin, 10)) if area else 0.0,
                    float(prob[added].mean()) if added.any() else 0.0, float(added.sum()) / barea,
                    float(prob[removed].mean()) if removed.any() else 0.0, float(removed.sum()) / barea,
                    float(gw[m].mean()) if area else 0.0, float(gw[ring].mean()) if ring.any() else 0.0,
                    float(prob[ring].mean()) if ring.any() else 0.0,
                    float(ndimage.label(m, structure=EIGHT)[1])]
            d[name] = {"area": int(area), "hits": hits(m, win, views, gas), "feat": feat}
            if name.startswith("ext") and "_" in name:
                j = int(name.split("_")[1]) - 1
                d[name]["pfeat"] = [*G["X"][f][j].tolist(), float(p[j])]
        inst.append(d)
        # Diagnostic for too-small masks: where is the rest of the ground truth?
        bx0, by0, bx1, by1 = [int(round(v)) for v in rows[idx]["bbox"]]
        for vi, gts in enumerate(views):
            for gi, g in enumerate(gts):
                gwn = g[ya:yb, xa:xb]
                inter = float((gwn & base).sum())
                if inter == 0:
                    continue
                ga = gas[vi][gi]
                iou = inter / (barea + ga - inter)
                if not (0.1 <= iou <= 0.5 and inter / barea >= 0.7):
                    continue
                ys, xs = np.nonzero(g)
                in_crop = float(((ys >= y0) & (ys < y0 + side) & (xs >= x0) & (xs < x0 + side)).mean())
                in_box = float(((ys >= by0) & (ys < by1) & (xs >= bx0) & (xs < bx1)).mean())
                rest = g & (labels != i + 1)
                rest_n = max(1.0, float(rest.sum()))
                diag.append({"file": f, "view": vi, "iou": round(iou, 4), "gt_area": int(ga), "pred_area": int(barea),
                             "gt_in_crop": round(in_crop, 4), "gt_in_box": round(in_box, 4),
                             "rest_cov_selected": round(float((labels[rest] > 0).sum()) / rest_n, 4),
                             "rest_cov_any_candidate": round(float((lab_all[rest] > 0).sum()) / rest_n, 4),
                             "rest_mean_prob_in_window": round(float(prob[gwn & ~base].mean()) if (gwn & ~base).any() else 0.0, 4)})
    return {"file": f, "n_gt": [len(g) for g in views], "inst": inst, "diag": diag, "p": p[sel],
            "X": G["X"][f][sel] if n else np.zeros((0, G["X"][f].shape[1]), np.float32)}


def value(h):
    return sum(i for _, i in h)


def choose(inst, family):
    """Per image: chosen variant name per instance (None = removed by a merge)."""
    n = len(inst)
    chosen = ["base"] * n
    fam = {"thr": ("thr",), "morph": ("dil", "ero"), "lcc": ("lcc",), "merge": (), "ext": ("ext",),
           "cv": ("grow", "trim", "bhgrow"),
           "all": ("thr", "dil", "ero", "lcc", "ext", "grow", "trim", "bhgrow"), "base": ()}[family]
    for i in range(n):
        best = value(inst[i]["base"]["hits"])
        for name, v in inst[i].items():
            if name.startswith(fam) and fam and value(v["hits"]) > best + 1e-9:
                best, chosen[i] = value(v["hits"]), name
    if family in ("merge", "all"):
        nv = len(inst[0]["base"]["hits"]) if n else 0
        cands = []
        for i in range(n):
            for name, v in inst[i].items():
                if name.startswith("merge"):
                    j = int(name[5:])
                    gain = value(v["hits"]) - value(inst[i][chosen[i]]["hits"]) - value(inst[j][chosen[j]]["hits"]) \
                        + 0.5 * PQ0 * nv
                    cands.append((gain, i, j, name))
        used = set()
        for gain, i, j, name in sorted(cands, reverse=True):
            if gain <= 0 or i in used or j in used:
                continue
            used.update((i, j)); chosen[i] = name; chosen[j] = None
    return chosen


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
    Xall = {f: t["X"][offs[i]:offs[i + 1]] for i, f in enumerate(files)}
    views, anns = L.load_gt(files)
    O.G.update(file_dump=file_dump, avg_suffix=a.avg_suffix)
    G.update(props=props, pred=pred, tau=a.tau, views=views, anns=anns, X=Xall)
    with Pool(a.workers) as pool:
        res = pool.map(process, files)
    out = L.RESULTS / "variant_lab" / Path(a.lab).name
    out.mkdir(parents=True, exist_ok=True)
    summary = {"lab": a.lab, "tau": a.tau, "images": len(files)}
    rows_out = []
    for family in ("base", "thr", "morph", "lcc", "merge", "ext", "cv", "all"):
        tp = fp = fn = 0; s = 0.0; picks = Counter()
        for r in res:
            ch = choose(r["inst"], family)
            for c in ch:
                picks["removed" if c is None else ("merge" if c.startswith("merge") else c.split("_")[0])] += 1
            for v, ng in enumerate(r["n_gt"]):
                best = {}
                kept = 0
                for i, c in enumerate(ch):
                    if c is None:
                        continue
                    kept += 1
                    gi, iou = r["inst"][i][c]["hits"][v]
                    if gi >= 0:
                        best[gi] = max(best.get(gi, 0.0), iou)
                tp += len(best); fp += kept - len(best); fn += ng - len(best); s += sum(best.values())
        summary[family] = {"pq": s / (tp + 0.5 * fp + 0.5 * fn), "tp": tp, "fp": fp, "fn": fn, "sq": s / max(1, tp),
                           "picks": dict(picks.most_common())}
    # Learned chooser: predict value(variant) - value(base), grouped 5-fold over images.
    import lightgbm as lgb
    KINDS = ["thr0.30", "thr0.45", "thr0.55", "thr0.80", "dil2", "dil4", "ero1", "lcc",
             "grow50r16", "grow30r16", "grow50r40", "trim30", "bhgrow50"]
    # Blanket: apply one fixed variant to every instance (no choice needed).
    blanket = {}
    for kname in KINDS:
        tp = fp = fn = 0; sb = 0.0; ratio = []
        for r in res:
            ratio += [d[kname]["area"] / max(1, d["base"]["area"]) for d in r["inst"]]
            for v, ng in enumerate(r["n_gt"]):
                best = {}
                for d in r["inst"]:
                    gi, iou = d[kname]["hits"][v]
                    if gi >= 0:
                        best[gi] = max(best.get(gi, 0.0), iou)
                tp += len(best); fp += len(r["inst"]) - len(best); fn += ng - len(best); sb += sum(best.values())
        blanket[kname] = {"pq": sb / (tp + 0.5 * fp + 0.5 * fn), "tp": tp, "fp": fp, "fn": fn,
                          "median_area_ratio": float(np.median(ratio))}
    summary["blanket"] = blanket
    Xs, ys, key = [], [], []
    for ri, r in enumerate(res):
        for i, d in enumerate(r["inst"]):
            b = value(d["base"]["hits"])
            for kname in KINDS:
                if d[kname]["area"] == d["base"]["area"]:
                    continue
                onehot = [float(kname == q) for q in KINDS]
                Xs.append(np.concatenate([r["X"][i], d["base"]["feat"], d[kname]["feat"], onehot,
                                          [len(r["n_gt"])]]))
                ys.append((value(d[kname]["hits"]) - b) / max(1, len(r["n_gt"])))
                key.append((ri, i, kname))
    Xs = np.asarray(Xs, np.float32); ys = np.asarray(ys, np.float32)
    Xs = Xs[:, :-1]  # the number of annotator views is unknown at test time
    fold = np.array([k[0] % 5 for k in key])
    gain = np.zeros(len(ys), np.float32)
    for kf in range(5):
        tr = fold != kf
        mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=kf)
        gain[~tr] = mdl.fit(Xs[tr], ys[tr]).predict(Xs[~tr])
    lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0).fit(Xs, ys) \
        .booster_.save_model(str(out / "variant_chooser.txt"))
    best_gain = {}
    for (ri, i, kname), gpred in zip(key, gain):
        if gpred > best_gain.get((ri, i), (0.0, "base"))[0]:
            best_gain[(ri, i)] = (float(gpred), kname)
    learned = {}
    for margin in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        tp = fp = fn = 0; s = 0.0; picks = Counter()
        for ri, r in enumerate(res):
            ch = []
            for i in range(len(r["inst"])):
                gpred, kname = best_gain.get((ri, i), (0.0, "base"))
                ch.append(kname if gpred > margin else "base")
            picks.update(ch)
            for v, ng in enumerate(r["n_gt"]):
                best = {}
                for i, c in enumerate(ch):
                    gi, iou = r["inst"][i][c]["hits"][v]
                    if gi >= 0:
                        best[gi] = max(best.get(gi, 0.0), iou)
                tp += len(best); fp += len(ch) - len(best); fn += ng - len(best); s += sum(best.values())
        learned[str(margin)] = {"pq": s / (tp + 0.5 * fp + 0.5 * fn), "tp": tp, "fp": fp, "fn": fn,
                                "sq": s / max(1, tp), "picks": dict(picks.most_common())}
    summary["learned_chooser"] = learned
    # Learned extension: attach one dropped neighbouring candidate to a selected instance.
    Xe, ye, ke = [], [], []
    for ri, r in enumerate(res):
        for i, d in enumerate(r["inst"]):
            b = value(d["base"]["hits"])
            for name, v in d.items():
                if "pfeat" not in v:
                    continue
                Xe.append(np.concatenate([r["X"][i], [r["p"][i]], v["pfeat"], d["base"]["feat"], v["feat"],
                                          [float(name.startswith("ext24"))]]))
                ye.append((value(v["hits"]) - b) / max(1, len(r["n_gt"])))
                ke.append((ri, i, name))
    Xe = np.asarray(Xe, np.float32); ye = np.asarray(ye, np.float32)
    folde = np.array([k[0] % 5 for k in ke])
    ge = np.zeros(len(ye), np.float32)
    for kf in range(5):
        tr = folde != kf
        ge[~tr] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                                    random_state=kf).fit(Xe[tr], ye[tr]).predict(Xe[~tr])
    beste = {}
    for (ri, i, name), gpred in zip(ke, ge):
        if gpred > beste.get((ri, i), (-1.0, "base"))[0]:
            beste[(ri, i)] = (float(gpred), name)
    ext_learned = {}
    for margin in (-0.02, 0.0, 0.01, 0.02, 0.04, 0.06, 0.1):
        tp = fp = fn = 0; sm = 0.0; used = 0
        for ri, r in enumerate(res):
            ch = []
            for i in range(len(r["inst"])):
                gpred, name = beste.get((ri, i), (-1.0, "base"))
                ch.append(name if gpred > margin else "base")
            used += sum(c != "base" for c in ch)
            for v, ng in enumerate(r["n_gt"]):
                best = {}
                for i, c in enumerate(ch):
                    gi, iou = r["inst"][i][c]["hits"][v]
                    if gi >= 0:
                        best[gi] = max(best.get(gi, 0.0), iou)
                tp += len(best); fp += len(ch) - len(best); fn += ng - len(best); sm += sum(best.values())
        ext_learned[str(margin)] = {"pq": sm / (tp + 0.5 * fp + 0.5 * fn), "tp": tp, "fp": fp, "fn": fn,
                                    "extended": used}
    summary["learned_extension"] = ext_learned
    summary["extension_pairs"] = int(len(ye))
    summary["extension_pairs_positive"] = int((ye > 0).sum())
    summary["extension_pairs_negative"] = int((ye < 0).sum())
    summary["extension_corr"] = float(np.corrcoef(ge, ye)[0, 1])
    summary["chooser_corr"] = float(np.corrcoef(gain, ys)[0, 1])
    diag = [x for r in res for x in r["diag"]]
    if diag:
        with (out / "small_mask_diag.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(diag[0])); w.writeheader(); w.writerows(diag)
        med = lambda k: float(np.median([x[k] for x in diag]))  # noqa: E731
        summary["small_mask_diag"] = {
            "cases": len(diag), "median_iou": med("iou"), "median_gt_over_pred_area": float(np.median(
                [x["gt_area"] / max(1, x["pred_area"]) for x in diag])),
            "median_gt_in_crop": med("gt_in_crop"), "median_gt_in_box": med("gt_in_box"),
            "share_gt_in_box<0.7": float(np.mean([x["gt_in_box"] < 0.7 for x in diag])),
            "share_gt_in_crop<0.9": float(np.mean([x["gt_in_crop"] < 0.9 for x in diag])),
            "median_rest_cov_selected": med("rest_cov_selected"),
            "median_rest_cov_any_candidate": med("rest_cov_any_candidate"),
            "share_rest_cov_selected>0.5": float(np.mean([x["rest_cov_selected"] > 0.5 for x in diag])),
            "share_rest_cov_any_candidate>0.5": float(np.mean([x["rest_cov_any_candidate"] > 0.5 for x in diag])),
            "median_rest_mean_prob": med("rest_mean_prob_in_window"),
            "share_rest_mean_prob>0.3": float(np.mean([x["rest_mean_prob_in_window"] > 0.3 for x in diag])),
        }
    for r in res:
        for i, d in enumerate(r["inst"]):
            for name, v in d.items():
                rows_out.append({"file": r["file"], "inst": i, "variant": name, "area": v["area"],
                                 "value": round(value(v["hits"]), 4), "n_hits": sum(g >= 0 for g, _ in v["hits"])})
    with (out / "variants.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
