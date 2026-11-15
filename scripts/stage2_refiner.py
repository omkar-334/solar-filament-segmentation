"""Stage 2: rotation-normalised, context-conditioned re-segmentation of selected instances.

Every pasted candidate of the stage-1 pipeline (detector pool, crop refiners, pixel
claiming) is cut out along its own long axis into a 384 x 128 crop with five channels:
gray, CLAHE, dense U-Net probability, its own stage-1 mask (the seed) and the masks of
the other selected instances. A ResNet34 U-Net predicts the probability that an annotator
who drew this filament marked each pixel (target: mean of the best-overlapping label of
every annotator view that has one with IoU > 0.2). Models are trained in the same grouped
5 folds as the detectors, so every prediction below is out of fold.

Evaluation replaces the selected stage-1 masks by the thresholded stage-2 masks (pixel
claiming in selector order) and reports the exact pooled PQ for several thresholds and
paste rules against the unchanged stage-1 masks.

Usage (GPU): python stage2_refiner.py --dump=a,b,... --lab=<postproc_lab dir> [--avg_suffix=_r34c384]
             [--pool=c,d,...] [--unet=/data/runs/unet_prob_oof] [--tau=0.24] [--tag=v6]
Writes results/stage2_refiner/<tag>/{summary.json,sweep.csv,instances.csv,train_log.csv}
and /data/runs/stage2_<tag>_fold{k}.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_filament.postproc import paste_masks  # noqa: E402
import postproc_lab as L  # noqa: E402

from solar_filament.stage2 import H, W, build_model, rect_affine, warp, window_prob  # noqa: E402

THRS = [0.30, 0.40, 0.50, 0.60]
MODES = ["all", "seeded", "guarded"]
TAUS = [0.20, 0.24, 0.28]
S: dict = {}


def stage1(f: str):
    rows, probs, _ = L.load_candidates(f)
    gray = cv2.imread(str(L.DATA / "train_images" / f), cv2.IMREAD_GRAYSCALE)
    lab, owner = paste_masks(rows, probs, gray.shape, S["thr"], 96)
    pred = S["pred"][f]
    assert len(pred) == len(owner), (f, len(pred), len(owner))
    return gray, lab, pred


def prep(f: str):
    gray, lab, pred = stage1(f)
    n = len(pred)
    out = {"file": f, "items": []}
    if n == 0:
        return out
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    unet = cv2.imread(str(Path(S["unet"]) / f"{Path(f).stem}.png"), cv2.IMREAD_GRAYSCALE)
    if unet.shape != gray.shape:
        unet = cv2.resize(unet, gray.shape[::-1], interpolation=cv2.INTER_LINEAR)
    selected = np.concatenate([[False], pred > S["tau"]])
    sel_map = selected[lab].astype(np.uint8) * 255
    views = L.gt_masks_for(S["views"][f], S["anns"])
    area = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
    best = []  # per view: (iou [n+1], index of best label per instance)
    for gts in views:
        iou = np.zeros((len(gts), n + 1))
        for gi, g in enumerate(gts):
            inter = np.bincount(lab[g], minlength=n + 1).astype(np.float64)
            iou[gi] = inter / (area + g.sum() - inter + 1e-9)
        best.append(iou)
    slices = ndimage.find_objects(lab)
    for i in range(n):
        if pred[i] <= S["train_tau"] or slices[i] is None:
            continue
        sl = slices[i]
        m_win = lab[sl] == i + 1
        M = rect_affine(m_win, sl[1].start, sl[0].start)
        seed_full = np.zeros(gray.shape, np.uint8)
        seed_full[sl][m_win] = 255
        others = sel_map.copy()
        others[sl][m_win] = 0
        target = np.zeros(gray.shape, np.float32)
        k = 0
        for vi, gts in enumerate(views):
            if len(gts) == 0:
                continue
            j = int(np.argmax(best[vi][:, i + 1]))
            if best[vi][j, i + 1] > 0.2:
                target += gts[j]
                k += 1
        crop = np.stack([warp(c, M) for c in (gray, clahe, unet, seed_full, others)])
        tgt = np.clip(warp(target / max(1, k), M) * 255 + 0.5, 0, 255).astype(np.uint8)
        out["items"].append({"idx": i, "pred": float(pred[i]), "M": M, "crop": crop, "target": tgt,
                             "has_target": k > 0})
    return out


def score_selected(labels, n, n_selected, views):
    """score_views counts every label id as a prediction; only the selected ones are."""
    return [(tp, n_selected - tp, fn, s) for tp, _, fn, s in L.score_views(labels, n, views)]


def evaluate(args):
    f, items = args  # items: (idx, M, prob[H, W] float16) for instances with pred > min(TAUS)
    gray, lab, pred = stage1(f)
    n = len(pred)
    views = L.gt_masks_for(S["views"][f], S["anns"])
    res = {}
    slices = ndimage.find_objects(lab)
    wins = {}
    for idx, M, prob in items:
        wins[idx] = window_prob(prob, M, lab.shape)
    for tau in TAUS:
        sel = np.nonzero(pred > tau)[0]
        keep = np.zeros(n + 1, bool); keep[sel + 1] = True
        base = np.where(keep[lab], lab, 0).astype(np.uint16)
        res[("stage1", 0.0, tau)] = score_selected(base, n, len(sel), views)
        order = sel[np.argsort(-pred[sel], kind="stable")]
        for mode in MODES:
            for t in THRS:
                new = np.zeros(lab.shape, np.uint16)
                for i in order:
                    sl = slices[i]
                    if sl is None:
                        continue
                    seed_win = lab[sl] == i + 1
                    done = False
                    if i in wins:
                        x0, y0, x1, y1, p = wins[i]
                        m = p >= t
                        seed = np.zeros(m.shape, bool)
                        ya, yb = max(sl[0].start, y0), min(sl[0].stop, y1)
                        xa, xb = max(sl[1].start, x0), min(sl[1].stop, x1)
                        if yb > ya and xb > xa:
                            seed[ya - y0:yb - y0, xa - x0:xb - x0] = seed_win[ya - sl[0].start:yb - sl[0].start,
                                                                              xa - sl[1].start:xb - sl[1].start]
                        if mode != "all" and m.any():
                            cc, k = ndimage.label(m, structure=np.ones((3, 3)))
                            if k > 1:
                                hit = np.unique(cc[seed & m]); hit = hit[hit > 0]
                                if len(hit) == 0:
                                    hit = [int(np.argmax(np.bincount(cc.ravel())[1:])) + 1]
                                m = np.isin(cc, hit)
                        ok = True
                        if mode == "guarded":
                            inter = float((m & seed).sum()); uni = float((m | seed).sum())
                            ok = uni > 0 and inter / uni >= 0.5
                        m = m & (new[y0:y1, x0:x1] == 0)
                        if ok and m.sum() >= 96:
                            new[y0:y1, x0:x1][m] = i + 1
                            done = True
                    if not done:
                        m = seed_win & (new[sl] == 0)
                        new[sl][m] = i + 1
                res[(mode, t, tau)] = score_selected(new, n, len(sel), views)
    return res


def train_fold(k, X, Y, has, fold, log_rows):
    import torch
    import torch.nn.functional as F

    dev = torch.device("cuda")
    torch.manual_seed(100 + k)
    model = build_model(pretrained=True).to(dev).to(memory_format=torch.channels_last)
    tr = np.nonzero(has & (fold != k))[0]
    xt = torch.from_numpy(X[tr]).to(dev); yt = torch.from_numpy(Y[tr]).to(dev)
    bs, epochs = 48, S["epochs"]
    steps = epochs * (len(tr) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        tot = 0.0
        for b in range(len(tr) // bs):
            idx = perm[b * bs:(b + 1) * bs]
            x = xt[idx].float() / 255; y = yt[idx].float().unsqueeze(1) / 255
            ang = (torch.rand(bs, device=dev) - 0.5) * 2 * np.deg2rad(6)
            sc = 1 + (torch.rand(bs, device=dev) - 0.5) * 0.2
            cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
            theta = torch.stack([torch.stack([cos, -sin * H / W, (torch.rand(bs, device=dev) - 0.5) * 0.08], 1),
                                 torch.stack([sin * W / H, cos, (torch.rand(bs, device=dev) - 0.5) * 0.08], 1)], 1)
            grid = F.affine_grid(theta, (bs, 6, H, W), align_corners=False)
            xy = F.grid_sample(torch.cat([x, y], 1), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            x, y = xy[:, :5], xy[:, 5:]
            if torch.rand(1).item() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1).item() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            gain = 1 + (torch.rand(bs, 1, 1, 1, device=dev) - 0.5) * 0.3
            x = torch.cat([(x[:, :2] * gain).clamp(0, 1), x[:, 2:]], 1)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(x.contiguous(memory_format=torch.channels_last)).float()
            p = torch.sigmoid(logit)
            dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(logit, y) + 0.5 * dice.mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        log_rows.append({"fold": k, "epoch": ep + 1, "train_loss": tot / max(1, len(tr) // bs), "train_items": len(tr)})
        print(log_rows[-1], flush=True)
    torch.save({"model": model.state_dict(), "W": W, "H": H}, f"/data/runs/stage2_{S['tag']}_fold{k}.pt")
    model.eval()
    va = np.nonzero(fold == k)[0]
    out = np.zeros((len(va), H, W), np.float16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for b in range(0, len(va), 96):
            x = (torch.from_numpy(X[va[b:b + 96]]).to(dev).float() / 255).contiguous(memory_format=torch.channels_last)
            p = torch.sigmoid(model(x).float())
            p = p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1)
            p = p + torch.sigmoid(model(x.flip(-2)).float()).flip(-2)
            p = p + torch.sigmoid(model(x.flip(-1, -2)).float()).flip(-1, -2)
            out[b:b + 96] = (p / 4)[:, 0].cpu().numpy()
    del xt, yt, model
    torch.cuda.empty_cache()
    return va, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--lab", required=True)
    ap.add_argument("--avg_suffix", default="")
    ap.add_argument("--pool", default="")
    ap.add_argument("--unet", default="/data/runs/unet_prob_oof")
    ap.add_argument("--tau", type=float, default=0.24)
    ap.add_argument("--train_tau", type=float, default=0.12)
    ap.add_argument("--thr", type=float, default=0.65)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--save_crops", default="", help="write the crop tensors (X, Y, has, fold) to this .npz")
    ap.add_argument("--eval_only", default="", help="comma list of saved OOF maps to average and evaluate")
    ap.add_argument("--blend", default="", help="oof_probs.npz of another run (same crops) to average with")
    ap.add_argument("--encoder", default="", help="segmentation_models_pytorch encoder (default: the ResNet34 CropUNet)")
    ap.add_argument("--raster", default="coco", help="polygon rasteriser for labels and targets (diagnostic)")
    a = ap.parse_args()
    os.environ["SOLAR_RASTER"] = a.raster
    if a.encoder:
        os.environ["STAGE2_ENCODER"] = a.encoder
    t0 = time.time()
    props, file_dump, fold_of = {}, {}, {}
    for k, d in enumerate(a.dump.split(",")):
        pr = json.loads((Path(d) / "proposals.json").read_text())["predictions"]
        props.update(pr)
        file_dump.update({f: Path(d) for f in pr})
        fold_of.update({f: k for f in pr})
    pools = []
    for group in [g for g in a.pool.split("+") if g]:
        pd_, pp_ = {}, {}
        for d in group.split(","):
            pp = json.loads((Path(d) / "proposals.json").read_text())["predictions"]
            pp_.update(pp); pd_.update({f: Path(d) for f in pp})
        pools.append((pd_, pp_))
    t = np.load(Path(a.lab) / f"instances_thr{a.thr}.npz", allow_pickle=True)
    files = list(t["files"])
    offs = np.concatenate([[0], np.cumsum(np.bincount(t["grp"], minlength=len(files)))])
    pred = {f: t["pred"][offs[i]:offs[i + 1]] for i, f in enumerate(files)}
    views, anns = L.load_gt(files)
    L.G.update(props=props, file_dump=file_dump, avg_suffix=a.avg_suffix, pools=pools)
    S.update(pred=pred, views=views, anns=anns, thr=a.thr, tau=a.tau, train_tau=a.train_tau, unet=a.unet,
             epochs=a.epochs, tag=a.tag)
    out = L.RESULTS / "stage2_refiner" / a.tag
    out.mkdir(parents=True, exist_ok=True)

    with Pool(a.workers) as pool:
        prepped = pool.map(prep, files, chunksize=4)
    meta, X, Y = [], [], []
    for r in prepped:
        for it in r["items"]:
            meta.append((r["file"], it["idx"], it["pred"], it["M"], it["has_target"]))
            X.append(it["crop"]); Y.append(it["target"])
    del prepped
    X = np.stack(X); Y = np.stack(Y)
    has = np.array([m[4] for m in meta]); fold = np.array([fold_of[m[0]] for m in meta])
    print(f"crops {X.shape}, with target {int(has.sum())}, prep {time.time() - t0:.0f}s", flush=True)
    if a.save_crops:
        np.savez(a.save_crops, X=X, Y=Y, has=has, fold=fold)

    probs = np.zeros((len(meta), H, W), np.float16)
    log_rows = []
    if a.eval_only:  # average saved out-of-fold maps (.npy raw or .npz with "probs"), no training
        acc = np.zeros(probs.shape, np.float32)
        srcs = a.eval_only.split(",")
        for src in srcs:
            acc += (np.load(src)["probs"] if src.endswith(".npz") else np.load(src)).astype(np.float32)
        probs = (acc / len(srcs)).astype(np.float16)
    else:
        for k in range(len(a.dump.split(","))):
            va, p = train_fold(k, X, Y, has, fold, log_rows)
            probs[va] = p
        np.save(f"/data/runs/stage2_{a.tag}_oof_raw.npy", probs)
    if a.blend:
        other = np.load(a.blend)["probs"]
        assert other.shape == probs.shape, (other.shape, probs.shape)
        probs = ((probs.astype(np.float32) + other.astype(np.float32)) / 2).astype(np.float16)
    # Crop-level check: soft Dice of the stage-2 map and of the seed against the target.
    yt = Y[has].astype(np.float32) / 255; pt = probs[has].astype(np.float32); st = X[has][:, 3].astype(np.float32) / 255
    crop_dice = {"stage2": float(np.mean(2 * (pt * yt).sum((1, 2)) / (pt.sum((1, 2)) + yt.sum((1, 2)) + 1e-6))),
                 "seed": float(np.mean(2 * (st * yt).sum((1, 2)) / (st.sum((1, 2)) + yt.sum((1, 2)) + 1e-6)))}
    del X, Y

    by_file: dict = {f: [] for f in files}
    for j, (f, idx, pr, M, _) in enumerate(meta):
        if pr > min(TAUS):
            by_file[f].append((idx, M, probs[j]))
    with Pool(a.workers) as pool:
        ev = pool.map(evaluate, list(by_file.items()), chunksize=4)
    sweep = []
    for key in ev[0]:
        row = L.pq([v for r in ev for v in r[key]])
        sweep.append({"mode": key[0], "mask_thr": key[1], "tau": key[2], **row})
    sweep.sort(key=lambda r: (r["tau"], -r["pq"]))
    with (out / "sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep[0])); w.writeheader(); w.writerows(sweep)
    with (out / "instances.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "idx", "pred", "fold", "has_target"])
        w.writerows([[m[0], m[1], round(m[2], 5), fold_of[m[0]], int(m[4])] for m in meta])
    with (out / "train_log.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log_rows[0]) if log_rows else ["fold"]); w.writeheader(); w.writerows(log_rows)
    np.savez_compressed(out / "oof_probs.npz", probs=probs, M=np.stack([m[3] for m in meta]),
                        files=np.array([m[0] for m in meta]), idx=np.array([m[1] for m in meta]))
    summary = {"args": vars(a), "images": len(files), "crops": len(meta), "crops_with_target": int(has.sum()),
               "crop_soft_dice": crop_dice,
               "stage1": {str(r["tau"]): r for r in sweep if r["mode"] == "stage1"},
               "best_per_tau": {str(tau): max((r for r in sweep if r["tau"] == tau and r["mode"] != "stage1"),
                                              key=lambda r: r["pq"]) for tau in TAUS},
               "sweep": sweep, "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "sweep"}, indent=2, default=float))


if __name__ == "__main__":
    main()
