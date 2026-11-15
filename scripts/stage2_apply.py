"""Apply the stage-2 refiner (scripts/stage2_refiner.py) to a stage-1 test submission.

Every row of the input submission is re-segmented in its rotation-normalised crop by the
mean of the five fold models (four flips each). Masks are pasted in selector order with
pixel claiming, exactly as in the out-of-fold evaluation (rule "all"): the stage-2 mask
replaces the stage-1 mask when at least 96 unclaimed pixels remain, otherwise the
stage-1 mask is kept.

Usage (GPU): python stage2_apply.py --csv=<stage-1 submission.csv> --pred=<..._pred.json>
             --models=a.pt,sam2@b.pt,... --out=<new submission.csv> [--t=0.40] [--tag=v7]
Writes the submission and results/stage2_apply/<tag>/{summary.json,rows.csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_filament.rle import audit_submission, decode_mask, encode_mask  # noqa: E402
from solar_filament import stage2 as R  # noqa: E402

TEST = Path(os.environ.get("SOLAR_TEST", "MAGFiLO_1.0_Kaggle_2026/test/test_images"))
RESULTS = Path(os.environ.get("SOLAR_RESULTS", Path(__file__).resolve().parents[1] / "results"))
S: dict = {}


def labels_of(stem: str) -> np.ndarray:
    lab = np.zeros((2048, 2048), np.uint16)
    for i, (_, rle) in enumerate(S["rows"][stem]):
        lab[decode_mask(rle, 2048, 2048)] = i + 1
    return lab


def prep(stem: str):
    lab = labels_of(stem)
    gray = cv2.imread(str(TEST / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    unet = cv2.imread(str(Path(S["unet"]) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if unet.shape != gray.shape:
        unet = cv2.resize(unet, gray.shape[::-1], interpolation=cv2.INTER_LINEAR)
    sel_map = (lab > 0).astype(np.uint8) * 255
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        m_win = lab[sl] == i + 1
        M = R.rect_affine(m_win, sl[1].start, sl[0].start)
        seed = np.zeros(gray.shape, np.uint8)
        seed[sl][m_win] = 255
        others = sel_map.copy()
        others[sl][m_win] = 0
        out.append((i, M, np.stack([R.warp(c, M) for c in (gray, clahe, unet, seed, others)])))
    return stem, out


def paste(args):
    stem, items = args  # items: (idx, M, prob)
    lab = labels_of(stem)
    ids = [r[0] for r in S["rows"][stem]]
    pred = np.array([S["pred"].get(i, 0.0) for i in ids])
    slices = ndimage.find_objects(lab)
    wins = {idx: R.window_prob(prob, M, lab.shape) for idx, M, prob in items}
    new = np.zeros(lab.shape, np.uint16)
    info = []
    for i in np.argsort(-pred, kind="stable"):
        sl = slices[i] if i < len(slices) else None
        if sl is None:
            continue
        seed_win = lab[sl] == i + 1
        used = "stage1"
        if i in wins:
            x0, y0, x1, y1, p = wins[i]
            m = (p >= S["t"]) & (new[y0:y1, x0:x1] == 0)
            if m.sum() >= 96:
                new[y0:y1, x0:x1][m] = i + 1
                used = "stage2"
        if used == "stage1":
            m = seed_win & (new[sl] == 0)
            new[sl][m] = i + 1
        a0 = int(seed_win.sum()); a1 = int((new == i + 1).sum())
        inter = int(((new == i + 1) & (lab == i + 1)).sum())
        info.append({"filament_id": ids[i], "pred": float(pred[i]), "used": used, "area_stage1": a0,
                     "area_out": a1, "iou_with_stage1": inter / max(1, a0 + a1 - inter)})
    keep = np.ones(len(ids), bool)
    if S.get("sc") is not None:  # second selection: mean of the selector value and the SAM quality score
        keep = np.array([0.5 * (pred[i] + S["sc"].get(ids[i], 0.0)) > S["select"] for i in range(len(ids))])
    for r in info:
        r["kept"] = bool(keep[ids.index(r["filament_id"])])
    rles = [(i, encode_mask(new == i + 1)) for i in range(len(ids)) if keep[i] and (new == i + 1).any()]
    return stem, rles, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unet", default="/data/runs/unet_prob_test")
    ap.add_argument("--t", type=float, default=0.40)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--tag", default="v7")
    ap.add_argument("--score_models", default="", help="sam2_refiner.py --score checkpoints (comma list)")
    ap.add_argument("--select", type=float, default=0.24, help="keep if mean(selector value, SAM score) exceeds this")
    a = ap.parse_args()
    t0 = time.time()
    csv.field_size_limit(10 ** 9)
    rows = defaultdict(list)
    with open(a.csv) as fh:
        for r in csv.DictReader(fh):
            rows[r["filament_id"].rsplit("_", 1)[0]].append((r["filament_id"], r["segmentation_rle"]))
    pred = json.loads(Path(a.pred).read_text())
    n_in = sum(len(v) for v in rows.values())
    assert set(i for v in rows.values() for i, _ in v) <= set(pred), "selector values missing for some rows"
    S.update(rows=dict(rows), pred=pred, unet=a.unet, t=a.t, sc=None)
    stems = sorted(rows)
    with Pool(a.workers) as pool:
        prepped = pool.map(prep, stems, chunksize=2)
    meta = [(stem, i, M) for stem, items in prepped for i, M, _ in items]
    X = np.stack([c for _, items in prepped for _, _, c in items])
    del prepped

    import torch
    dev = torch.device("cuda")
    probs = np.zeros((len(X), R.H, R.W), np.float32)
    paths = a.models.split(",")
    for path in paths:
        enc, _, path = path.rpartition("@")  # optional "<smp encoder>@<checkpoint>"
        if enc == "sam2":  # checkpoint of scripts/sam2_refiner.py
            import sam2_refiner as SR
            from transformers import Sam2Model
            ck = torch.load(path, map_location="cpu")
            net = Sam2Model.from_pretrained(ck["name"])
            net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
            net = net.to(dev).eval()
            mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

            def model(x, net=net, mean=mean, std=std):
                return torch.cat([SR.forward(net, x[i:i + 16], 0.0, mean, std) for i in range(0, len(x), 16)])
        else:
            os.environ["STAGE2_ENCODER"] = enc
            model = R.build_model(pretrained=False)
            model.load_state_dict(torch.load(path, map_location="cpu")["model"])
            model = model.to(dev).to(memory_format=torch.channels_last).eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for b in range(0, len(X), 96):
                x = (torch.from_numpy(X[b:b + 96]).to(dev).float() / 255).contiguous(memory_format=torch.channels_last)
                p = torch.sigmoid(model(x).float())
                p = p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1)
                p = p + torch.sigmoid(model(x.flip(-2)).float()).flip(-2)
                p = p + torch.sigmoid(model(x.flip(-1, -2)).float()).flip(-1, -2)
                probs[b:b + 96] += (p / 4)[:, 0].cpu().numpy() / len(paths)
        del model
    if a.score_models:
        import sam2_refiner as SR
        from transformers import Sam2Model
        sc = np.zeros(len(X), np.float32)
        sp = a.score_models.split(",")
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        for path in sp:
            ck = torch.load(path, map_location="cpu")
            net = Sam2Model.from_pretrained(ck["name"])
            net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
            net = net.to(dev).eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for b in range(0, len(X), 16):
                    x = torch.from_numpy(X[b:b + 16]).to(dev).float() / 255
                    sc[b:b + 16] += SR.forward(net, x, 0.0, mean, std, with_score=True)[1].cpu().numpy() / len(sp)
            del net
        S.update(sc={rows[stem][i][0]: float(sc[j]) for j, (stem, i, M) in enumerate(meta)}, select=a.select)
    by_stem = defaultdict(list)
    for j, (stem, i, M) in enumerate(meta):
        by_stem[stem].append((i, M, probs[j]))
    with Pool(a.workers) as pool:
        pasted = pool.map(paste, [(s, by_stem[s]) for s in stems], chunksize=2)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = []
    with out.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["filament_id", "segmentation_rle"])
        for stem, rles, inf in pasted:
            for k, (_, rle) in enumerate(rles, 1):
                w.writerow([f"{stem}_{k}", rle])
            info += inf
    shapes = {p.stem: (2048, 2048) for p in sorted(TEST.glob("*.jp*g"))}
    n_out = audit_submission(out, shapes, expected_image_ids=shapes.keys())
    res = RESULTS / "stage2_apply" / a.tag
    res.mkdir(parents=True, exist_ok=True)
    with (res / "rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(info[0])); w.writeheader(); w.writerows(info)
    iou = np.array([r["iou_with_stage1"] for r in info])
    summary = {"args": vars(a), "images": len(stems), "rows_in": n_in, "rows_out": int(n_out),
               "stage2_used": sum(r["used"] == "stage2" for r in info),
               "stage1_kept": sum(r["used"] == "stage1" for r in info),
               "mean_iou_with_stage1": float(iou.mean()), "share_iou_below_0.5": float((iou < 0.5).mean()),
               "area_ratio_median": float(np.median([r["area_out"] / max(1, r["area_stage1"]) for r in info])),
               "seconds": round(time.time() - t0, 1)}
    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Audit passed: {n_out} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
