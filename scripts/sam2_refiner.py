"""SAM 2.1 as the stage-2 refiner: box-prompted, fully fine-tuned, on the stage-2 crops.

Reads the crop tensors written by stage2_refiner.py --save_crops (X: gray, CLAHE, dense map,
seed mask, other instances; Y: mean annotator mask). The first three channels are stretched
to 1024 x 1024 as the image, the bounding box of the seed mask is the prompt. One fold per
call so that folds run in parallel; --merge assembles the out-of-fold maps into one array
that stage2_refiner.py --eval_only scores with the exact pooled PQ.

Usage (GPU): python sam2_refiner.py --fold=0 [--model=facebook/sam2.1-hiera-small] [--epochs=6] [--limit=0]
             python sam2_refiner.py --merge
Writes /data/runs/sam2_<tag>_fold{k}.npz, /data/runs/sam2_<tag>_oof.npy and
results/sam2_refiner/<tag>/{fold{k}.json,train_log_fold{k}.csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar_filament.sam_refiner import H, S, W, boxes_of, forward  # noqa: E402,F401

RESULTS = Path(os.environ.get("SOLAR_RESULTS", Path(__file__).resolve().parents[1] / "results"))


def soft_skeleton(x, iters: int = 10):
    """Differentiable skeleton by iterated min and max pooling (soft-clDice, Shit et al., CVPR 2021)."""
    import torch.nn.functional as F

    def erode(v):
        return torch_min(-F.max_pool2d(-v, (3, 1), 1, (1, 0)), -F.max_pool2d(-v, (1, 3), 1, (0, 1)))

    def opened(v):
        return F.max_pool2d(erode(v), 3, 1, 1)

    import torch
    torch_min = torch.minimum
    skel = F.relu(x - opened(x))
    for _ in range(iters):
        x = erode(x)
        delta = F.relu(x - opened(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(p, y, iters: int = 10, eps: float = 1.0):
    """1 - clDice per sample for probabilities p and (soft) targets y, both [B, 1, H, W]."""
    sp, sy = soft_skeleton(p, iters), soft_skeleton(y, iters)
    tprec = ((sp * y).sum((1, 2, 3)) + eps) / (sp.sum((1, 2, 3)) + eps)
    tsens = ((sy * p).sum((1, 2, 3)) + eps) / (sy.sum((1, 2, 3)) + eps)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="/data/runs/stage2_crops.npz")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="small")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--cldice", type=float, default=0.0, help="weight of the soft-clDice loss term")
    ap.add_argument("--score", action="store_true", help="also train SAM's mask-quality head on the matched IoU")
    ap.add_argument("--instances", default="/data/results/stage2_refiner/crops/instances.csv")
    ap.add_argument("--per_view", default="/data/results/rescore2_lab/dump_kf5b_fold0_x5_avg384_pool1536/per_view.npy")
    a = ap.parse_args()
    t0 = time.time()
    res = RESULTS / "sam2_refiner" / a.tag
    res.mkdir(parents=True, exist_ok=True)
    z = np.load(a.crops)
    X, Y, has, fold = z["X"], z["Y"], z["has"], z["fold"]
    if a.merge:
        probs = np.zeros((len(X), H, W), np.float16)
        seen = np.zeros(len(X), bool)
        score = np.zeros(len(X), np.float32)
        for k in sorted(set(fold.tolist())):
            f = np.load(f"/data/runs/sam2_{a.tag}_fold{k}.npz")
            probs[f["idx"]] = f["probs"]; seen[f["idx"]] = True
            if "score" in f.files:
                score[f["idx"]] = f["score"]
        np.save(f"/data/runs/sam2_{a.tag}_oof.npy", probs)
        np.save(res / "oof_score.npy", score)
        (res / "merge.json").write_text(json.dumps({"crops": len(X), "covered": int(seen.sum())}))
        print("merged", len(X), int(seen.sum()), flush=True)
        return

    import torch
    import torch.nn.functional as F
    from transformers import Sam2Model

    dev = torch.device("cuda")
    torch.manual_seed(100 + a.fold)
    model = Sam2Model.from_pretrained(a.model).to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    yq = None
    if a.score:  # matched IoU (mean over annotator views) of every crop, aligned through the selector value
        files = [r["file"] for r in csv.DictReader(open(a.instances))]
        pv = {d["file"]: d for d in np.load(a.per_view, allow_pickle=True)}
        yq = np.zeros(len(X), np.float32); off = 0
        order = list(dict.fromkeys(files))
        for f in order:
            n = files.count(f); d = pv[f]
            assert len(d["pred"]) == n, (f, len(d["pred"]), n)
            yq[off:off + n] = np.asarray(d["y_views"]).mean(axis=0); off += n
        assert off == len(X)
        print("score target: mean %.3f, share > 0: %.3f" % (yq.mean(), (yq > 0).mean()), flush=True)
    tr = np.nonzero((has | a.score) & (fold != a.fold))[0]
    va = np.nonzero(fold == a.fold)[0]
    if a.limit:
        tr, va = tr[:a.limit], va[:a.limit]
    steps = a.epochs * (len(tr) // a.bs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    log = []
    rng = np.random.default_rng(a.fold)
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for b in range(len(tr) // a.bs):
            idx = np.sort(perm[b * a.bs:(b + 1) * a.bs])
            x = torch.from_numpy(X[idx]).to(dev).float() / 255
            y = torch.from_numpy(Y[idx]).to(dev).float().unsqueeze(1) / 255
            if rng.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if rng.random() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            gain = 1 + (torch.rand(len(idx), 1, 1, 1, device=dev) - 0.5) * 0.3
            x = torch.cat([(x[:, :2] * gain).clamp(0, 1), x[:, 2:]], 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, sc = forward(model, x, 0.08, mean, std, with_score=True)
            p = torch.sigmoid(logit)
            hw = torch.from_numpy(has[idx]).to(dev).float()
            dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
            bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none").mean((1, 2, 3))
            per = bce + 0.5 * dice
            if a.cldice:
                per = per + a.cldice * cldice_loss(p, y)
            loss = (per * hw).sum() / hw.sum().clamp(min=1)
            if a.score:
                loss = loss + F.mse_loss(sc, torch.from_numpy(yq[idx]).to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        log.append({"fold": a.fold, "epoch": ep + 1, "train_loss": tot / max(1, len(tr) // a.bs),
                    "train_items": len(tr), "seconds": round(time.time() - t0, 1)})
        print(log[-1], flush=True)
    model.eval()
    out = np.zeros((len(va), H, W), np.float16)
    vs = np.zeros(len(va), np.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, len(va), 16):
            x = torch.from_numpy(X[va[b:b + 16]]).to(dev).float() / 255
            l0, s0 = forward(model, x, 0.0, mean, std, with_score=True)
            p = torch.sigmoid(l0)
            vs[b:b + 16] = s0.cpu().numpy()
            p = p + torch.sigmoid(forward(model, x.flip(-1), 0.0, mean, std)).flip(-1)
            p = p + torch.sigmoid(forward(model, x.flip(-2), 0.0, mean, std)).flip(-2)
            p = p + torch.sigmoid(forward(model, x.flip(-1, -2), 0.0, mean, std)).flip(-1, -2)
            out[b:b + 16] = (p / 4)[:, 0].float().cpu().numpy()
    hv = has[va]
    yt = Y[va][hv].astype(np.float32) / 255; pt = out[hv].astype(np.float32); st = X[va][hv][:, 3].astype(np.float32) / 255
    sd = lambda q: float(np.mean(2 * (q * yt).sum((1, 2)) / (q.sum((1, 2)) + yt.sum((1, 2)) + 1e-6)))  # noqa: E731
    summary = {"args": vars(a), "train": len(tr), "val": len(va), "soft_dice_sam2": sd(pt), "soft_dice_seed": sd(st),
               "seconds": round(time.time() - t0, 1)}
    if not a.limit:
        torch.save({"model": {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                    "name": a.model}, f"/data/runs/sam2_{a.tag}_fold{a.fold}.pt")
        np.savez(f"/data/runs/sam2_{a.tag}_fold{a.fold}.npz", idx=va, probs=out, score=vs)
    (res / f"fold{a.fold}.json").write_text(json.dumps(summary, indent=2))
    with (res / f"train_log_fold{a.fold}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
