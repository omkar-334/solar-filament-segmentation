"""End-to-end inference of the final (v7) pipeline for one image.

    1. Two YOLO11s-seg detectors (1024 px and 1536 px) propose boxes (confidence >= 0.05).
    2. Seed-conditioned crop refiners turn every box into a probability crop
       (ResNet18/256 and ResNet34/384 averaged for the 1024 px detector, ResNet18/256
       for the 1536 px detector), four flips each.
    3. All candidates are pasted in detector-score order with pixel claiming
       (threshold 0.65, minimum area 96 px).
    4. A LightGBM selector predicts the expected matched IoU of every pasted instance;
       instances at or below tau = 0.24 are dropped.
    5. Stage 2 re-segments every kept instance in a rotation-normalised crop
       (five fold models, four flips, threshold 0.40).

Weights are read from one directory, see ``WEIGHT_FILES``.

``version="v11"`` replaces steps 4 and 5: instances down to a selector value of 0.12 are
re-segmented by fine-tuned SAM 2.1 models (base-plus and large, five folds each, four
flips, threshold 0.40), the IoU head of five further SAM 2.1 base-plus models scores every
instance, and an instance is kept when the mean of selector value and SAM score exceeds
0.24. The extra weights are listed in ``WEIGHT_FILES_V11``; transformers is required.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from . import crop_refiner as _cr
from . import stage2
from .crop_refiner import CropUNet, make_inference_crop
from .inference import load_model, predict_probability
from .postproc import rescored_labels

WEIGHT_FILES = {
    "detector_1024": "detector_1024.pt",
    "detector_1536": "detector_1536.pt",
    "refiner_r18_1024": "refiner_r18_1024.pt",
    "refiner_r34c384_1024": "refiner_r34c384_1024.pt",
    "refiner_r18_1536": "refiner_r18_1536.pt",
    "unet_dense": "unet_dense.pt",
    "selector": "selector.txt",
    **{f"stage2_fold{k}": f"stage2_fold{k}.pt" for k in range(5)},
}
TAU, REFINER_THRESHOLD, STAGE2_THRESHOLD, MIN_CONF = 0.24, 0.65, 0.40, 0.05
SAM_MASK_FILES = [f"sam_{t}_fold{k}.pt" for t in ("bplus", "large") for k in range(5)]
SAM_SCORE_FILES = [f"sam_score_fold{k}.pt" for k in range(5)]
WEIGHT_FILES_V11 = {**{k: v for k, v in WEIGHT_FILES.items() if not k.startswith("stage2_")},
                    **{f[:-3]: f for f in SAM_MASK_FILES + SAM_SCORE_FILES}}
FLOOR_V11 = 0.12


def detect_version(weights_dir: str | Path) -> str:
    """``v11`` when the SAM 2.1 weight files are in ``weights_dir``, otherwise ``v7``."""
    return "v11" if (Path(weights_dir) / SAM_MASK_FILES[0]).is_file() else "v7"


def _resize_stack(a: np.ndarray, size: int) -> np.ndarray:
    if not len(a) or a.shape[-1] == size:
        return a
    return np.stack([cv2.resize(x, (size, size), interpolation=cv2.INTER_LINEAR) for x in a])


class FilamentPipeline:
    def __init__(self, weights_dir: str | Path, device: str | torch.device | None = None, version: str = "auto"):
        import lightgbm as lgb
        from ultralytics import YOLO

        if version == "auto":
            version = detect_version(weights_dir)
        assert version in ("v7", "v11"), version
        self.version = version
        w = self.weights_dir = Path(weights_dir)
        files = WEIGHT_FILES if version == "v7" else WEIGHT_FILES_V11
        missing = [f for f in files.values() if not (w / f).is_file()]
        if missing:
            raise FileNotFoundError(f"missing weight files in {w}: {missing}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.detectors = {1024: YOLO(str(w / WEIGHT_FILES["detector_1024"])),
                          1536: YOLO(str(w / WEIGHT_FILES["detector_1536"]))}
        self.refiners = {1024: [self._refiner(w / WEIGHT_FILES["refiner_r18_1024"]),
                                self._refiner(w / WEIGHT_FILES["refiner_r34c384_1024"])],
                         1536: [self._refiner(w / WEIGHT_FILES["refiner_r18_1536"])]}
        self.unet, _ = load_model(w / WEIGHT_FILES["unet_dense"], self.device)
        self.selector = lgb.Booster(model_file=str(w / WEIGHT_FILES["selector"]))
        self.stage2 = []
        if version == "v11":
            from . import sam_refiner
            self.sam_masks = [sam_refiner.load(w / f, self.device) for f in SAM_MASK_FILES]
            self.sam_scores = [sam_refiner.load(w / f, self.device) for f in SAM_SCORE_FILES]
        for k in range(5 if version == "v7" else 0):
            m = stage2.build_model(pretrained=False)
            m.load_state_dict(torch.load(w / WEIGHT_FILES[f"stage2_fold{k}"], map_location="cpu")["model"])
            self.stage2.append(m.to(self.device).eval())

    def _refiner(self, path: Path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = CropUNet(pretrained=False, encoder_name=state.get("encoder", "resnet18"))
        model.load_state_dict(state["model"])
        return model.to(self.device).eval(), int(state.get("crop_size", 256))

    def _refine(self, model, crops: list[np.ndarray]) -> np.ndarray:
        amp = self.device.type == "cuda"
        acc = []
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.float16, enabled=amp):
            for chunk in torch.from_numpy(np.stack(crops)).split(32):
                chunk = chunk.to(self.device)
                p = torch.sigmoid(model(chunk).float())
                p = p + torch.sigmoid(model(chunk.flip(-1)).float()).flip(-1)
                p = p + torch.sigmoid(model(chunk.flip(-2)).float()).flip(-2)
                p = p + torch.sigmoid(model(chunk.flip(-1, -2)).float()).flip(-1, -2)
                acc.append((p / 4)[:, 0].cpu().numpy())
        return np.clip(np.concatenate(acc) * 255 + 0.5, 0, 255).astype(np.uint8)

    def candidates(self, gray: np.ndarray, size: int):
        """Proposal rows, uint8 refiner probability crops and detector mask crops of one detector."""
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        dev = 0 if self.device.type == "cuda" else self.device.type
        det = self.detectors[size].predict(source=rgb, imgsz=size, device=dev, conf=MIN_CONF, iou=0.7,
                                           max_det=100, verbose=False, retina_masks=True)[0]
        if det.boxes is None or not len(det.boxes):
            return [], np.zeros((0, 256, 256), np.uint8), np.zeros((0, 256, 256), np.uint8)
        boxes = det.boxes.xyxy.cpu().numpy()
        scores = det.boxes.conf.cpu().numpy()
        masks = det.masks.data.cpu().numpy() > 0.5 if det.masks is not None else None
        rows, yolo = [], []
        for k, (box, score) in enumerate(zip(boxes, scores)):
            _cr.set_crop_size(256)
            _, (x0, y0, side) = make_inference_crop(gray, tuple(map(float, box)))
            ym = np.zeros((side, side), dtype=np.uint8)
            if masks is not None:
                xa, ya = max(0, x0), max(0, y0)
                xb, yb = min(gray.shape[1], x0 + side), min(gray.shape[0], y0 + side)
                ym[ya - y0:yb - y0, xa - x0:xb - x0] = masks[k][ya:yb, xa:xb]
            yolo.append(cv2.resize(ym * 255, (256, 256), interpolation=cv2.INTER_AREA))
            rows.append({"bbox": [float(v) for v in box], "score": float(score), "crop": [x0, y0, side]})
        yolo = np.stack(yolo)
        per_refiner = []
        for model, crop_size in self.refiners[size]:
            _cr.set_crop_size(crop_size)
            per_refiner.append(self._refine(model, [make_inference_crop(gray, tuple(r["bbox"]))[0] for r in rows]))
        _cr.set_crop_size(256)
        probs = per_refiner[0]
        if len(per_refiner) == 2:  # average the two refiners at the larger crop size
            big = max(p.shape[-1] for p in per_refiner)
            a, b = (_resize_stack(p, big) for p in per_refiner)
            probs = ((a.astype(np.uint16) + b) // 2).astype(np.uint8)
            yolo = _resize_stack(yolo, big)
        return rows, probs, yolo

    def predict(self, gray: np.ndarray, return_stages: bool = False):
        """Label map (uint16, 0 = background) of one 2048 x 2048 grayscale image."""
        rows, probs, yolo = self.candidates(gray, 1024)
        rows2, probs2, yolo2 = self.candidates(gray, 1536)
        n1 = len(rows)
        if len(rows2):
            size = probs.shape[-1] if len(probs) else probs2.shape[-1]
            probs2, yolo2 = _resize_stack(probs2, size), _resize_stack(yolo2, size)
            probs = np.concatenate([probs, probs2]) if len(probs) else probs2
            yolo = np.concatenate([yolo, yolo2]) if len(yolo) else yolo2
            rows = rows + rows2
        if not rows:
            empty = np.zeros(gray.shape, np.uint16)
            return (empty, {"stage1": empty, "candidates": (0, 0)}) if return_stages else empty
        tau1 = TAU if self.version == "v7" else FLOOR_V11
        labels1, kept = rescored_labels(rows, probs, yolo, gray, self.selector, tau1, REFINER_THRESHOLD)
        dense = predict_probability(self.unet, gray, self.device, 1024, 256, 4)
        dense = np.clip(dense * 255 + 0.5, 0, 255).astype(np.uint8)
        labels = labels1
        if len(kept):
            items = stage2.instance_crops(labels1, gray, dense)
            crops = np.stack([c for _, _, c in items])
            if self.version == "v7":
                crop_probs = stage2.predict_crops(self.stage2, crops, self.device)
            else:
                from . import sam_refiner
                crop_probs = sam_refiner.predict_masks(self.sam_masks, crops, self.device)
            new, _ = stage2.paste_refined(labels1, kept, [(i, M, p) for (i, M, _), p in zip(items, crop_probs)],
                                          STAGE2_THRESHOLD)
            if self.version == "v11":  # second selection: mean of selector value and SAM score
                score = np.zeros(len(kept), np.float32)
                score[[i for i, _, _ in items]] = sam_refiner.predict_scores(self.sam_scores, crops, self.device)
                drop = np.nonzero(0.5 * (np.asarray(kept) + score) <= TAU)[0] + 1
                new[np.isin(new, drop)] = 0
            ids = [i for i in range(1, len(kept) + 1) if (new == i).any()]
            remap = np.zeros(len(kept) + 1, np.uint16)
            remap[ids] = np.arange(1, len(ids) + 1, dtype=np.uint16)
            labels = remap[new]
        if return_stages:
            return labels, {"stage1": labels1, "dense": dense, "candidates": (n1, len(rows2)),
                            "selector_value": kept}
        return labels
