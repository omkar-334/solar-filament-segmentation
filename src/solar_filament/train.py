from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    FilamentCropDataset,
    grouped_split,
    polygon_mask,
    prepare_records,
    view_balanced_sampler,
)
from .inference import predict_instances
from .metrics import aggregate_pq, pq_counts
from .model import (
    build_model,
    binary_dice_bce,
    discriminative_instance_loss,
    spine_heatmap_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment_on_gpu(
    x: torch.Tensor, *targets: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Fast tensor-only geometric and mild intensity augmentation on the active device."""
    batch = x.shape[0]
    for dim in (-1, -2):
        select = torch.rand(batch, device=x.device) < 0.5
        if select.any():
            x[select] = torch.flip(x[select], (dim,))
            for target in targets:
                target[select] = torch.flip(target[select], (dim,))
    rotations = torch.randint(0, 4, (batch,), device=x.device)
    for k in (1, 2, 3):
        select = rotations == k
        if select.any():
            x[select] = torch.rot90(x[select], k, (-2, -1))
            for target in targets:
                target[select] = torch.rot90(target[select], k, (-2, -1))
    gain = 0.85 + 0.30 * torch.rand((batch, 1, 1, 1), device=x.device)
    bias = -0.06 + 0.12 * torch.rand((batch, 1, 1, 1), device=x.device)
    return ((x * gain + bias).clamp_(0.0, 1.0), *targets)


def evaluate(
    model: torch.nn.Module,
    val_records: list[dict],
    coco: dict,
    device: torch.device,
    threshold: float,
    min_area: int,
    tile_size: int,
    overlap: int,
    batch_size: int,
    merge_distance: int,
) -> dict:
    annotations_by_image: dict[str, list[dict]] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)
    images_by_id = {image["id"]: image for image in coco["images"]}
    prediction_by_file: dict[str, np.ndarray] = {}
    gt_metrics = []
    for record in tqdm(val_records, desc="validate", leave=False):
        filename = record["file_name"]
        if filename not in prediction_by_file:
            image = np.asarray(
                Image.open(record["image_path"]).convert("L"), dtype=np.uint8
            )
            _, prediction_by_file[filename] = predict_instances(
                model,
                image,
                device,
                threshold=threshold,
                min_area=min_area,
                tile_size=tile_size,
                overlap=overlap,
                batch_size=batch_size,
                merge_distance=merge_distance,
            )
        image_meta = images_by_id[record["image_id"]]
        gt_masks = [
            polygon_mask(
                ann.get("segmentation"), image_meta["height"], image_meta["width"]
            ).astype(bool)
            for ann in annotations_by_image.get(record["image_id"], [])
        ]
        gt_metrics.append(pq_counts(prediction_by_file[filename], gt_masks))
    return aggregate_pq(gt_metrics)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_pq: float,
    args,
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_pq": best_pq,
            "base_channels": args.base_channels,
            "architecture": args.architecture,
            "config": vars(args),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def train(args: argparse.Namespace, checkpoint_callback=None) -> Path:
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root, cache_dir, output_dir = (
        Path(args.data_root),
        Path(args.cache_dir),
        Path(args.output_dir),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = (
        data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    coco = json.loads(annotation_path.read_text())
    use_spine_instances = args.architecture == "resnet34_unetpp_spine"
    eval_threshold = args.threshold
    eval_min_area = args.min_area
    if eval_threshold is None:
        eval_threshold = 0.4 if use_spine_instances else 0.6
    if eval_min_area is None:
        eval_min_area = 64 if use_spine_instances else 512
    args.threshold = eval_threshold
    args.min_area = eval_min_area
    records = prepare_records(
        data_root, cache_dir, preserve_instances=use_spine_instances
    )
    train_records, val_records = grouped_split(records, args.val_fraction, args.seed)
    train_files_json = getattr(args, "train_files_json", "")
    if train_files_json:
        # k-fold / full-data override: train on the listed files, validate on the rest.
        keep = set(json.loads(Path(train_files_json).read_text()))
        train_records = [r for r in records if r["file_name"] in keep]
        val_records = [r for r in records if r["file_name"] not in keep]
    (output_dir / "validation_files.txt").write_text(
        "\n".join(sorted({r["file_name"] for r in val_records})) + "\n"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True)
    )

    dataset = FilamentCropDataset(
        train_records,
        patch_size=args.patch_size,
        include_instances=use_spine_instances,
        spine_width=args.spine_width,
    )
    sampler = view_balanced_sampler(
        train_records, samples_per_image=args.samples_per_image
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    resume_path_value = getattr(args, "resume_checkpoint", "")
    resume_path = Path(resume_path_value) if resume_path_value else None
    if resume_path is not None and not resume_path.is_file():
        resume_path = None
    resume_exists = resume_path is not None
    model = build_model(
        architecture=args.architecture,
        base_channels=args.base_channels,
        pretrained_encoder=(not resume_exists) and args.architecture in {
            "resnet34_unet",
            "resnet34_unetpp",
            "resnet34_unetpp_spine",
            "ijepa_vith14_unet",
        },
    ).to(device=device)
    if checkpoint_callback is not None:
        checkpoint_callback()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.architecture == "ijepa_vith14_unet":
        optimizer_groups = [
            {"params": list(model.decoder_parameters()), "lr": args.learning_rate},
            {
                "params": list(model.encoder_parameters()),
                "lr": getattr(args, "encoder_learning_rate", 1e-5),
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )
    best_pq = -1.0
    best_path = output_dir / "best.pt"
    start_epoch = 1
    if resume_exists:
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        if "scaler" in resume:
            scaler.load_state_dict(resume["scaler"])
        best_pq = float(resume.get("best_pq", -1.0))
        start_epoch = int(resume["epoch"]) + 1
        if "python_rng_state" in resume:
            random.setstate(resume["python_rng_state"])
            np.random.set_state(resume["numpy_rng_state"])
            torch.set_rng_state(resume["torch_rng_state"])
            if torch.cuda.is_available() and resume.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        print(
            f"Resuming from {resume_path} after epoch {start_epoch - 1}; "
            f"best_pq={best_pq:.6f}",
            flush=True,
        )
    log_path = output_dir / "metrics.jsonl"
    print(
        f"device={device}; train_views={len(train_records)}; train_files={len({r['file_name'] for r in train_records})}; "
        f"val_views={len(val_records)}; val_files={len({r['file_name'] for r in val_records})}"
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}; amp={amp_dtype}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss, running_mask, running_spine, running_instance = 0.0, 0.0, 0.0, 0.0
        n_batches = 0
        start_time = time.time()
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            x = batch[0].to(device, non_blocking=True)
            targets = [target.to(device, non_blocking=True) for target in batch[1:]]
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            x, *targets = augment_on_gpu(x, *targets)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                outputs = model(x).float()
                if use_spine_instances:
                    foreground, spine, embeddings = (
                        outputs[:, 0:1],
                        outputs[:, 1:2],
                        outputs[:, 2:10],
                    )
                    mask_target, spine_target, instance_target = targets
                    mask_loss = binary_dice_bce(foreground, mask_target)
                    spine_loss = spine_heatmap_loss(spine, spine_target)
                    instance_loss = discriminative_instance_loss(
                        embeddings, instance_target
                    )
                    loss = (
                        mask_loss
                        + args.spine_loss_weight * spine_loss
                        + args.embedding_loss_weight * instance_loss
                    )
                else:
                    mask_loss = binary_dice_bce(outputs, targets[0])
                    spine_loss = outputs.new_zeros(())
                    instance_loss = outputs.new_zeros(())
                    loss = mask_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            running_mask += float(mask_loss.detach())
            running_spine += float(spine_loss.detach())
            running_instance += float(instance_loss.detach())
            n_batches += 1
            progress.set_postfix(
                loss=f"{running_loss / n_batches:.4f}",
                mask=f"{running_mask / n_batches:.4f}",
                spine=f"{running_spine / n_batches:.4f}",
                inst=f"{running_instance / n_batches:.4f}",
            )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, n_batches),
            "train_mask_loss": running_mask / max(1, n_batches),
            "train_spine_loss": running_spine / max(1, n_batches),
            "train_instance_loss": running_instance / max(1, n_batches),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - start_time,
        }

        if val_records and (epoch % args.eval_every == 0 or epoch == args.epochs):
            model.eval()
            validation = evaluate(
                model,
                val_records,
                coco,
                device,
                eval_threshold,
                eval_min_area,
                args.tile_size,
                args.tile_overlap,
                args.infer_batch_size,
                args.merge_distance,
            )
            record.update({f"val_{k}": v for k, v in validation.items()})
            score = float(validation["pq_ratio_of_sums"])
            if score > best_pq:
                best_pq = score
                save_checkpoint(
                    best_path, model, optimizer, scheduler, scaler, epoch, best_pq, args
                )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_pq,
            args,
        )
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        if checkpoint_callback is not None:
            checkpoint_callback()

    return best_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a tiled solar-filament U-Net baseline."
    )
    p.add_argument("--data-root", default="data/MAGFiLO_1.0_Kaggle_2026")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--output-dir", default="runs/unet_baseline")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patch-size", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--samples-per-image", type=int, default=2)
    p.add_argument("--base-channels", type=int, default=24)
    p.add_argument(
        "--architecture",
        choices=(
            "small_unet",
            "resnet34_unet",
            "resnet34_unetpp",
            "resnet34_unetpp_spine",
        ),
        default="small_unet",
    )
    p.add_argument("--spine-width", type=int, default=7)
    p.add_argument("--spine-loss-weight", type=float, default=0.35)
    p.add_argument("--embedding-loss-weight", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--min-area", type=int, default=None)
    p.add_argument("--merge-distance", type=int, default=10)
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--tile-overlap", type=int, default=256)
    p.add_argument("--infer-batch-size", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
