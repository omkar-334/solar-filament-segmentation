"""Modal jobs for training and generating the Kaggle submission."""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import modal


APP_NAME = "solar-filament-segmentation"
HF_REPO_ID = "omkar334/solar-filament-segmentation-2026-data"
ARCHIVE_NAME = "filament-segmentation-2026.zip"
DATASET_ROOT = Path("/data/raw/MAGFiLO_1.0_Kaggle_2026")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("solar-filament-data", create_if_missing=True)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("opencv-python-headless==4.12.0.88")
)
image = base_image.add_local_dir("src", remote_path="/root/src")
ijepa_image = (
    base_image.pip_install("transformers==4.57.1")
    .add_local_dir("src", remote_path="/root/src")
)
rfdetr_image = (
    base_image.pip_install("rfdetr[train,augment]==1.9.1")
    .add_local_dir("src", remote_path="/root/src")
)
hf_secret = modal.Secret.from_name("hf-read-secret")
lab_image = (
    base_image.pip_install("lightgbm==4.6.0", "scikit-learn==1.7.2")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("scripts", remote_path="/root/scripts")
)


def _get_hf_token() -> str:
    for name in (
        "HF_READ_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACE_API_TOKEN",
        "HUGGINGFACE_TOKEN",
    ):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(
        "The Modal secret hf-read-secret needs one of: HF_TOKEN, "
        "HUGGINGFACE_HUB_TOKEN, HUGGINGFACE_API_TOKEN, HUGGINGFACE_TOKEN"
    )


def _ensure_dataset() -> Path:
    from huggingface_hub import hf_hub_download

    if (DATASET_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").is_file():
        return DATASET_ROOT

    archive_path = Path("/data/downloads") / ARCHIVE_NAME
    if not archive_path.is_file():
        archive_path = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=ARCHIVE_NAME,
                token=_get_hf_token(),
                local_dir="/data/downloads",
            )
        )
    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        root = DATASET_ROOT.parent.resolve()
        for member in archive.infolist():
            destination = (DATASET_ROOT.parent / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"Unsafe path in dataset archive: {member.filename}")
        archive.extractall(DATASET_ROOT.parent)
    if not (DATASET_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").is_file():
        raise FileNotFoundError("The downloaded archive has an unexpected layout")
    volume.commit()
    return DATASET_ROOT


@app.function(
    image=image,
    gpu=["A10"],
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def train_job(
    epochs: int = 20,
    patch_size: int = 768,
    batch_size: int = 4,
    samples_per_image: int = 2,
    base_channels: int = 24,
    learning_rate: float = 3e-4,
    seed: int = 42,
    architecture: str = "small_unet",
    run_name: str = "unet_baseline",
    train_files_json: str = "",
    eval_every: int = 2,
):
    import torch

    sys.path.insert(0, "/root/src")
    from solar_filament.train import train

    root = _ensure_dataset()
    args = argparse.Namespace(
        data_root=str(root),
        cache_dir="/data/cache",
        output_dir=f"/data/runs/{run_name}",
        epochs=epochs,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=4,
        samples_per_image=samples_per_image,
        base_channels=base_channels,
        architecture=architecture,
        learning_rate=learning_rate,
        val_fraction=0.2,
        seed=seed,
        eval_every=eval_every,
        train_files_json=train_files_json,
        threshold=0.4 if architecture == "resnet34_unetpp_spine" else 0.6,
        min_area=64 if architecture == "resnet34_unetpp_spine" else 512,
        spine_width=7,
        spine_loss_weight=0.35,
        embedding_loss_weight=0.1,
        merge_distance=10,
        tile_size=1024,
        tile_overlap=256,
        infer_batch_size=batch_size,
    )
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" in gpu_name and args.batch_size > 2:
        args.batch_size = args.infer_batch_size = 2

    best = train(args, checkpoint_callback=volume.commit)
    volume.commit()
    print(f"Best checkpoint saved at {best}")
    print(f"Modal Volume: solar-filament-data; path: {args.output_dir}")


@app.function(
    image=ijepa_image,
    gpu=["A10"],
    cpu=8,
    memory=32768,
    timeout=8 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def ijepa_train_job(
    run_name: str = "ijepa_vith14_seg_seed42",
    epochs: int = 10,
    patch_size: int = 448,
    batch_size: int = 2,
    samples_per_image: int = 1,
    learning_rate: float = 2e-4,
    encoder_learning_rate: float = 1e-5,
):
    import torch

    os.environ["HF_HOME"] = "/data/hf-cache"
    sys.path.insert(0, "/root/src")
    from solar_filament.train import train

    root = _ensure_dataset()
    args = argparse.Namespace(
        data_root=str(root),
        cache_dir="/data/cache",
        output_dir=f"/data/runs/{run_name}",
        epochs=epochs,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=6,
        samples_per_image=samples_per_image,
        base_channels=24,
        architecture="ijepa_vith14_unet",
        learning_rate=learning_rate,
        encoder_learning_rate=encoder_learning_rate,
        val_fraction=0.2,
        seed=42,
        eval_every=epochs,
        threshold=0.4,
        min_area=200,
        spine_width=7,
        spine_loss_weight=0.35,
        embedding_loss_weight=0.1,
        merge_distance=10,
        tile_size=patch_size,
        tile_overlap=128,
        infer_batch_size=batch_size,
    )
    last_checkpoint = Path(args.output_dir) / "last.pt"
    args.resume_checkpoint = str(last_checkpoint) if last_checkpoint.is_file() else ""
    print(
        f"Starting I-JEPA ViT-H/14 segmentation on {torch.cuda.get_device_name(0)}; "
        f"tile={patch_size}, batch={batch_size}, epochs={epochs}, "
        f"samples_per_image={samples_per_image}, encoder_lr={encoder_learning_rate}",
        flush=True,
    )
    best = train(args, checkpoint_callback=volume.commit)
    volume.commit()
    print(f"Best I-JEPA segmentation checkpoint saved at {best}", flush=True)


@app.function(
    image=image,
    gpu=["L4", "A10", "T4"],
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def detector_instance_validation_job(
    detector_checkpoint: str,
    output_path: str,
    detector_image_size: int = 1024,
):
    """Score native YOLO instance masks on the grouped PQ validation split."""
    import json
    from collections import defaultdict

    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    from ultralytics import YOLO

    sys.path.insert(0, "/root/src")
    from solar_filament.data import grouped_split, polygon_mask, prepare_records
    from solar_filament.metrics import aggregate_pq, pq_counts

    root = _ensure_dataset()
    records = prepare_records(root, "/data/cache")
    _, val_records = grouped_split(records, val_fraction=0.2, seed=42)
    coco_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = json.loads(coco_path.read_text())
    images_by_id = {image["id"]: image for image in coco["images"]}
    anns_by_image: dict[str, list[dict]] = defaultdict(list)
    records_by_file: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        anns_by_image[annotation["image_id"]].append(annotation)
    for record in val_records:
        records_by_file[record["file_name"]].append(record)

    detector = YOLO(detector_checkpoint)
    settings = [
        (confidence, area)
        for confidence in (0.05, 0.1, 0.2, 0.35, 0.5)
        for area in (96, 200, 400)
    ]
    rows_by_setting = {setting: [] for setting in settings}
    for filename, file_records in tqdm(
        sorted(records_by_file.items()), desc="native instance PQ validation"
    ):
        image_path = root / "train" / "train_images" / filename
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        rgb = np.repeat(image[:, :, None], 3, axis=2)
        result = detector.predict(
            source=rgb,
            imgsz=detector_image_size,
            device=0,
            conf=0.01,
            iou=0.7,
            max_det=120,
            retina_masks=True,
            verbose=False,
        )[0]
        candidates = []
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy().astype(bool)
            scores = result.boxes.conf.detach().cpu().numpy()
            for mask, score in zip(masks, scores):
                if mask.shape != image.shape:
                    mask = np.asarray(
                        Image.fromarray(mask.astype(np.uint8)).resize(
                            (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
                        ),
                        dtype=np.uint8,
                    ).astype(bool)
                candidates.append((float(score), mask))
        candidates.sort(key=lambda item: item[0], reverse=True)

        for confidence, min_area in settings:
            labels = np.zeros(image.shape, dtype=np.uint16)
            claimed = np.zeros(image.shape, dtype=bool)
            instance_id = 1
            for score, mask in candidates:
                if score < confidence:
                    continue
                free = mask & ~claimed
                if int(free.sum()) < min_area:
                    continue
                labels[free] = instance_id
                claimed |= free
                instance_id += 1
            for record in file_records:
                image_meta = images_by_id[record["image_id"]]
                gt_masks = [
                    polygon_mask(
                        ann.get("segmentation"),
                        image_meta["height"],
                        image_meta["width"],
                    ).astype(bool)
                    for ann in anns_by_image.get(record["image_id"], [])
                ]
                rows_by_setting[(confidence, min_area)].append(
                    pq_counts(labels, gt_masks)
                )

    result_rows = []
    for (confidence, min_area), rows in rows_by_setting.items():
        result_rows.append(
            {
                "detection_threshold": confidence,
                "min_area": min_area,
                **aggregate_pq(rows),
            }
        )
    result_rows.sort(key=lambda row: row["pq_ratio_of_sums"], reverse=True)
    payload = {
        "detector_checkpoint": detector_checkpoint,
        "validation_physical_images": len(records_by_file),
        "validation_views": len(val_records),
        "settings": result_rows,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    volume.commit()
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved native instance validation: {destination}", flush=True)


@app.function(
    image=image,
    gpu=["L4", "A10", "T4"],
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def inference_job(
    checkpoint: str = "/data/runs/unet_baseline/best.pt",
    output_csv: str = "/data/runs/unet_baseline/submission.csv",
    threshold: float | None = None,
    min_area: int | None = None,
    merge_distance: int = 10,
    batch_size: int = 4,
):
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, "/root/src")
    from solar_filament.inference import encode_instance, load_model, predict_instances
    from solar_filament.rle import audit_submission

    root = _ensure_dataset()
    device = torch.device("cuda")
    if "T4" in torch.cuda.get_device_name(0) and batch_size > 2:
        batch_size = 2
    model, checkpoint_state = load_model(checkpoint, device)
    model_config = checkpoint_state.get("config", {})
    inference_tile_size = int(model_config.get("tile_size", 1024))
    inference_overlap = int(model_config.get("tile_overlap", 256))
    use_spine_instances = int(getattr(model, "output_channels", 1)) >= 10
    if threshold is None:
        threshold = 0.4 if use_spine_instances else 0.6
    if min_area is None:
        min_area = 64 if use_spine_instances else 512
    test_dir = root / "test" / "test_images"
    paths = sorted(test_dir.glob("*.jpeg"))
    if not paths:
        raise FileNotFoundError(f"No test JPEGs found in {test_dir}")
    import csv

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        for index, path in enumerate(paths, start=1):
            image_array = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            _, labels = predict_instances(
                model,
                image_array,
                device,
                threshold=threshold,
                min_area=min_area,
                tile_size=inference_tile_size,
                overlap=inference_overlap,
                batch_size=batch_size,
                merge_distance=merge_distance,
            )
            for instance_id in range(1, int(labels.max()) + 1):
                writer.writerow(
                    [f"{path.stem}_{instance_id}", encode_instance(labels == instance_id)]
                )
            if index % 10 == 0:
                print(f"Predicted {index}/{len(paths)} test images")
    image_shapes = {path.stem: (2048, 2048) for path in paths}
    row_count = audit_submission(
        output_path, image_shapes, expected_image_ids=image_shapes.keys()
    )
    print(f"Submission audit passed: {row_count} rows across {len(paths)} images")
    volume.commit()
    print(f"Submission saved at {output_path}")


@app.function(
    image=ijepa_image,
    gpu=["L4", "A10", "T4"],
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/data": volume},
)
def validation_sweep_job(
    checkpoint: str = "/data/runs/unet_baseline/best.pt",
    ensemble_checkpoint: str = "",
):
    os.environ["HF_HOME"] = "/data/hf-cache"
    import json

    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    sys.path.insert(0, "/root/src")
    from solar_filament.data import grouped_split, polygon_mask, prepare_records
    from solar_filament.inference import (
        load_model,
        predict_fields,
        predict_probability,
        spine_seeded_instances,
    )
    from solar_filament.metrics import (
        aggregate_pq,
        connected_components,
        filter_components,
        pq_counts,
    )

    root = _ensure_dataset()
    device = torch.device("cuda")
    model, checkpoint_state = load_model(checkpoint, device)
    model_config = checkpoint_state.get("config", {})
    inference_tile_size = int(model_config.get("tile_size", 1024))
    inference_overlap = int(model_config.get("tile_overlap", 256))
    use_spine_instances = int(getattr(model, "output_channels", 1)) >= 10
    records = prepare_records(root, "/data/cache")
    _, val_records = grouped_split(records, val_fraction=0.2, seed=42)
    coco_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = json.loads(coco_path.read_text())
    images_by_id = {image["id"]: image for image in coco["images"]}
    anns_by_image: dict[str, list[dict]] = {}
    for annotation in coco["annotations"]:
        anns_by_image.setdefault(annotation["image_id"], []).append(annotation)

    ensemble_model = None
    if ensemble_checkpoint:
        ensemble_model, _ = load_model(ensemble_checkpoint, device)
    default_infer_batch_size = 2 if "T4" in torch.cuda.get_device_name(0) else 4
    infer_batch_size = int(
        model_config.get("infer_batch_size", default_infer_batch_size)
    )
    sweep_thresholds = (0.4,) if use_spine_instances else (0.35, 0.4, 0.45)
    sweep_areas = (64, 256, 400)
    sweep_spine_thresholds = (0.35, 0.5, 0.65) if use_spine_instances else (None,)
    sweep_merge_distances = (0,) if use_spine_instances else (10,)
    probs_by_file: dict[str, np.ndarray] = {}
    seeded_labels_by_file: dict[str, dict[float, np.ndarray]] = {}
    for record in tqdm(val_records, desc="cache val probabilities"):
        filename = record["file_name"]
        if use_spine_instances and filename not in seeded_labels_by_file:
            image_array = np.asarray(
                Image.open(record["image_path"]).convert("L"), dtype=np.uint8
            )
            probability, spine_probability, embeddings = predict_fields(
                model,
                image_array,
                device,
                tile_size=inference_tile_size,
                overlap=inference_overlap,
                batch_size=infer_batch_size,
            )
            seeded_labels_by_file[filename] = {}
            for spine_threshold in sweep_spine_thresholds:
                seeded_labels_by_file[filename][spine_threshold] = spine_seeded_instances(
                    probability,
                    spine_probability,
                    embeddings,
                    threshold=sweep_thresholds[0],
                    min_area=1,
                    spine_threshold=spine_threshold,
                )
        elif not use_spine_instances and filename not in probs_by_file:
            image_array = np.asarray(
                Image.open(record["image_path"]).convert("L"), dtype=np.uint8
            )
            probability = predict_probability(
                model,
                image_array,
                device,
                tile_size=inference_tile_size,
                overlap=inference_overlap,
                batch_size=infer_batch_size,
            )
            if ensemble_model is not None:
                probability = 0.5 * (
                    probability
                    + predict_probability(
                        ensemble_model,
                        image_array,
                        device,
                        tile_size=inference_tile_size,
                        overlap=inference_overlap,
                        batch_size=infer_batch_size,
                    )
                )
            probs_by_file[filename] = probability

    if use_spine_instances:
        settings = [
            (threshold, spine_threshold, min_area, merge_distance)
            for threshold in sweep_thresholds
            for spine_threshold in sweep_spine_thresholds
            for min_area in sweep_areas
            for merge_distance in sweep_merge_distances
        ]
    else:
        settings = [
            (threshold, None, min_area, merge_distance)
            for threshold in sweep_thresholds
            for min_area in sweep_areas
            for merge_distance in sweep_merge_distances
        ]
    rows_by_setting = {setting: [] for setting in settings}
    for record in tqdm(val_records, desc="PQ threshold sweep"):
        image_meta = images_by_id[record["image_id"]]
        gt_masks = [
            polygon_mask(
                ann.get("segmentation"), image_meta["height"], image_meta["width"]
            ).astype(bool)
            for ann in anns_by_image.get(record["image_id"], [])
        ]
        prob = probs_by_file.get(record["file_name"])
        for threshold in sweep_thresholds:
            for spine_threshold in sweep_spine_thresholds:
                for merge_distance in sweep_merge_distances:
                    if use_spine_instances:
                        grouped_labels = seeded_labels_by_file[record["file_name"]][spine_threshold]
                    else:
                        grouped_labels = connected_components(
                            prob,
                            threshold=threshold,
                            min_area=1,
                            merge_distance=merge_distance,
                        )
                    for min_area in sweep_areas:
                        labels = filter_components(grouped_labels, min_area)
                        rows_by_setting[(
                            threshold,
                            spine_threshold,
                            min_area,
                            merge_distance,
                        )].append(pq_counts(labels, gt_masks))

    result = []
    for (threshold, spine_threshold, min_area, merge_distance), rows in rows_by_setting.items():
        result.append({
            "threshold": threshold,
            "spine_threshold": spine_threshold,
            "min_area": min_area,
            "merge_distance": merge_distance,
            **aggregate_pq(rows),
        })
    result.sort(key=lambda row: row["pq_ratio_of_sums"], reverse=True)
    output_dir = Path(checkpoint).parent
    if ensemble_checkpoint:
        output_dir = Path("/data/runs/ensembles") / (
            f"{Path(checkpoint).parent.name}__{Path(ensemble_checkpoint).parent.name}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "threshold_sweep.json"
    output.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(json.dumps(result, indent=2))
    print(f"Saved validation sweep: {output}")


@app.function(
    image=image,
    gpu=["A10", "L4", "T4"],
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def detector_train_job(
    run_name: str = "detector_fusion_yolo11s_seed42",
    detector_model: str = "yolo11s-seg",
    epochs: int = 60,
    image_size: int = 1024,
    batch_size: int = 8,
    resume_checkpoint: str = "",
    full_data: bool = False,
    seed: int = 42,
    lr0: float = 0.01,
):
    """Train the instance-proposal model on annotator-grouped YOLO masks.

    ``full_data`` trains on all 707 physical images. The seed-42 validation views
    are still passed to YOLO for its internal metrics, so they are contaminated
    and ``last.pt`` (not ``best.pt``) must be used for the final model.
    """
    import json
    import torch
    from ultralytics import YOLO

    sys.path.insert(0, "/root/src")
    from solar_filament.data import grouped_split, prepare_records
    from solar_filament.detector_fusion import export_yolo_segmentation_dataset

    root = _ensure_dataset()
    records = prepare_records(root, "/data/cache")
    train_records, val_records = grouped_split(records, val_fraction=0.2, seed=42)
    if full_data:
        train_records = list(records)
    annotation_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    image_root = root / "train" / "train_images"
    dataset_root = Path("/data/yolo/filament_seg_full" if full_data else "/data/yolo/filament_seg_seed42")
    yaml_path = export_yolo_segmentation_dataset(
        annotation_path,
        image_root,
        records,
        train_records,
        val_records,
        dataset_root,
    )
    output_dir = Path("/data/runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": detector_model,
        "pretrained": f"{detector_model}.pt",
        "epochs": epochs,
        "built_in_validation_interval": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "seed": 42,
        "train_views": len(train_records),
        "train_physical_images": len({r["file_name"] for r in train_records}),
        "validation_views": len(val_records),
        "validation_physical_images": len({r["file_name"] for r in val_records}),
        "dataset_yaml": str(yaml_path),
        "augmentations": {
            "mosaic": 0,
            "mixup": 0,
            "flips": True,
            "rotation_degrees": 180,
            "brightness": 0.1,
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" in gpu_name:
        batch_size = min(batch_size, 4)
    print(
        f"Starting {config['architecture']} on {gpu_name}; "
        f"train={config['train_views']} views/{config['train_physical_images']} images; "
        f"val={config['validation_views']} views/{config['validation_physical_images']} images"
    )
    if resume_checkpoint:
        print(f"Resuming from saved detector checkpoint: {resume_checkpoint}")
        model = YOLO(resume_checkpoint)
        model.add_callback("on_model_save", lambda _trainer: volume.commit())
        result = model.train(resume=True)
    else:
        model = YOLO(f"{detector_model}.pt")
        model.add_callback("on_train_epoch_end", _nan_guard)
        # YOLO writes its latest checkpoint every epoch. Commit those writes to
        # the shared Modal Volume so a worker restart does not discard progress.
        model.add_callback("on_model_save", lambda _trainer: volume.commit())
        result = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            workers=4,
            device=0,
            project=str(output_dir),
            name="train",
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=lr0,
            cos_lr=True,
            patience=0 if full_data else 15,
            close_mosaic=0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            degrees=180.0,
            translate=0.05,
            scale=0.2,
            fliplr=0.5,
            flipud=0.5,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.1,
            amp=True,
            cache=False,
            plots=False,
            save=True,
            save_period=10,
            seed=seed,
            deterministic=False,
            verbose=True,
        )
    volume.commit()
    save_dir = Path(result.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    print(f"Best YOLO checkpoint: {best}")
    print(f"Last YOLO checkpoint: {last}")
    print(f"Modal Volume: solar-filament-data; path: {output_dir}")


@app.function(
    image=image,
    gpu=["A10"],
    cpu=8,
    memory=32768,
    timeout=8 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def oof_detector_proposal_job(
    run_name: str = "oof_yolo11s_seed42_3fold",
    folds: int = 3,
    epochs: int = 60,
    image_size: int = 1024,
    batch_size: int = 8,
):
    """Train grouped detector folds and save proposals only for unseen images."""
    import csv
    import json
    import random

    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO

    sys.path.insert(0, "/root/src")
    from solar_filament.data import grouped_split, prepare_records
    from solar_filament.detector_fusion import export_yolo_segmentation_dataset

    root = _ensure_dataset()
    records = prepare_records(root, "/data/cache")
    train_records, heldout_records = grouped_split(records, val_fraction=0.2, seed=42)
    train_files = sorted({record["file_name"] for record in train_records})
    if folds < 2 or folds > len(train_files):
        raise ValueError(f"Invalid grouped fold count: {folds}")
    rng = random.Random(42)
    rng.shuffle(train_files)
    fold_files = [set(train_files[index::folds]) for index in range(folds)]
    output_dir = Path("/data/runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": "yolo11s-seg",
        "seed": 42,
        "folds": folds,
        "epochs": epochs,
        "image_size": image_size,
        "train_physical_images": len(train_files),
        "heldout_physical_images": len({r["file_name"] for r in heldout_records}),
        "fold_sizes": [len(files) for files in fold_files],
        "split_rule": "all annotator views of each physical image stay in one fold",
        "proposal_confidence": 0.05,
        "proposal_max_det": 80,
    }
    (output_dir / "run_config.json").write_text(json.dumps(manifest, indent=2))
    volume.commit()

    annotation_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    image_root = root / "train" / "train_images"
    fold_outputs = []
    for fold_index, validation_files in enumerate(fold_files):
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        proposal_path = fold_dir / "oof_proposals.json"
        if proposal_path.is_file():
            fold_outputs.append(json.loads(proposal_path.read_text()))
            print(f"Fold {fold_index + 1}/{folds}: reusing saved OOF proposals", flush=True)
            continue

        fold_train = [r for r in train_records if r["file_name"] not in validation_files]
        fold_validation = [r for r in train_records if r["file_name"] in validation_files]
        dataset_root = Path("/data/yolo") / run_name / f"fold_{fold_index}"
        yaml_path = export_yolo_segmentation_dataset(
            annotation_path,
            image_root,
            records,
            fold_train,
            fold_validation,
            dataset_root,
        )
        train_dir = fold_dir / "train"
        results_csv = train_dir / "results.csv"
        last_checkpoint = train_dir / "weights" / "last.pt"
        completed_epochs = 0
        if results_csv.is_file():
            with results_csv.open(newline="") as handle:
                completed_epochs = sum(1 for _ in csv.DictReader(handle))

        if completed_epochs < epochs or not last_checkpoint.is_file():
            if last_checkpoint.is_file():
                print(
                    f"Fold {fold_index + 1}/{folds}: resuming at epoch "
                    f"{completed_epochs + 1}/{epochs}",
                    flush=True,
                )
                model = YOLO(str(last_checkpoint))
                model.add_callback("on_model_save", lambda _trainer: volume.commit())
                model.train(resume=True)
            else:
                print(
                    f"Fold {fold_index + 1}/{folds}: training on "
                    f"{len({r['file_name'] for r in fold_train})} physical images; "
                    f"predicting {len(validation_files)} unseen images",
                    flush=True,
                )
                model = YOLO("yolo11s-seg.pt")
                model.add_callback("on_model_save", lambda _trainer: volume.commit())
                model.train(
                    data=str(yaml_path),
                    epochs=epochs,
                    imgsz=image_size,
                    batch=batch_size,
                    workers=4,
                    device=0,
                    project=str(fold_dir),
                    name="train",
                    exist_ok=True,
                    pretrained=True,
                    optimizer="AdamW",
                    cos_lr=True,
                    patience=0,
                    val=False,
                    mosaic=0.0,
                    mixup=0.0,
                    copy_paste=0.0,
                    degrees=180.0,
                    translate=0.05,
                    scale=0.2,
                    fliplr=0.5,
                    flipud=0.5,
                    hsv_h=0.0,
                    hsv_s=0.0,
                    hsv_v=0.1,
                    amp=True,
                    cache=False,
                    plots=False,
                    save=True,
                    save_period=10,
                    seed=42 + fold_index,
                    deterministic=False,
                    verbose=True,
                )
            volume.commit()

        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Missing fold checkpoint: {last_checkpoint}")
        detector = YOLO(str(last_checkpoint))
        predictions = {}
        for filename in sorted(validation_files):
            gray = np.asarray(
                Image.open(image_root / filename).convert("L"), dtype=np.uint8
            )
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            result = detector.predict(
                source=rgb,
                imgsz=image_size,
                device=0,
                conf=0.05,
                iou=0.7,
                max_det=80,
                verbose=False,
            )[0]
            proposals = []
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                proposals = [
                    {"bbox": [float(v) for v in box], "score": float(score)}
                    for box, score in zip(boxes, scores)
                ]
            predictions[filename] = proposals

        fold_result = {
            "fold": fold_index,
            "heldout_files": sorted(validation_files),
            "predictions": predictions,
        }
        temporary_path = proposal_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(fold_result, separators=(",", ":")))
        os.replace(temporary_path, proposal_path)
        volume.commit()
        fold_outputs.append(fold_result)
        print(
            f"Fold {fold_index + 1}/{folds}: saved OOF proposals for "
            f"{len(predictions)} unseen images",
            flush=True,
        )

    merged = {}
    for fold_result in fold_outputs:
        for filename, proposals in fold_result["predictions"].items():
            if filename in merged:
                raise RuntimeError(f"Image received OOF predictions twice: {filename}")
            merged[filename] = proposals
    if set(merged) != set(train_files):
        missing = sorted(set(train_files) - set(merged))
        raise RuntimeError(f"OOF proposals missing {len(missing)} training images")
    merged_path = output_dir / "oof_proposals.json"
    merged_path.write_text(
        json.dumps(
            {
                **manifest,
                "predictions": merged,
            },
            separators=(",", ":"),
        )
    )
    volume.commit()
    print(
        f"Saved grouped OOF proposals for {len(merged)} training images: {merged_path}",
        flush=True,
    )


@app.function(
    image=image,
    gpu=["L4", "A10", "T4"],
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def detector_fusion_validation_job(
    detector_checkpoint: str = "/data/runs/detector_fusion_yolo11s_seed42/train/weights/best.pt",
    segmentation_checkpoint: str = "/data/runs/resnet34_unetpp_seed42_detached/best.pt",
    output_path: str = "/data/runs/detector_fusion_yolo11s_seed42/fusion_validation.json",
    detector_image_size: int = 1024,
):
    """Evaluate detector proposals fused with the best foreground U-Net."""
    import json
    from collections import defaultdict

    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from ultralytics import YOLO

    sys.path.insert(0, "/root/src")
    from solar_filament.data import grouped_split, polygon_mask, prepare_records
    from solar_filament.detector_fusion import (
        detector_gated_instances,
        solar_disk_masks,
    )
    from solar_filament.inference import load_model, predict_probability
    from solar_filament.metrics import aggregate_pq, connected_components, pq_counts

    root = _ensure_dataset()
    records = prepare_records(root, "/data/cache")
    _, val_records = grouped_split(records, val_fraction=0.2, seed=42)
    coco_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = json.loads(coco_path.read_text())
    images_by_id = {image["id"]: image for image in coco["images"]}
    anns_by_image: dict[str, list[dict]] = defaultdict(list)
    records_by_file: dict[str, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    for record in val_records:
        records_by_file[record["file_name"]].append(record)

    device = torch.device("cuda")
    detector = YOLO(detector_checkpoint)
    segmentation_model, _ = load_model(segmentation_checkpoint, device)
    base_settings = [
        (foreground_threshold, detection_threshold, grow_radius, min_area, None)
        for foreground_threshold, detection_threshold, grow_radius, min_area in (
            (0.4, 0.15, 4, 200),
            (0.4, 0.3, 4, 200),
            (0.5, 0.3, 4, 200),
            (0.4, 0.3, 0, 200),
            (0.4, 0.3, 4, 400),
        )
    ]
    disk_settings = [
        (0.5, 0.3, 4, 200, margin)
        for margin in (0, 8, 16, 24)
    ]
    settings = base_settings + disk_settings
    rows_by_setting = {setting: [] for setting in settings}
    baseline_rows = []
    batch_size = 2 if "T4" in torch.cuda.get_device_name(0) else 4

    for filename, file_records in tqdm(
        sorted(records_by_file.items()), desc="detector/U-Net PQ sweep"
    ):
        image_path = root / "train" / "train_images" / filename
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        probability = predict_probability(
            segmentation_model,
            image,
            device,
            tile_size=1024,
            overlap=256,
            batch_size=batch_size,
        )
        rgb_image = np.repeat(image[:, :, None], 3, axis=2)
        result = detector.predict(
            source=rgb_image,
            imgsz=detector_image_size,
            device=0,
            conf=0.01,
            iou=0.7,
            retina_masks=True,
            verbose=False,
        )[0]
        detections = []
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy().astype(bool)
            scores = result.boxes.conf.detach().cpu().numpy()
            for mask, score in zip(masks, scores):
                if mask.shape != image.shape:
                    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
                    )
                    mask = np.asarray(resized, dtype=np.uint8) > 0
                detections.append((mask, float(score)))

        baseline_labels = connected_components(
            probability,
            threshold=0.4,
            min_area=400,
            merge_distance=10,
        )

        disk_regions = solar_disk_masks(
            image,
            {setting[4] for setting in settings if setting[4] is not None},
        )
        for setting in settings:
            foreground_threshold, detection_threshold, grow_radius, min_area, disk_margin = setting
            labels = detector_gated_instances(
                probability,
                detections,
                foreground_threshold=foreground_threshold,
                detection_threshold=detection_threshold,
                grow_radius=grow_radius,
                min_area=min_area,
                valid_region=disk_regions.get(disk_margin),
            )
            for record in file_records:
                image_meta = images_by_id[record["image_id"]]
                gt_masks = [
                    polygon_mask(
                        ann.get("segmentation"),
                        image_meta["height"],
                        image_meta["width"],
                    ).astype(bool)
                    for ann in anns_by_image.get(record["image_id"], [])
                ]
                rows_by_setting[setting].append(pq_counts(labels, gt_masks))
        for record in file_records:
            image_meta = images_by_id[record["image_id"]]
            gt_masks = [
                polygon_mask(
                    ann.get("segmentation"),
                    image_meta["height"],
                    image_meta["width"],
                ).astype(bool)
                for ann in anns_by_image.get(record["image_id"], [])
            ]
            baseline_rows.append(pq_counts(baseline_labels, gt_masks))

    result_rows = []
    for setting, rows in rows_by_setting.items():
        foreground_threshold, detection_threshold, grow_radius, min_area, disk_margin = setting
        result_rows.append(
            {
                "foreground_threshold": foreground_threshold,
                "detection_threshold": detection_threshold,
                "grow_radius": grow_radius,
                "min_area": min_area,
                "disk_margin_px": disk_margin,
                **aggregate_pq(rows),
            }
        )
    result_rows.append(
        {
            "pipeline": "unet_connected_components_baseline",
            "foreground_threshold": 0.4,
            "min_area": 400,
            "merge_distance": 10,
            **aggregate_pq(baseline_rows),
        }
    )
    result_rows.sort(key=lambda row: row["pq_ratio_of_sums"], reverse=True)
    payload = {
        "detector_checkpoint": detector_checkpoint,
        "segmentation_checkpoint": segmentation_checkpoint,
        "validation_physical_images": len(records_by_file),
        "validation_views": len(val_records),
        "settings": result_rows,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    volume.commit()
    print(json.dumps(payload, indent=2))
    print(f"Saved detector fusion validation: {destination}")


@app.function(
    image=image,
    gpu=["L4", "A10", "T4"],
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def detector_submission_job(
    detector_checkpoint: str = "/data/runs/detector_fusion_yolo11s_seed42/train/weights/best.pt",
    segmentation_checkpoint: str = "/data/runs/resnet34_unetpp_seed42_detached/best.pt",
    output_csv: str = "/data/runs/detector_fusion_yolo11s_seed42/submission.csv",
    foreground_threshold: float = 0.5,
    detection_threshold: float = 0.3,
    grow_radius: int = 4,
    min_area: int = 200,
    detector_image_size: int = 1024,
    disk_margin_px: int | None = None,
):
    """Create and audit test predictions; this job never submits to Kaggle."""
    import csv
    import json

    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from ultralytics import YOLO

    sys.path.insert(0, "/root/src")
    from solar_filament.detector_fusion import (
        detector_gated_instances,
        solar_disk_mask,
    )
    from solar_filament.inference import (
        encode_instance,
        load_model,
        predict_probability,
    )
    from solar_filament.rle import audit_submission

    root = _ensure_dataset()
    test_dir = root / "test" / "test_images"
    paths = sorted(test_dir.glob("*.jpeg"))
    if not paths:
        raise FileNotFoundError(f"No test JPEGs found in {test_dir}")
    device = torch.device("cuda")
    detector = YOLO(detector_checkpoint)
    segmentation_model, _ = load_model(segmentation_checkpoint, device)
    batch_size = 2 if "T4" in torch.cuda.get_device_name(0) else 4
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    instance_counts = {}

    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["filament_id", "segmentation_rle"])
        for index, path in enumerate(tqdm(paths, desc="test detector/U-Net fusion"), start=1):
            image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            probability = predict_probability(
                segmentation_model,
                image,
                device,
                tile_size=1024,
                overlap=256,
                batch_size=batch_size,
            )
            rgb_image = np.repeat(image[:, :, None], 3, axis=2)
            prediction = detector.predict(
                source=rgb_image,
                imgsz=detector_image_size,
                device=0,
                conf=0.01,
                iou=0.7,
                retina_masks=True,
                verbose=False,
            )[0]
            detections = []
            if prediction.masks is not None and prediction.boxes is not None:
                masks = prediction.masks.data.detach().cpu().numpy().astype(bool)
                scores = prediction.boxes.conf.detach().cpu().numpy()
                for mask, score in zip(masks, scores):
                    if mask.shape != image.shape:
                        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
                            (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
                        )
                        mask = np.asarray(resized, dtype=np.uint8) > 0
                    detections.append((mask, float(score)))
            labels = detector_gated_instances(
                probability,
                detections,
                foreground_threshold=foreground_threshold,
                detection_threshold=detection_threshold,
                grow_radius=grow_radius,
                min_area=min_area,
                valid_region=(
                    solar_disk_mask(image, disk_margin_px)
                    if disk_margin_px is not None
                    else None
                ),
            )
            count = int(labels.max())
            instance_counts[path.stem] = count
            for instance_id in range(1, count + 1):
                writer.writerow(
                    [f"{path.stem}_{instance_id}", encode_instance(labels == instance_id)]
                )
            if index % 20 == 0:
                print(f"Predicted {index}/{len(paths)} test images")

    image_shapes = {path.stem: (2048, 2048) for path in paths}
    row_count = audit_submission(
        output_path,
        image_shapes,
        expected_image_ids=image_shapes.keys(),
    )
    config = {
        "detector_checkpoint": detector_checkpoint,
        "segmentation_checkpoint": segmentation_checkpoint,
        "foreground_threshold": foreground_threshold,
        "detection_threshold": detection_threshold,
        "grow_radius": grow_radius,
        "min_area": min_area,
        "detector_image_size": detector_image_size,
        "disk_margin_px": disk_margin_px,
        "test_images": len(paths),
        "submission_rows": row_count,
        "instances_per_image": instance_counts,
        "rle_audit": "passed",
    }
    config_path = output_path.with_name("submission_config.json")
    config_path.write_text(json.dumps(config, indent=2))
    volume.commit()
    print(f"Submission audit passed: {row_count} rows across {len(paths)} test images")
    print(f"Candidate CSV saved at {output_path}")
    print(f"Candidate config saved at {config_path}")


@app.function(
    image=image,
    gpu=["L4", "A10", "T4"],
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def validation_diagnostics_job(
    checkpoint: str = "/data/runs/resnet34_unet_seed42_full/best.pt",
    output_zip: str = "/data/runs/resnet34_unet_seed42_full/resnet_validation_diagnostics.zip",
    threshold: float = 0.5,
    min_area: int = 400,
    merge_distance: int = 10,
):
    """Package validation thumbnails, confusion overlays, and instance errors."""
    import hashlib
    import json
    import shutil
    import tempfile
    from collections import defaultdict

    import numpy as np
    import torch
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    from tqdm import tqdm

    sys.path.insert(0, "/root/src")
    from solar_filament.data import grouped_split, polygon_mask, prepare_records
    from solar_filament.inference import load_model, predict_instances

    root = _ensure_dataset()
    records = prepare_records(root, "/data/cache")
    _, val_records = grouped_split(records, val_fraction=0.2, seed=42)
    coco_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = json.loads(coco_path.read_text())
    images_by_id = {image["id"]: image for image in coco["images"]}
    anns_by_image: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        anns_by_image[annotation["image_id"]].append(annotation)
    records_by_file: dict[str, list[dict]] = defaultdict(list)
    for record in val_records:
        records_by_file[record["file_name"]].append(record)

    device = torch.device("cuda")
    model, _ = load_model(checkpoint, device)
    batch_size = 2 if "T4" in torch.cuda.get_device_name(0) else 4
    thumb_size = 1024
    temp_root = Path(tempfile.mkdtemp(prefix="solar-validation-diagnostics-"))
    image_dir = temp_root / "assets" / "images"
    overlay_dir = temp_root / "assets" / "overlays"
    image_dir.mkdir(parents=True)
    overlay_dir.mkdir(parents=True)

    colors = {
        "tp": (45, 220, 150, 175),
        "fp": (255, 105, 87, 195),
        "fn": (180, 125, 255, 195),
        "tn": (72, 165, 220, 55),
        "pred": (70, 205, 255, 175),
        "gt": (255, 195, 72, 175),
    }

    def small_mask(mask: np.ndarray) -> np.ndarray:
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        resized = mask_image.resize(
            (thumb_size, thumb_size), resample=Image.Resampling.BOX
        )
        return np.asarray(resized, dtype=np.uint8) > 0

    def write_overlay(path: Path, mask: np.ndarray, color: tuple[int, int, int, int]):
        rgba = np.zeros((thumb_size, thumb_size, 4), dtype=np.uint8)
        rgba[mask] = color
        Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)

    def match_instances(pred_labels: np.ndarray, gt_masks: list[np.ndarray]):
        pred_ids = np.unique(pred_labels)
        pred_ids = pred_ids[pred_ids != 0]
        n_pred, n_gt = len(pred_ids), len(gt_masks)
        ious = np.zeros((n_pred, n_gt), dtype=np.float64)
        pred_areas = {
            int(pid): int(np.count_nonzero(pred_labels == pid)) for pid in pred_ids
        }
        gt_areas = [int(mask.sum()) for mask in gt_masks]
        intersections = np.zeros((n_pred, n_gt), dtype=np.int64)
        for j, mask in enumerate(gt_masks):
            if n_pred:
                intersections[:, j] = np.bincount(
                    pred_labels[mask], minlength=int(pred_labels.max()) + 1
                )[pred_ids]
                unions = np.asarray([pred_areas[int(pid)] for pid in pred_ids]) + gt_areas[j] - intersections[:, j]
                ious[:, j] = intersections[:, j] / np.maximum(unions, 1)
        if n_pred and n_gt:
            rows, cols = linear_sum_assignment(ious, maximize=True)
            matches = [
                (int(r), int(c), float(ious[r, c]))
                for r, c in zip(rows, cols)
                if ious[r, c] > 0.5
            ]
        else:
            matches = []
        matched_pred = {r for r, _, _ in matches}
        matched_gt = {c for _, c, _ in matches}
        fp_rows = [i for i in range(n_pred) if i not in matched_pred]
        fn_cols = [j for j in range(n_gt) if j not in matched_gt]
        tp_rows = [
            {"iou": iou, "pred_area": pred_areas[int(pred_ids[r])], "gt_area": gt_areas[c]}
            for r, c, iou in matches
        ]
        return pred_ids, ious, tp_rows, fp_rows, fn_cols

    manifest_views = []
    manifest_images = {}
    ordered_files = sorted(records_by_file.items())
    for filename, file_records in tqdm(ordered_files, desc="diagnostic images"):
        file_key = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]
        image_array = np.asarray(
            Image.open(root / "train" / "train_images" / filename).convert("L"),
            dtype=np.uint8,
        )
        image_asset = f"assets/images/{file_key}.jpg"
        Image.fromarray(image_array).resize(
            (thumb_size, thumb_size), resample=Image.Resampling.LANCZOS
        ).save(temp_root / image_asset, quality=84, optimize=True)
        manifest_images[filename] = image_asset

        probability, pred_labels = predict_instances(
            model,
            image_array,
            device,
            threshold=threshold,
            min_area=min_area,
            batch_size=batch_size,
            merge_distance=merge_distance,
        )
        pred_foreground = pred_labels > 0
        for record in file_records:
            image_id = record["image_id"]
            image_meta = images_by_id[image_id]
            gt_masks = [
                polygon_mask(
                    annotation.get("segmentation"),
                    image_meta["height"],
                    image_meta["width"],
                ).astype(bool)
                for annotation in anns_by_image.get(image_id, [])
            ]
            gt_union = np.zeros_like(pred_foreground)
            for mask in gt_masks:
                gt_union |= mask

            tp_pixels = pred_foreground & gt_union
            fp_pixels = pred_foreground & ~gt_union
            fn_pixels = ~pred_foreground & gt_union
            tn_pixels = ~pred_foreground & ~gt_union
            pred_ids, ious, tp_rows, fp_rows, fn_cols = match_instances(
                pred_labels, gt_masks
            )
            view_key = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:16]
            overlay_paths = {}
            for name, mask in (
                ("tp", tp_pixels),
                ("fp", fp_pixels),
                ("fn", fn_pixels),
                ("tn", tn_pixels),
                ("pred", pred_foreground),
                ("gt", gt_union),
            ):
                asset = f"assets/overlays/{view_key}_{name}.png"
                write_overlay(temp_root / asset, small_mask(mask), colors[name])
                overlay_paths[name] = asset

            prob_small = Image.fromarray(probability.astype(np.float32), mode="F").resize(
                (thumb_size, thumb_size), resample=Image.Resampling.BILINEAR
            )
            prob_array = np.clip(np.asarray(prob_small) * 255, 0, 255).astype(np.uint8)
            heat_rgba = np.zeros((thumb_size, thumb_size, 4), dtype=np.uint8)
            heat_rgba[:, :, 0] = np.clip(22 + prob_array * 0.91, 0, 255).astype(np.uint8)
            heat_rgba[:, :, 1] = np.clip(35 + prob_array * 0.68, 0, 255).astype(np.uint8)
            heat_rgba[:, :, 2] = np.clip(95 - prob_array * 0.28, 0, 255).astype(np.uint8)
            heat_rgba[:, :, 3] = np.clip(35 + prob_array * 0.72, 0, 190).astype(np.uint8)
            probability_asset = f"assets/overlays/{view_key}_probability.png"
            Image.fromarray(heat_rgba, mode="RGBA").save(
                temp_root / probability_asset, optimize=True
            )

            fp_instances = []
            for i in fp_rows:
                pid = int(pred_ids[i])
                instance = pred_labels == pid
                fp_instances.append({
                    "area": int(instance.sum()),
                    "mean_probability": float(probability[instance].mean()),
                    "max_probability": float(probability[instance].max()),
                    "best_gt_iou": float(ious[i].max()) if ious.shape[1] else 0.0,
                })
            fn_instances = []
            for j in fn_cols:
                mask = gt_masks[j]
                fn_instances.append({
                    "area": int(mask.sum()),
                    "mean_probability": float(probability[mask].mean()) if mask.any() else 0.0,
                    "max_probability": float(probability[mask].max()) if mask.any() else 0.0,
                    "best_pred_iou": float(ious[:, j].max()) if ious.shape[0] else 0.0,
                })

            pixel_counts = {
                "tp": int(tp_pixels.sum()),
                "fp": int(fp_pixels.sum()),
                "fn": int(fn_pixels.sum()),
                "tn": int(tn_pixels.sum()),
            }
            matched_iou_sum = float(sum(row["iou"] for row in tp_rows))
            denom = len(tp_rows) + 0.5 * (len(fp_rows) + len(fn_cols))
            manifest_views.append({
                "image_id": image_id,
                "filename": filename,
                "image": image_asset,
                "overlays": overlay_paths | {"probability": probability_asset},
                "gt_instances": len(gt_masks),
                "pred_instances": int(len(pred_ids)),
                "instance_tp": len(tp_rows),
                "instance_fp": len(fp_rows),
                "instance_fn": len(fn_cols),
                "sum_iou": matched_iou_sum,
                "pq": matched_iou_sum / denom if denom else 0.0,
                "pixel_counts": pixel_counts,
                "fp_instances": fp_instances,
                "fn_instances": fn_instances,
                "tp_instances": tp_rows,
            })

    manifest = {
        "title": "ResNet34 validation error review",
        "checkpoint": str(checkpoint),
        "threshold": threshold,
        "min_area": min_area,
        "merge_distance": merge_distance,
        "iou_threshold": 0.5,
        "thumbnail_size": thumb_size,
        "views": manifest_views,
        "images": manifest_images,
    }
    (temp_root / "manifest.json").write_text(json.dumps(manifest))
    archive_path = Path(output_zip)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in temp_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(temp_root))
    volume.commit()
    print(f"Saved diagnostic bundle: {archive_path}; views={len(manifest_views)}")
    shutil.rmtree(temp_root)


@app.function(
    image=image,
    gpu=["A10", "L4", "L40S", "A100", "T4"],
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def crop_refiner_train_job(
    run_name: str = "seed_refiner_resnet18_seed42",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    seed: int = 42,
    oof_predictions_path: str = "",
    init_checkpoint: str = "",
    full_data: bool = False,
    soft_consensus: bool = False,
    train_files_json: str = "",
    encoder: str = "resnet18",
    crop_size: int = 256,
):
    """Train/fine-tune a seed-conditioned crop refiner (``full_data``: all 707 images).

    ``train_files_json`` (a JSON list of filenames) overrides the split, for k-fold.
    """
    import json
    import random

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    sys.path.insert(0, "/root/src")
    from solar_filament.crop_refiner import (
        CropUNet,
        SeedCropDataset,
        build_oof_proposal_crop_samples,
        build_medoid_crop_samples,
        crop_refiner_loss,
    )
    from solar_filament.data import grouped_split, prepare_records
    from solar_filament import crop_refiner as _cr

    _cr.set_crop_size(crop_size)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    root = _ensure_dataset()
    records = prepare_records(root, "/data/cache")
    train_records, val_records = grouped_split(records, val_fraction=0.2, seed=seed)
    if full_data:
        train_records, val_records = list(records), []
    if train_files_json:
        keep = set(json.loads(Path(train_files_json).read_text()))
        train_records = [r for r in records if r["file_name"] in keep]
        val_records = [r for r in records if r["file_name"] not in keep]
    coco_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = json.loads(coco_path.read_text())
    output_dir = Path("/data/runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    use_oof = bool(oof_predictions_path)
    samples_path = output_dir / (
        "oof_proposal_crop_samples.json" if use_oof else "medoid_crop_samples.json"
    )
    if samples_path.is_file():
        samples = json.loads(samples_path.read_text())
    else:
        if use_oof:
            proposal_payload = json.loads(Path(oof_predictions_path).read_text())
            proposals_by_file = proposal_payload["predictions"]
            expected_files = {row["file_name"] for row in train_records}
            proposals_by_file = {k: v for k, v in proposals_by_file.items() if k in expected_files}
            if set(proposals_by_file) != expected_files:
                raise ValueError(
                    "OOF proposals must cover exactly the grouped training images; "
                    f"expected {len(expected_files)}, got {len(proposals_by_file)}"
                )
            samples = build_oof_proposal_crop_samples(
                coco,
                root / "train" / "train_images",
                train_records,
                proposals_by_file,
                seed=seed,
                soft_consensus=soft_consensus,
            )
        else:
            samples = build_medoid_crop_samples(
                coco,
                root / "train" / "train_images",
                train_records,
                seed=seed,
            )
        samples_path.write_text(json.dumps(samples))
        volume.commit()
    positive_count = sum(bool(sample["polygons"]) for sample in samples)
    print(
        f"Crop refiner data: {len(samples)} crops, {positive_count} positive, "
        f"{len(samples) - positive_count} hard negative; "
        f"train={len({row['file_name'] for row in train_records})} images, "
        f"val={len({row['file_name'] for row in val_records})} images",
        flush=True,
    )

    dataset = SeedCropDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    model = CropUNet(pretrained=not bool(init_checkpoint), encoder_name=encoder).cuda()
    if init_checkpoint:
        initial_state = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initial_state.get("model", initial_state), strict=True)
        print(f"Initialized crop refiner from {init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    start_epoch, best_loss = 0, float("inf")
    if last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        best_loss = float(state.get("best_loss", best_loss))
        print(f"Resuming crop refiner at epoch {start_epoch + 1}/{epochs}", flush=True)

    metrics_path = output_dir / "training_metrics.jsonl"
    for epoch in range(start_epoch, epochs):
        model.train()
        losses = []
        for inputs, targets in tqdm(loader, desc=f"refiner epoch {epoch + 1}/{epochs}"):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(inputs)
                loss = crop_refiner_loss(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        mean_loss = float(np.mean(losses)) if losses else float("inf")
        is_best = mean_loss < best_loss
        best_loss = min(best_loss, mean_loss)
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
            "seed": seed,
            "oof_predictions_path": oof_predictions_path,
            "init_checkpoint": init_checkpoint,
            "encoder": encoder,
            "crop_size": crop_size,
            "train_physical_images": len({row["file_name"] for row in train_records}),
            "validation_physical_images": len({row["file_name"] for row in val_records}),
        }
        torch.save(checkpoint, last_path)
        if is_best:
            torch.save(checkpoint, best_path)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"epoch": epoch + 1, "train_loss": mean_loss, "lr": scheduler.get_last_lr()[0]}
                )
                + "\n"
            )
        volume.commit()
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": mean_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "best_loss": best_loss,
                }
            ),
            flush=True,
        )
    print(f"Crop refiner checkpoint: {best_path}", flush=True)


@app.function(
    image=image,
    gpu=["L4", "A10", "T4"],
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def crop_refiner_validation_job(
    refiner_checkpoint: str = "/data/runs/seed_refiner_resnet18_seed42/best.pt",
    detector_checkpoint: str = "/data/runs/detector_fusion_yolo11s_seed42/train/weights/best.pt",
    detector_checkpoints: str = "",
    ensemble_nms_iou: float = 0.5,
    output_path: str = "/data/runs/seed_refiner_resnet18_seed42/validation.json",
    detector_image_size: int = 1024,
):
    """Validate seed-conditioned crop refinement against grouped held-out views."""
    import json
    from collections import defaultdict

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from ultralytics import YOLO

    sys.path.insert(0, "/root/src")
    from solar_filament.crop_refiner import CropUNet, make_inference_crop
    from solar_filament.data import grouped_split, polygon_mask, prepare_records
    from solar_filament.metrics import aggregate_pq, pq_counts

    root = _ensure_dataset()
    records = prepare_records(root, "/data/cache")
    _, val_records = grouped_split(records, val_fraction=0.2, seed=42)
    coco_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = json.loads(coco_path.read_text())
    images_by_id = {image["id"]: image for image in coco["images"]}
    anns_by_image: dict[str, list[dict]] = defaultdict(list)
    records_by_file: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        anns_by_image[annotation["image_id"]].append(annotation)
    for record in val_records:
        records_by_file[record["file_name"]].append(record)

    device = torch.device("cuda")
    checkpoint = torch.load(refiner_checkpoint, map_location="cpu", weights_only=False)
    refiner = CropUNet(pretrained=False).to(device)
    refiner.load_state_dict(checkpoint["model"])
    refiner.eval()
    detector_paths = [
        value.strip()
        for value in detector_checkpoints.split(",")
        if value.strip()
    ] or [detector_checkpoint]
    detectors = [YOLO(path) for path in detector_paths]
    settings = [
        (det_conf, ref_threshold)
        for det_conf in (0.15, 0.25, 0.35)
        for ref_threshold in (0.55, 0.65, 0.75)
    ]
    tta_modes = ("four_flip_prob", "d4_logit")
    rows_by_setting = {
        (tta_mode, det_conf, ref_threshold): []
        for tta_mode in tta_modes
        for det_conf, ref_threshold in settings
    }

    for filename, file_records in tqdm(
        sorted(records_by_file.items()), desc="crop refiner validation"
    ):
        image_path = root / "train" / "train_images" / filename
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        boxes_with_scores = []
        for detector in detectors:
            detection = detector.predict(
                source=rgb,
                imgsz=detector_image_size,
                device=0,
                conf=0.05,
                iou=0.7,
                max_det=80,
                verbose=False,
            )[0]
            if detection.boxes is not None and len(detection.boxes):
                boxes = detection.boxes.xyxy.detach().cpu().numpy()
                scores = detection.boxes.conf.detach().cpu().numpy()
                boxes_with_scores.extend(
                    (tuple(map(float, box)), float(score))
                    for box, score in zip(boxes, scores)
                )
        if len(detectors) > 1 and boxes_with_scores:
            boxes_with_scores.sort(key=lambda item: item[1], reverse=True)
            kept = []
            for candidate_box, candidate_score in boxes_with_scores:
                x0, y0, x1, y1 = candidate_box
                area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                should_keep = True
                for kept_box, _ in kept:
                    kx0, ky0, kx1, ky1 = kept_box
                    iw = max(0.0, min(x1, kx1) - max(x0, kx0))
                    ih = max(0.0, min(y1, ky1) - max(y0, ky0))
                    intersection = iw * ih
                    kept_area = max(0.0, kx1 - kx0) * max(0.0, ky1 - ky0)
                    union = area + kept_area - intersection
                    if union > 0 and intersection / union >= ensemble_nms_iou:
                        should_keep = False
                        break
                if should_keep:
                    kept.append((candidate_box, candidate_score))
            boxes_with_scores = kept

        crops, bounds = [], []
        for box, _score in boxes_with_scores:
            crop, crop_bounds = make_inference_crop(gray, box)
            crops.append(crop)
            bounds.append(crop_bounds)
        proposal_rows = []
        if crops:
            crop_batch = np.stack(crops).astype(np.float32)
            # Keep the first four views in the existing identity/h/v/hv
            # order, then add the four diagonal-reflection views to complete
            # D4. Each tuple means rotate counter-clockwise, then h-flip.
            d4_transforms = (
                (0, False), (0, True), (2, True), (2, False),
                (1, False), (1, True), (3, False), (3, True),
            )
            transformed = []
            for k, hflip in d4_transforms:
                rotated = np.rot90(crop_batch, k=k, axes=(-2, -1)).copy()
                transformed.append(
                    rotated[:, :, :, ::-1].copy() if hflip else rotated
                )
            all_inputs = torch.from_numpy(np.concatenate(transformed, axis=0))
            logits = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for batch in all_inputs.split(32):
                    logits.append(refiner(batch.to(device)).float().cpu().numpy()[:, 0])
            logits = np.concatenate(logits, axis=0).reshape(8, len(crops), 256, 256)
            # Undo each transform: optional horizontal flip first, then
            # reverse the quarter-turn rotation.
            aligned_logits = []
            for view_index, (k, hflip) in enumerate(d4_transforms):
                transformed_logits = logits[view_index]
                if hflip:
                    transformed_logits = transformed_logits[:, :, ::-1]
                aligned_logits.append(
                    np.rot90(transformed_logits, k=-k, axes=(-2, -1)).copy()
                )
            aligned_logits = np.stack(aligned_logits)
            d4_probabilities = 1.0 / (1.0 + np.exp(-np.clip(aligned_logits, -30, 30)))
            probability_by_mode = {
                "four_flip_prob": d4_probabilities[:4].mean(axis=0),
                "d4_logit": 1.0 / (1.0 + np.exp(-np.clip(aligned_logits.mean(axis=0), -30, 30))),
            }
            proposal_rows_by_mode = {
                mode: [
                    (score, bounds[i], probability_by_mode[mode][i])
                    for i, (_box, score) in enumerate(boxes_with_scores)
                ]
                for mode in tta_modes
            }
        else:
            proposal_rows_by_mode = {mode: [] for mode in tta_modes}

        for tta_mode in tta_modes:
            proposal_rows = proposal_rows_by_mode[tta_mode]
            proposal_rows.sort(key=lambda row: row[0], reverse=True)
            for det_conf, ref_threshold in settings:
                labels = np.zeros(gray.shape, dtype=np.uint16)
                claimed = np.zeros(gray.shape, dtype=bool)
                next_id = 1
                for score, (x0, y0, side), probability in proposal_rows:
                    if score < det_conf:
                        continue
                    crop_mask = cv2.resize(
                        probability,
                        (side, side),
                        interpolation=cv2.INTER_LINEAR,
                    ) >= ref_threshold
                    xa, ya = max(0, x0), max(0, y0)
                    xb, yb = min(gray.shape[1], x0 + side), min(gray.shape[0], y0 + side)
                    if xb <= xa or yb <= ya:
                        continue
                    candidate = crop_mask[ya - y0 : yb - y0, xa - x0 : xb - x0]
                    free = candidate & ~claimed[ya:yb, xa:xb]
                    if int(free.sum()) < 96:
                        continue
                    labels[ya:yb, xa:xb][free] = next_id
                    claimed[ya:yb, xa:xb] |= free
                    next_id += 1

                for record in file_records:
                    image_meta = images_by_id[record["image_id"]]
                    gt_masks = [
                        polygon_mask(
                            annotation.get("segmentation"),
                            image_meta["height"],
                            image_meta["width"],
                        ).astype(bool)
                        for annotation in anns_by_image.get(record["image_id"], [])
                    ]
                    rows_by_setting[(tta_mode, det_conf, ref_threshold)].append(
                        pq_counts(labels, gt_masks)
                    )

    result_rows = []
    for setting, rows in rows_by_setting.items():
        tta_mode, det_conf, ref_threshold = setting
        result_rows.append(
            {
                "tta_mode": tta_mode,
                "detection_threshold": det_conf,
                "refiner_threshold": ref_threshold,
                "min_area": 96,
                **aggregate_pq(rows),
            }
        )
    result_rows.sort(key=lambda row: row["pq_ratio_of_sums"], reverse=True)
    payload = {
        "refiner_checkpoint": refiner_checkpoint,
        "detector_checkpoint": detector_checkpoint,
        "detector_checkpoints": detector_paths,
        "ensemble_nms_iou": ensemble_nms_iou if len(detectors) > 1 else None,
        "validation_physical_images": len(records_by_file),
        "validation_views": len(val_records),
        "training_checkpoint_epoch": int(checkpoint["epoch"]),
        "settings": result_rows,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    volume.commit()
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved crop refiner validation: {destination}", flush=True)


@app.function(
    image=image,
    gpu=["L4", "A10", "T4"],
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def crop_refiner_submission_job(
    refiner_checkpoint: str = "/data/runs/seed_refiner_resnet18_seed42/best.pt",
    detector_checkpoint: str = "/data/runs/detector_fusion_yolo11s_seed42/train/weights/best.pt",
    output_csv: str = "/data/runs/seed_refiner_resnet18_seed42/submission.csv",
    detector_image_size: int = 1024,
    detection_threshold: float = 0.35,
    refiner_threshold: float = 0.55,
    min_area: int = 96,
):
    """Generate and audit test predictions for the selected crop-refiner setup."""
    import csv
    import json

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from ultralytics import YOLO

    sys.path.insert(0, "/root/src")
    from solar_filament.crop_refiner import CropUNet, make_inference_crop
    from solar_filament.inference import encode_instance
    from solar_filament.rle import audit_submission

    root = _ensure_dataset()
    test_dir = root / "test" / "test_images"
    paths = sorted(test_dir.glob("*.jpeg"))
    if not paths:
        raise FileNotFoundError(f"No test JPEGs found in {test_dir}")
    state = torch.load(refiner_checkpoint, map_location="cpu", weights_only=False)
    refiner = CropUNet(pretrained=False).cuda()
    refiner.load_state_dict(state["model"])
    refiner.eval()
    detector = YOLO(detector_checkpoint)
    transforms = ("identity", "h", "v", "hv")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    instance_counts: dict[str, int] = {}

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["filament_id", "segmentation_rle"])
        for index, path in enumerate(tqdm(paths, desc="test seed-conditioned refiner"), start=1):
            gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            detection = detector.predict(
                source=rgb,
                imgsz=detector_image_size,
                device=0,
                conf=0.05,
                iou=0.7,
                max_det=80,
                verbose=False,
            )[0]
            proposals = []
            if detection.boxes is not None and len(detection.boxes):
                boxes = detection.boxes.xyxy.detach().cpu().numpy()
                scores = detection.boxes.conf.detach().cpu().numpy()
                crops, bounds = [], []
                for box in boxes:
                    crop, crop_bounds = make_inference_crop(gray, tuple(map(float, box)))
                    crops.append(crop)
                    bounds.append(crop_bounds)
                if crops:
                    crop_batch = np.stack(crops).astype(np.float32)
                    transformed = []
                    for transform in transforms:
                        if transform == "identity":
                            transformed.append(crop_batch)
                        elif transform == "h":
                            transformed.append(crop_batch[:, :, :, ::-1].copy())
                        elif transform == "v":
                            transformed.append(crop_batch[:, :, ::-1, :].copy())
                        else:
                            transformed.append(crop_batch[:, :, ::-1, ::-1].copy())
                    all_inputs = torch.from_numpy(np.concatenate(transformed, axis=0))
                    probabilities = []
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        for batch in all_inputs.split(32):
                            probabilities.append(
                                torch.sigmoid(refiner(batch.cuda())).float().cpu().numpy()[:, 0]
                            )
                    probs = np.concatenate(probabilities, axis=0).reshape(
                        4, len(crops), 256, 256
                    )
                    probs[1] = probs[1, :, :, ::-1]
                    probs[2] = probs[2, :, ::-1, :]
                    probs[3] = probs[3, :, ::-1, ::-1]
                    probs = probs.mean(axis=0)
                    proposals = [
                        (float(score), bounds[i], probs[i])
                        for i, score in enumerate(scores)
                    ]

            labels = np.zeros(gray.shape, dtype=np.uint16)
            claimed = np.zeros(gray.shape, dtype=bool)
            next_id = 1
            for score, (x0, y0, side), probability in sorted(
                proposals, key=lambda row: row[0], reverse=True
            ):
                if score < detection_threshold:
                    continue
                crop_mask = cv2.resize(
                    probability,
                    (side, side),
                    interpolation=cv2.INTER_LINEAR,
                ) >= refiner_threshold
                xa, ya = max(0, x0), max(0, y0)
                xb, yb = min(gray.shape[1], x0 + side), min(gray.shape[0], y0 + side)
                if xb <= xa or yb <= ya:
                    continue
                candidate = crop_mask[ya - y0 : yb - y0, xa - x0 : xb - x0]
                free = candidate & ~claimed[ya:yb, xa:xb]
                if int(free.sum()) < min_area:
                    continue
                labels[ya:yb, xa:xb][free] = next_id
                claimed[ya:yb, xa:xb] |= free
                next_id += 1

            count = next_id - 1
            instance_counts[path.stem] = count
            for instance_id in range(1, count + 1):
                writer.writerow(
                    [f"{path.stem}_{instance_id}", encode_instance(labels == instance_id)]
                )
            if index % 20 == 0:
                print(f"Predicted {index}/{len(paths)} test images", flush=True)

    image_shapes = {path.stem: (2048, 2048) for path in paths}
    row_count = audit_submission(
        output_path, image_shapes, expected_image_ids=image_shapes.keys()
    )
    config = {
        "pipeline": "YOLO11s-seg proposals + seed-conditioned ResNet18 U-Net crop refiner",
        "refiner_checkpoint": refiner_checkpoint,
        "refiner_epoch": int(state["epoch"]),
        "detector_checkpoint": detector_checkpoint,
        "detector_image_size": detector_image_size,
        "detection_threshold": detection_threshold,
        "refiner_threshold": refiner_threshold,
        "min_area": min_area,
        "tta": ["identity", "hflip", "vflip", "hvflip"],
        "test_images": len(paths),
        "submission_rows": row_count,
        "instances_per_image": instance_counts,
        "rle_audit": "passed",
    }
    config_path = output_path.with_name("submission_config.json")
    config_path.write_text(json.dumps(config, indent=2))
    volume.commit()
    print(f"Submission audit passed: {row_count} rows across {len(paths)} images", flush=True)
    print(f"Candidate CSV saved at {output_path}", flush=True)
    print(f"Candidate config saved at {config_path}", flush=True)


@app.function(
    image=rfdetr_image,
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=8 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def rfdetr_seg_train_job(
    run_name: str = "rfdetr_seg_medium_seed42",
    epochs: int = 25,
    resolution: int = 576,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
):
    """Fine-tune RF-DETR-Seg on the same physical-image grouped split."""
    import json
    import os
    import shutil
    from collections import defaultdict

    import torch
    from pytorch_lightning.callbacks import Callback
    from rfdetr.config import RFDETRSegMediumConfig, SegmentationTrainConfig
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    sys.path.insert(0, "/root/src")
    from solar_filament.data import grouped_split, prepare_records

    root = _ensure_dataset()
    records = prepare_records(root, "/data/cache")
    train_records, val_records = grouped_split(records, val_fraction=0.2, seed=42)
    annotation_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = json.loads(annotation_path.read_text())
    anns_by_image: dict[str, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    dataset_root = Path("/data/rfdetr") / run_name / "dataset"
    image_root = root / "train" / "train_images"
    dataset_root.mkdir(parents=True, exist_ok=True)
    next_annotation_id = 1
    split_manifest = {}
    for split_name, split_records in (("train", train_records), ("valid", val_records)):
        split_dir = dataset_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        images_json, annotations_json = [], []
        for image_index, record in enumerate(split_records, start=1):
            view_id = str(record["image_id"])
            image_name = f"view_{image_index:05d}.jpeg"
            image_link = split_dir / image_name
            source = image_root / record["file_name"]
            if image_link.is_symlink() or image_link.exists():
                image_link.unlink()
            image_link.symlink_to(source)
            image_row_id = image_index
            images_json.append(
                {
                    "id": image_row_id,
                    "file_name": image_name,
                    "width": int(record["width"]),
                    "height": int(record["height"]),
                }
            )
            for ann in anns_by_image.get(view_id, []):
                segmentation = ann.get("segmentation")
                if not isinstance(segmentation, list) or not segmentation:
                    continue
                polygons = (
                    [segmentation]
                    if isinstance(segmentation[0], (int, float))
                    else segmentation
                )
                polygons = [
                    [float(value) for value in polygon]
                    for polygon in polygons
                    if isinstance(polygon, list) and len(polygon) >= 6
                ]
                if not polygons:
                    continue
                bbox = ann.get("bbox", [0, 0, 0, 0])
                annotations_json.append(
                    {
                        "id": next_annotation_id,
                        "image_id": image_row_id,
                        # The four source categories describe filament labels;
                        # the task's scored target is a single filament class.
                        "category_id": 1,
                        "segmentation": polygons,
                        "area": float(ann.get("area", 0.0)),
                        "bbox": [float(value) for value in bbox],
                        "iscrowd": 0,
                    }
                )
                next_annotation_id += 1
        payload = {
            "info": {"description": "MAGFiLO filament instance segmentation"},
            "licenses": [],
            "images": images_json,
            "annotations": annotations_json,
            "categories": [{"id": 1, "name": "filament", "supercategory": "filament"}],
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(payload, separators=(",", ":"))
        )
        split_manifest[split_name] = {
            "annotator_views": len(images_json),
            "physical_images": len({record["file_name"] for record in split_records}),
            "instances": len(annotations_json),
        }
    volume.commit()

    output_dir = Path("/data/runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model": "RF-DETR-Seg-Medium",
        "pretrained": "rf-detr-seg-medium.pt",
        "resolution": resolution,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "seed": 42,
        "grouped_split_seed": 42,
        "split": split_manifest,
        "image_rule": "same physical image stays in one split; each annotator view is a training example",
        "categories": "all filament instance categories collapsed to one filament class",
        "augmentation": "horizontal/vertical flips, 180-degree rotations, mild brightness/contrast and blur",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    model_config = RFDETRSegMediumConfig(
        num_classes=1,
        resolution=resolution,
        device="cuda",
    )
    train_config = SegmentationTrainConfig(
        dataset_dir=str(dataset_root),
        dataset_file="roboflow",
        class_names=["filament"],
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=1e-4,
        lr_encoder=1e-5,
        num_workers=4,
        seed=42,
        use_ema=True,
        early_stopping=False,
        eval_interval=epochs,
        multi_scale=False,
        expanded_scales=False,
        scale_jitter=False,
        amp_dtype="bf16",
        progress_bar="tqdm",
        tensorboard=False,
        run_test=False,
        aug_config={
            "HorizontalFlip": {"p": 0.5},
            "VerticalFlip": {"p": 0.5},
            "Rotate": {"limit": 180, "p": 0.5},
            "RandomBrightnessContrast": {
                "brightness_limit": 0.12,
                "contrast_limit": 0.12,
                "p": 0.35,
            },
            "GaussianBlur": {"blur_limit": (3, 3), "p": 0.1},
        },
    )

    last_checkpoint = output_dir / "last_modal.ckpt"

    class DurableEpochCheckpoint(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            if trainer.is_global_zero:
                trainer.save_checkpoint(str(last_checkpoint))
                volume.commit()
                print(
                    f"Committed RF-DETR checkpoint after epoch {trainer.current_epoch + 1}",
                    flush=True,
                )

    module = RFDETRModelModule(model_config, train_config)
    datamodule = RFDETRDataModule(model_config, train_config)
    trainer = build_trainer(
        train_config,
        model_config,
        log_every_n_steps=25,
    )
    # RF-DETR 1.9.1 exposes custom callbacks through the Trainer callback list;
    # appending preserves its built-in EMA, COCO evaluation, and best-checkpoint callbacks.
    trainer.callbacks.append(DurableEpochCheckpoint())
    resume_path = str(last_checkpoint) if last_checkpoint.exists() else None
    print(
        f"Starting RF-DETR-Seg-Medium on {torch.cuda.get_device_name(0)}; "
        f"resolution={resolution}; train={split_manifest['train']}; "
        f"val={split_manifest['valid']}; resume={resume_path}",
        flush=True,
    )
    trainer.fit(module, datamodule, ckpt_path=resume_path)
    volume.commit()
    best_checkpoint = output_dir / "checkpoint_best_total.pth"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"RF-DETR did not produce {best_checkpoint}")
    print(f"RF-DETR best checkpoint: {best_checkpoint}", flush=True)
    print(f"RF-DETR config: {output_dir / 'run_config.json'}", flush=True)


@app.function(
    image=rfdetr_image,
    gpu=["A10", "L4", "T4"],
    cpu=8,
    memory=32768,
    timeout=4 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def rfdetr_seg_validation_job(
    run_name: str = "rfdetr_seg_medium_seed42",
    rfdetr_checkpoint: str = "/data/runs/rfdetr_seg_medium_seed42/checkpoint_best_total.pth",
    detector_checkpoint: str = "/data/runs/detector_fusion_yolo11s_seed42/train/weights/best.pt",
    refiner_checkpoint: str = "/data/runs/seed_refiner_resnet18_seed42/best.pt",
    resolution: int = 576,
    output_path: str = "",
):
    """Score RF-DETR-Seg and test mask-set fusion with the current best pipeline."""
    import json
    from collections import defaultdict

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from pycocotools import mask as mask_utils
    from rfdetr import RFDETRSegMedium
    from tqdm import tqdm
    from ultralytics import YOLO

    sys.path.insert(0, "/root/src")
    from solar_filament.crop_refiner import CropUNet, make_inference_crop
    from solar_filament.data import grouped_split, polygon_mask, prepare_records
    from solar_filament.metrics import aggregate_pq, pq_counts

    root = _ensure_dataset()
    records = prepare_records(root, "/data/cache")
    _, val_records = grouped_split(records, val_fraction=0.2, seed=42)
    coco = json.loads(
        (root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text()
    )
    images_by_id = {image["id"]: image for image in coco["images"]}
    anns_by_image: dict[str, list[dict]] = defaultdict(list)
    records_by_file: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        anns_by_image[annotation["image_id"]].append(annotation)
    for record in val_records:
        records_by_file[record["file_name"]].append(record)

    model = RFDETRSegMedium(
        pretrain_weights=rfdetr_checkpoint,
        resolution=resolution,
        device="cuda",
    )
    yolo = YOLO(detector_checkpoint)
    refiner_state = torch.load(refiner_checkpoint, map_location="cpu", weights_only=False)
    refiner = CropUNet(pretrained=False).cuda().eval()
    refiner.load_state_dict(refiner_state["model"])

    rf_thresholds = (0.15, 0.25, 0.35, 0.45)
    min_areas = (64, 128, 200)
    setting_keys = []
    for threshold in rf_thresholds:
        for min_area in min_areas:
            setting_keys.append(("rfdetr", threshold, min_area))
            setting_keys.append(("ensemble", threshold, min_area))
    setting_keys.append(("yolo_refiner", 0.35, 96))
    rows_by_setting = {setting: [] for setting in setting_keys}
    qualitative_dir = Path("/data/runs") / run_name / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    random_state = np.random.default_rng(42)
    all_validation_files = sorted(records_by_file)
    qualitative_files = set(
        random_state.choice(
            all_validation_files,
            size=min(4, len(all_validation_files)),
            replace=False,
        ).tolist()
    )

    def render_panel(gray_image, label_map, title):
        base = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        labels = np.asarray(label_map, dtype=np.int32)
        colored = np.zeros_like(base)
        palette = [
            (40, 80, 255),
            (40, 220, 80),
            (255, 130, 30),
            (220, 40, 220),
            (30, 220, 220),
            (220, 220, 30),
        ]
        for instance_id in np.unique(labels):
            if instance_id == 0:
                continue
            colored[labels == instance_id] = palette[(int(instance_id) - 1) % len(palette)]
        foreground = labels > 0
        base[foreground] = (
            0.35 * base[foreground] + 0.65 * colored[foreground]
        ).astype(np.uint8)
        cv2.putText(
            base,
            title,
            (32, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        return base

    def predictions_to_labels(candidates, score_threshold, min_area, normalize_by=1.0, iou_suppress=0.0):
        ordered = sorted(
            [
                (float(score) / normalize_by, np.asarray(mask, dtype=bool))
                for score, mask in candidates
                if float(score) >= score_threshold and np.asarray(mask).any()
            ],
            key=lambda item: item[0],
            reverse=True,
        )
        labels = np.zeros((2048, 2048), dtype=np.uint16)
        claimed = np.zeros((2048, 2048), dtype=bool)
        accepted = []
        instance_id = 1
        for score, candidate in ordered:
            if iou_suppress > 0:
                skip = False
                area_candidate = int(candidate.sum())
                for accepted_mask in accepted:
                    intersection = int(np.count_nonzero(candidate & accepted_mask))
                    if not intersection:
                        continue
                    union = area_candidate + int(accepted_mask.sum()) - intersection
                    if union and intersection / union >= iou_suppress:
                        skip = True
                        break
                if skip:
                    continue
            free = candidate & ~claimed
            if int(free.sum()) < min_area:
                continue
            labels[free] = instance_id
            claimed |= free
            accepted.append(candidate)
            instance_id += 1
        return labels

    transforms = ("identity", "h", "v", "hv")
    for filename, file_records in tqdm(
        sorted(records_by_file.items()), desc="RF-DETR/PQ and ensemble validation"
    ):
        image_path = root / "train" / "train_images" / filename
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        with torch.inference_mode():
            detections = model.predict(rgb, threshold=0.05, include_source_image=False)
        rf_scores = (
            np.asarray(detections.confidence, dtype=np.float32)
            if detections.confidence is not None
            else np.zeros(0, dtype=np.float32)
        )
        rf_masks_raw = detections.mask
        rf_candidates = []
        if rf_masks_raw is not None:
            rf_masks = np.asarray(rf_masks_raw)
            if rf_masks.ndim == 4 and rf_masks.shape[1] == 1:
                rf_masks = rf_masks[:, 0]
            for score, mask in zip(rf_scores, rf_masks):
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != gray.shape:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (gray.shape[1], gray.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                if mask.any():
                    rf_candidates.append((float(score), mask))

        detection = yolo.predict(
            source=rgb,
            imgsz=1024,
            device=0,
            conf=0.05,
            iou=0.7,
            max_det=80,
            verbose=False,
        )[0]
        yolo_candidates = []
        if detection.boxes is not None and len(detection.boxes):
            boxes = detection.boxes.xyxy.detach().cpu().numpy()
            scores = detection.boxes.conf.detach().cpu().numpy()
            crops, bounds = [], []
            for box in boxes:
                crop, crop_bounds = make_inference_crop(gray, tuple(map(float, box)))
                crops.append(crop)
                bounds.append(crop_bounds)
            if crops:
                crop_batch = np.stack(crops).astype(np.float32)
                flipped = []
                for transform in transforms:
                    if transform == "identity":
                        flipped.append(crop_batch)
                    elif transform == "h":
                        flipped.append(crop_batch[:, :, :, ::-1].copy())
                    elif transform == "v":
                        flipped.append(crop_batch[:, :, ::-1, :].copy())
                    else:
                        flipped.append(crop_batch[:, :, ::-1, ::-1].copy())
                all_inputs = torch.from_numpy(np.concatenate(flipped, axis=0))
                probabilities = []
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    for batch in all_inputs.split(32):
                        probabilities.append(
                            torch.sigmoid(refiner(batch.cuda())).float().cpu().numpy()[:, 0]
                        )
                probs = np.concatenate(probabilities, axis=0).reshape(
                    4, len(crops), 256, 256
                )
                probs[1] = probs[1, :, :, ::-1]
                probs[2] = probs[2, :, ::-1, :]
                probs[3] = probs[3, :, ::-1, ::-1]
                probs = probs.mean(axis=0)
                for index, (score, (x0, y0, side)) in enumerate(zip(scores, bounds)):
                    crop_mask = cv2.resize(
                        probs[index], (side, side), interpolation=cv2.INTER_LINEAR
                    ) >= 0.55
                    xa, ya = max(0, x0), max(0, y0)
                    xb, yb = min(gray.shape[1], x0 + side), min(gray.shape[0], y0 + side)
                    if xb <= xa or yb <= ya:
                        continue
                    mask = np.zeros(gray.shape, dtype=bool)
                    mask[ya:yb, xa:xb] = crop_mask[
                        ya - y0 : yb - y0, xa - x0 : xb - x0
                    ]
                    if mask.any():
                        yolo_candidates.append((float(score), mask))

        labels_by_setting = {}
        for threshold in rf_thresholds:
            for min_area in min_areas:
                labels_by_setting[("rfdetr", threshold, min_area)] = predictions_to_labels(
                    rf_candidates, threshold, min_area, iou_suppress=0.7
                )
                combined = [
                    (score / 0.35, mask)
                    for score, mask in yolo_candidates
                    if score >= 0.35
                ] + [
                    (score / threshold, mask)
                    for score, mask in rf_candidates
                    if score >= threshold
                ]
                labels_by_setting[("ensemble", threshold, min_area)] = predictions_to_labels(
                    combined, 0.0, min_area, iou_suppress=0.7
                )
        labels_by_setting[("yolo_refiner", 0.35, 96)] = predictions_to_labels(
            yolo_candidates, 0.35, 96
        )

        if filename in qualitative_files and file_records:
            selected_record = file_records[0]
            selected_meta = images_by_id[selected_record["image_id"]]
            gt_labels = np.zeros(gray.shape, dtype=np.uint16)
            for instance_index, annotation in enumerate(
                anns_by_image.get(selected_record["image_id"], []), start=1
            ):
                gt_mask = polygon_mask(
                    annotation.get("segmentation"),
                    selected_meta["height"],
                    selected_meta["width"],
                ).astype(bool)
                gt_labels[gt_mask & (gt_labels == 0)] = instance_index
            sample_panels = [
                render_panel(gray, np.zeros(gray.shape, dtype=np.uint16), "H-alpha input"),
                render_panel(gray, gt_labels, "One annotator view: reference instances"),
                render_panel(
                    gray,
                    labels_by_setting[("rfdetr", 0.25, 64)],
                    "RF-DETR-Seg (conf .25)" ,
                ),
                render_panel(
                    gray,
                    labels_by_setting[("ensemble", 0.25, 64)],
                    "YOLO refiner + RF-DETR (mask NMS)",
                ),
            ]
            cv2.imwrite(
                str(qualitative_dir / f"{Path(filename).stem}_comparison.png"),
                np.concatenate(
                    [
                        np.concatenate(sample_panels[:2], axis=1),
                        np.concatenate(sample_panels[2:], axis=1),
                    ],
                    axis=0,
                ),
            )

        for record in file_records:
            meta = images_by_id[record["image_id"]]
            gt_masks = [
                polygon_mask(
                    ann.get("segmentation"), meta["height"], meta["width"]
                ).astype(bool)
                for ann in anns_by_image.get(record["image_id"], [])
            ]
            for setting, labels in labels_by_setting.items():
                rows_by_setting[setting].append(pq_counts(labels, gt_masks))

    result_rows = []
    for setting, rows in rows_by_setting.items():
        model_name, threshold, min_area = setting
        result_rows.append(
            {
                "model": model_name,
                "confidence_threshold": threshold,
                "min_area": min_area,
                **aggregate_pq(rows),
            }
        )
    result_rows.sort(key=lambda row: row["pq_ratio_of_sums"], reverse=True)
    payload = {
        "rfdetr_checkpoint": rfdetr_checkpoint,
        "yolo_checkpoint": detector_checkpoint,
        "refiner_checkpoint": refiner_checkpoint,
        "resolution": resolution,
        "validation_physical_images": len(records_by_file),
        "validation_views": len(val_records),
        "ensemble_method": "confidence-normalized instance-mask union, duplicate mask IoU suppression at 0.7, then disjoint score-ordered assignment",
        "qualitative_samples": [
            str(path)
            for path in sorted(qualitative_dir.glob("*_comparison.png"))
        ],
        "results": result_rows,
    }
    destination = Path(output_path) if output_path else Path("/data/runs") / run_name / "validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    volume.commit()
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved RF-DETR validation and ensemble sweep: {destination}", flush=True)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=5 * 60 * 60,
    volumes={"/data": volume},
)
def oof_refiner_pipeline_job(
    oof_predictions_path: str = "/data/runs/oof_yolo11s_seed42_3fold/oof_proposals.json",
    run_name: str = "oof_crop_refiner_resnet18_seed42",
    init_checkpoint: str = "/data/runs/seed_refiner_resnet18_seed42/best.pt",
):
    """Wait for OOF proposals, then train and score the crop-refiner experiment."""
    import time

    deadline = time.time() + 3 * 60 * 60
    while time.time() < deadline:
        volume.reload()
        if Path(oof_predictions_path).is_file():
            break
        print("Waiting for committed grouped OOF proposals...", flush=True)
        time.sleep(60)
    else:
        raise TimeoutError(f"OOF proposal job did not finish: {oof_predictions_path}")

    print(f"Starting OOF crop-refiner fine-tune: {run_name}", flush=True)
    crop_refiner_train_job.remote(
        run_name=run_name,
        epochs=5,
        batch_size=32,
        learning_rate=1e-4,
        seed=42,
        oof_predictions_path=oof_predictions_path,
        init_checkpoint=init_checkpoint,
    )
    crop_refiner_validation_job.remote(
        refiner_checkpoint=f"/data/runs/{run_name}/best.pt",
        detector_checkpoint="/data/runs/detector_fusion_yolo11s_seed42/train/weights/best.pt",
        output_path=f"/data/runs/{run_name}/validation.json",
        detector_image_size=1024,
    )
    print(f"OOF refiner validation saved under /data/runs/{run_name}", flush=True)


@app.function(
    image=image,
    gpu=["L4", "A10", "L40S", "A100", "T4"],
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def dump_proposals_job(
    detector_checkpoint: str = "/data/runs/detector_fusion_yolo11s_seed42/train/weights/best.pt",
    refiner_checkpoint: str = "/data/runs/oof_crop_refiner_resnet18_seed42/best.pt",
    output_dir: str = "/data/runs/dump_val_yolo11s_oofref",
    split: str = "val",
    detector_image_size: int = 1024,
    min_conf: float = 0.05,
    files_json: str = "",
    proposals_json: str = "",
    dense_dir: str = "",
    aux_detectors: str = "yolo11m=/data/runs/detector_fusion_yolo11m_1536_seed42/train/weights/best.pt@1536,yolo11x=/data/runs/yolo11x_seg_seed42_e50/train/weights/best.pt@1024",
):
    """Save every detector proposal plus its 4-flip refiner probability crop.

    ``aux_detectors`` (name=path@imgsz, comma-separated) are run too; their boxes
    and scores are stored under ``aux`` for agreement features.

    Output: ``proposals.json`` ({file: [{bbox, score, crop:[x0,y0,side]}]}),
    and one ``<stem>.npz`` per image with ``prob`` uint8 [N,256,256] (p*255) and
    ``yolo`` uint8 [N,256,256] (native YOLO mask resampled to the same crop).
    Post-processing can then be tuned on CPU without the GPU.
    """
    import json

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from ultralytics import YOLO

    sys.path.insert(0, "/root/src")
    from solar_filament import crop_refiner as _cr
    from solar_filament.crop_refiner import CropUNet, make_inference_crop
    from solar_filament.data import grouped_split, prepare_records

    root = _ensure_dataset()
    volume.reload()
    if files_json:
        paths = [root / "train" / "train_images" / f for f in json.loads(Path(files_json).read_text())]
    elif split == "test":
        paths = sorted((root / "test" / "test_images").glob("*.jp*g"))
        if len(paths) != 180:
            raise RuntimeError(f"Expected 180 test images, found {len(paths)}")
    else:
        records = prepare_records(root, "/data/cache")
        _, val_records = grouped_split(records, val_fraction=0.2, seed=42)
        paths = [root / "train" / "train_images" / f for f in sorted({r["file_name"] for r in val_records})]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    state = torch.load(refiner_checkpoint, map_location="cpu", weights_only=False)
    _cr.set_crop_size(int(state.get("crop_size", 256)))
    CROP_SIZE = _cr.CROP_SIZE
    refiner = CropUNet(pretrained=False, encoder_name=state.get("encoder", "resnet18")).to(device)
    refiner.load_state_dict(state["model"])
    refiner.eval()
    # External proposals (boxes from dense probability maps) replace the detector when given.
    external = json.loads(Path(proposals_json).read_text())["predictions"] if proposals_json else None
    detector = None if external is not None else YOLO(detector_checkpoint)
    aux_models = {}
    for spec in filter(None, (x.strip() for x in aux_detectors.split(","))):
        name, rest = spec.split("=", 1)
        ckpt, size = rest.rsplit("@", 1)
        aux_models[name] = (YOLO(ckpt), int(size))
    aux_out = {name: {} for name in aux_models}
    proposals = {}
    for path in tqdm(paths, desc=f"dump {split}"):
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        rows, crops, yolo_crops = [], [], []
        boxes = None
        if external is not None:
            ext = external.get(path.name, [])
            if ext:
                boxes = np.array([r["bbox"] for r in ext], dtype=np.float32)
                scores = np.array([r["score"] for r in ext], dtype=np.float32)
                dense = cv2.imread(str(Path(dense_dir) / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
                if dense.shape != gray.shape:
                    dense = cv2.resize(dense, gray.shape[::-1], interpolation=cv2.INTER_LINEAR)
                masks = []
                for r in ext:  # the "detector mask" of a dense proposal is the thresholded map inside its box
                    m = np.zeros(gray.shape, dtype=bool)
                    bx0, by0, bx1, by1 = (int(v) for v in r["bbox"])
                    m[by0:by1, bx0:bx1] = dense[by0:by1, bx0:bx1] >= r["level"] * 255
                    masks.append(m)
        else:
            det = detector.predict(source=rgb, imgsz=detector_image_size, device=0, conf=min_conf,
                                   iou=0.7, max_det=100, verbose=False, retina_masks=True)[0]
            if det.boxes is not None and len(det.boxes):
                boxes = det.boxes.xyxy.cpu().numpy()
                scores = det.boxes.conf.cpu().numpy()
                masks = det.masks.data.cpu().numpy() > 0.5 if det.masks is not None else None
        if boxes is not None:
            for k, (box, score) in enumerate(zip(boxes, scores)):
                crop, (x0, y0, side) = make_inference_crop(gray, tuple(map(float, box)))
                crops.append(crop)
                ym = np.zeros((side, side), dtype=np.uint8)
                if masks is not None:
                    xa, ya = max(0, x0), max(0, y0)
                    xb, yb = min(gray.shape[1], x0 + side), min(gray.shape[0], y0 + side)
                    ym[ya - y0:yb - y0, xa - x0:xb - x0] = masks[k][ya:yb, xa:xb]
                yolo_crops.append(cv2.resize(ym * 255, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA))
                rows.append({"bbox": [float(v) for v in box], "score": float(score), "crop": [x0, y0, side]})
        probs = np.zeros((0, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        if crops:
            batch = torch.from_numpy(np.stack(crops))
            acc = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for chunk in batch.split(32):
                    chunk = chunk.to(device)
                    p = torch.sigmoid(refiner(chunk).float())
                    p = p + torch.sigmoid(refiner(chunk.flip(-1)).float()).flip(-1)
                    p = p + torch.sigmoid(refiner(chunk.flip(-2)).float()).flip(-2)
                    p = p + torch.sigmoid(refiner(chunk.flip(-1, -2)).float()).flip(-1, -2)
                    acc.append((p / 4)[:, 0].cpu().numpy())
            probs = np.clip(np.concatenate(acc) * 255 + 0.5, 0, 255).astype(np.uint8)
        for name, (aux_model, aux_size) in aux_models.items():
            ad = aux_model.predict(source=rgb, imgsz=aux_size, device=0, conf=min_conf,
                                   iou=0.7, max_det=100, verbose=False)[0]
            aux_out[name][path.name] = [
                {"bbox": [float(v) for v in b], "score": float(c)}
                for b, c in zip(ad.boxes.xyxy.cpu().numpy(), ad.boxes.conf.cpu().numpy())
            ] if ad.boxes is not None and len(ad.boxes) else []
        np.savez_compressed(out / f"{path.stem}.npz", prob=probs,
                            yolo=np.stack(yolo_crops) if yolo_crops else probs)
        proposals[path.name] = rows
    (out / "proposals.json").write_text(json.dumps({
        "detector_checkpoint": detector_checkpoint, "refiner_checkpoint": refiner_checkpoint,
        "proposals_json": proposals_json,
        "split": split, "detector_image_size": detector_image_size, "min_conf": min_conf,
        "aux": aux_out, "predictions": proposals}))
    volume.commit()
    print(f"Dumped {sum(len(v) for v in proposals.values())} proposals for {len(paths)} images to {out}", flush=True)


@app.function(image=image, cpu=2, memory=4096, timeout=10 * 60 * 60, volumes={"/data": volume})
def full_data_pipeline_job(run_prefix: str = "full707", seed: int = 42):
    """Train final models on all 707 images with the validated recipe.

    1) YOLO11s-seg 1024 px, 60 epochs on all images (runs in parallel).
    2) Refiner: OOF proposals for all 707 images (3-fold OOF for 566 + seed-42
       held-out detector for the 141 validation images), fine-tuned 5 epochs at
       1e-4 from the medoid-trained refiner, matching the validated OOF recipe.
    """
    import json

    det_call = detector_train_job.spawn(run_name=f"{run_prefix}_yolo11s_s{seed}", detector_model="yolo11s-seg",
                                        epochs=60, image_size=1024, batch_size=8, full_data=True, seed=seed)
    val_dump = "/data/runs/dump_val_yolo11s_oofref"
    volume.reload()
    if not Path(f"{val_dump}/proposals.json").is_file():
        dump_proposals_job.remote(output_dir=val_dump, split="val")
        volume.reload()
    oof = json.loads(Path("/data/runs/oof_yolo11s_seed42_3fold/oof_proposals.json").read_text())["predictions"]
    val = json.loads(Path(f"{val_dump}/proposals.json").read_text())["predictions"]
    merged = {**oof, **{k: [{"bbox": r["bbox"], "score": r["score"]} for r in v] for k, v in val.items()}}
    merged_path = Path(f"/data/runs/{run_prefix}_oof_proposals.json")
    merged_path.write_text(json.dumps({"predictions": merged}))
    volume.commit()
    print(f"Merged OOF proposals for {len(merged)} images", flush=True)
    crop_refiner_train_job.remote(run_name=f"{run_prefix}_refiner_s{seed}", epochs=5, batch_size=32,
                                  learning_rate=1e-4, seed=seed, oof_predictions_path=str(merged_path),
                                  init_checkpoint="/data/runs/seed_refiner_resnet18_seed42/best.pt",
                                  full_data=True)
    det_call.get()
    print("Full-data detector and refiner finished", flush=True)


KF_BASE = ["/data/runs/dump_kf5b_fold0", "/data/runs/dump_kf5b_fold1", "/data/runs/dump_kf5b_fold2",
           "/data/runs/dump_kf5_fold3", "/data/runs/dump_kf5_fold4"]
KF_1536 = [f"/data/runs/dump_kf5_1536_fold{k}" for k in range(5)]


@app.function(image=image, cpu=2, memory=4096, timeout=6 * 60 * 60, volumes={"/data": volume})
def dense_pool_job(tag: str = "dense"):
    """Dense-map candidates: proposals from U-Net maps, refined per fold, pooled in the lab."""
    import json

    for split, maps in (("oof", "unet_prob_oof"), ("test", "unet_prob_test")):
        lab_script_job.remote(script="dense_proposals.py",
                              args=f"--maps=/data/runs/{maps} --out=/data/runs/{tag}_proposals_{split}.json")
    volume.reload()
    calls = []
    for k, d in enumerate(KF_BASE):
        hdr = json.loads((Path(d) / "proposals.json").read_text())
        files = f"/data/runs/{tag}_fold{k}_files.json"
        Path(files).write_text(json.dumps(sorted(hdr["predictions"])))
        volume.commit()
        calls.append(dump_proposals_job.spawn(
            detector_checkpoint="", refiner_checkpoint=hdr["refiner_checkpoint"], files_json=files,
            output_dir=f"/data/runs/dump_{tag}_fold{k}", split="val", aux_detectors="",
            proposals_json=f"/data/runs/{tag}_proposals_oof.json", dense_dir="/data/runs/unet_prob_oof"))
    hdr = json.loads(Path("/data/runs/dump_test_final707/proposals.json").read_text())
    calls.append(dump_proposals_job.spawn(
        detector_checkpoint="", refiner_checkpoint=hdr["refiner_checkpoint"],
        output_dir=f"/data/runs/dump_test_{tag}", split="test", aux_detectors="",
        proposals_json=f"/data/runs/{tag}_proposals_test.json", dense_dir="/data/runs/unet_prob_test"))
    for c in calls:
        c.get()
    base, p1 = ",".join(KF_BASE), ",".join(KF_1536)
    pd_ = ",".join(f"/data/runs/dump_{tag}_fold{k}" for k in range(5))
    labs = [lab_script_job.spawn(script="postproc_lab.py", args=(
        f"--dump={base} --avg_suffix=_r34c384 --pool={pool} --tag={t} --workers=30"))
        for pool, t in ((f"{p1}+{pd_}", f"_avg384_pool1536_{tag}"), (pd_, f"_avg384_pool{tag}"))]
    for c in labs:
        c.get()
    print("dense pool labs finished", flush=True)


sam_image = (
    base_image.pip_install("lightgbm==4.6.0", "scikit-learn==1.7.2", "transformers==4.57.1")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("scripts", remote_path="/root/scripts")
)


@app.function(image=sam_image, gpu=["L40S", "A100", "A10"], cpu=8, memory=49152, timeout=6 * 60 * 60,
              volumes={"/data": volume})
def sam_script_job(script: str, args: str = ""):
    """Run scripts/<script> on a GPU in a container that has transformers (SAM 2.1)."""
    import shlex
    import subprocess

    root = _ensure_dataset()
    env = {**os.environ, "SOLAR_DATA": str(root / "train"), "SOLAR_RESULTS": "/data/results",
           "SOLAR_TEST": str(root / "test" / "test_images"), "PYTHONPATH": "/root/src:/root/scripts"}
    volume.reload()
    subprocess.run([sys.executable, f"/root/scripts/{script}", *shlex.split(args)], check=True, env=env)
    volume.commit()


@app.function(image=lab_image, gpu=["A10", "L4", "L40S", "A100"], cpu=16, memory=65536, timeout=6 * 60 * 60,
              volumes={"/data": volume})
def lab_script_gpu_job(script: str, args: str = ""):
    """Run scripts/<script> in the lab container with a GPU."""
    import shlex
    import subprocess

    root = _ensure_dataset()
    env = {**os.environ, "SOLAR_DATA": str(root / "train"), "SOLAR_RESULTS": "/data/results",
           "SOLAR_TEST": str(root / "test" / "test_images"), "PYTHONPATH": "/root/src:/root/scripts"}
    volume.reload()
    subprocess.run([sys.executable, f"/root/scripts/{script}", *shlex.split(args)], check=True, env=env)
    volume.commit()


@app.function(image=lab_image, cpu=32, memory=65536, timeout=3 * 60 * 60, volumes={"/data": volume})
def lab_script_job(script: str, args: str = ""):
    """Run any scripts/<script> in the CPU lab container with dataset env vars set."""
    import shlex
    import subprocess

    root = _ensure_dataset()
    env = {**os.environ, "SOLAR_DATA": str(root / "train"), "SOLAR_RESULTS": "/data/results",
           "SOLAR_TEST": str(root / "test" / "test_images"), "PYTHONPATH": "/root/src:/root/scripts"}
    volume.reload()
    subprocess.run([sys.executable, f"/root/scripts/{script}", *shlex.split(args)], check=True, env=env)
    volume.commit()


@app.function(image=lab_image, cpu=32, memory=65536, timeout=3 * 60 * 60, volumes={"/data": volume})
def postproc_lab_job(dump_dir: str = "/data/runs/dump_val_yolo11s_oofref", workers: int = 30, then_cnn: bool = False):
    """Run scripts/postproc_lab.py on a dumped proposal set (CPU container)."""
    import subprocess

    root = _ensure_dataset()
    env = {**os.environ, "SOLAR_DATA": str(root / "train"), "SOLAR_RESULTS": "/data/results",
           "PYTHONPATH": "/root/src"}
    volume.reload()
    subprocess.run([sys.executable, "/root/scripts/postproc_lab.py", f"--dump={dump_dir}",
                    f"--workers={workers}"], check=True, env=env)
    volume.commit()
    if then_cnn:
        dumps = dump_dir.split(",")
        lab = f"/data/results/postproc_lab/{Path(dumps[0]).name}" + (f"_x{len(dumps)}" if len(dumps) > 1 else "")
        cnn_rescorer_job.remote(lab_dir=lab, dump_dirs=dump_dir)


@app.function(image=image, cpu=2, memory=4096, timeout=8 * 60 * 60, volumes={"/data": volume})
def soft_refiner_experiment_job(run_name: str = "oof_soft_refiner_seed42"):
    """Val-split experiment: soft-consensus OOF refiner, then dump val proposals."""
    crop_refiner_train_job.remote(
        run_name=run_name, epochs=5, batch_size=32, learning_rate=1e-4, seed=42,
        oof_predictions_path="/data/runs/oof_yolo11s_seed42_3fold/oof_proposals.json",
        init_checkpoint="/data/runs/seed_refiner_resnet18_seed42/best.pt", soft_consensus=True)
    dump_proposals_job.remote(refiner_checkpoint=f"/data/runs/{run_name}/last.pt",
                              output_dir=f"/data/runs/dump_val_3det_{run_name}", split="val")
    postproc_lab_job.remote(dump_dir=f"/data/runs/dump_val_3det_{run_name}")


KFOLDS = 5
KFOLD_SEED = 7


def _nan_guard(trainer):
    """Stop YOLO training as soon as the running loss becomes non-finite."""
    import math

    loss = getattr(trainer, "tloss", None)
    values = loss.tolist() if hasattr(loss, "tolist") else [loss]
    values = values if isinstance(values, list) else [values]
    if any(v is not None and not math.isfinite(float(v)) for v in values):
        raise RuntimeError(f"Non-finite YOLO training loss at epoch {trainer.epoch + 1}: {values}")


def _kfold_files(root: Path, fold: int) -> tuple[list[str], list[str]]:
    import json
    import random

    coco = json.loads((root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
    files = sorted({im["file_name"] for im in coco["images"]})
    random.Random(KFOLD_SEED).shuffle(files)
    held = sorted(files[fold::KFOLDS])
    return sorted(set(files) - set(held)), held


@app.function(
    image=image, gpu=["A10", "L4"], cpu=8, memory=32768, timeout=14 * 60 * 60,
    volumes={"/data": volume}, secrets=[hf_secret],
)
def kfold_detector_job(fold: int, run_prefix: str = "kf5", epochs: int = 60, image_size: int = 1024,
                       lr0: float = 0.002, detector_model: str = "yolo11s-seg"):
    """Train YOLO11s-seg on 4/5 of the images; save in-sample proposals for the refiner."""
    import json

    import numpy as np
    from PIL import Image
    from ultralytics import YOLO

    sys.path.insert(0, "/root/src")
    from solar_filament.data import prepare_records
    from solar_filament.detector_fusion import export_yolo_segmentation_dataset

    root = _ensure_dataset()
    records = prepare_records(root, "/data/cache")
    train_files, held_files = _kfold_files(root, fold)
    out = Path(f"/data/runs/{run_prefix}_fold{fold}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_files.json").write_text(json.dumps(train_files))
    (out / "heldout_files.json").write_text(json.dumps(held_files))
    tr = [r for r in records if r["file_name"] in set(train_files)]
    va = [r for r in records if r["file_name"] in set(held_files)]
    image_root = root / "train" / "train_images"
    yaml_path = export_yolo_segmentation_dataset(
        root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json", image_root, records, tr, va,
        Path(f"/data/yolo/{run_prefix}_fold{fold}"))
    last = out / "train" / "weights" / "last.pt"
    if not (out / "train" / "weights" / "done").is_file():
        if last.is_file():
            model = YOLO(str(last))
            model.add_callback("on_model_save", lambda _t: volume.commit())
            model.train(resume=True)
        else:
            model = YOLO(f"{detector_model}.pt")
            model.add_callback("on_model_save", lambda _t: volume.commit())
            model.add_callback("on_train_epoch_end", _nan_guard)
            model.train(data=str(yaml_path), epochs=epochs, imgsz=image_size, batch=8 if image_size <= 1024 else 4, workers=4, device=0, lr0=lr0,
                        project=str(out), name="train", exist_ok=True, pretrained=True, optimizer="AdamW",
                        cos_lr=True, patience=0, val=False, mosaic=0.0, mixup=0.0, copy_paste=0.0,
                        degrees=180.0, translate=0.05, scale=0.2, fliplr=0.5, flipud=0.5, hsv_h=0.0,
                        hsv_s=0.0, hsv_v=0.1, amp=True, cache=False, plots=False, save=True,
                        save_period=10, seed=KFOLD_SEED + fold, deterministic=False, verbose=False)
        (out / "train" / "weights" / "done").write_text("ok")
        volume.commit()
    detector = YOLO(str(last))
    preds = {}
    for f in train_files:
        gray = np.asarray(Image.open(image_root / f).convert("L"), dtype=np.uint8)
        r = detector.predict(source=np.repeat(gray[:, :, None], 3, axis=2), imgsz=image_size, device=0,
                             conf=0.05, iou=0.7, max_det=80, verbose=False)[0]
        preds[f] = [{"bbox": [float(v) for v in b], "score": float(c)}
                    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())] if len(r.boxes) else []
    (out / "insample_proposals.json").write_text(json.dumps({"predictions": preds}))
    volume.commit()
    print(f"fold {fold}: detector done, {len(train_files)} train / {len(held_files)} held-out", flush=True)


@app.function(image=image, cpu=2, memory=4096, timeout=20 * 60 * 60, volumes={"/data": volume})
def kfold_fold_job(fold: int, run_prefix: str = "kf5", soft_consensus: bool = False,
                   refiner_variant: str = "", encoder: str = "resnet18", crop_size: int = 256,
                   detector_prefix: str = "", image_size: int = 1024, detector_model: str = "yolo11s-seg"):
    out = f"/data/runs/{detector_prefix or run_prefix}_fold{fold}"
    if not detector_prefix:
        kfold_detector_job.remote(fold=fold, run_prefix=run_prefix, image_size=image_size,
                                  detector_model=detector_model)
    volume.reload()
    ref = f"{run_prefix}_fold{fold}_refiner{refiner_variant}"
    if True:  # crop_refiner_train_job resumes from its own last.pt
        crop_refiner_train_job.remote(run_name=ref, epochs=12, batch_size=32, learning_rate=3e-4,
                                      seed=KFOLD_SEED + fold, oof_predictions_path=f"{out}/insample_proposals.json",
                                      init_checkpoint="", soft_consensus=soft_consensus,
                                      train_files_json=f"{out}/train_files.json",
                                      encoder=encoder, crop_size=crop_size)
    dump_proposals_job.remote(detector_checkpoint=f"{out}/train/weights/last.pt",
                              refiner_checkpoint=f"/data/runs/{ref}/last.pt",
                              output_dir=f"/data/runs/dump_{run_prefix}_fold{fold}{refiner_variant}",
                              files_json=f"{out}/heldout_files.json", aux_detectors="",
                              detector_image_size=image_size)
    print(f"fold {fold} complete", flush=True)


@app.function(image=image, cpu=2, memory=4096, timeout=24 * 60 * 60, volumes={"/data": volume})
def kfold_all_job(run_prefix: str = "kf5", soft_consensus: bool = False, folds: str = "0,1,2,3,4",
                  image_size: int = 1024, detector_model: str = "yolo11s-seg"):
    calls = [kfold_fold_job.spawn(fold=int(k), run_prefix=run_prefix, soft_consensus=soft_consensus,
                                  image_size=image_size, detector_model=detector_model)
             for k in folds.split(",")]
    for c in calls:
        c.get()
    print("k-fold runs finished", flush=True)

@app.function(image=lab_image, cpu=16, memory=32768, timeout=2 * 60 * 60, volumes={"/data": volume})
def rescored_submission_job(
    dump_dir: str, rescorer_path: str, output_csv: str, tau: float = 0.26, refiner_threshold: float = 0.65,
    robust: bool = False,
    unet_dir: str = "",
    avg_dump_dir: str = "",
    pool_dump_dirs: str = "",
):
    """Build the submission from a test dump using the learned expected-IoU selector.

    ``pool_dump_dirs`` (comma separated) are test dumps of further detectors whose
    candidates are appended to the candidate pool, as in ``postproc_lab.py --pool``.
    """
    import csv
    import json

    import cv2
    import lightgbm as lgb
    import numpy as np

    sys.path.insert(0, "/root/src")
    from solar_filament.inference import encode_instance
    from solar_filament.postproc import rescored_labels, scorer_robust
    from solar_filament.rle import audit_submission

    root = _ensure_dataset()
    volume.reload()
    model = lgb.Booster(model_file=rescorer_path)
    payload = json.loads((Path(dump_dir) / "proposals.json").read_text())
    paths = sorted((root / "test" / "test_images").glob("*.jp*g"))
    if len(paths) != 180:
        raise RuntimeError(f"Expected 180 test images, found {len(paths)}")
    assert {p.name for p in paths} == set(payload["predictions"]), "dump does not cover the test set"
    pools = [(d, json.loads((Path(d) / "proposals.json").read_text())) for d in pool_dump_dirs.split(",") if d]
    for d, pp in pools:
        assert {p.name for p in paths} == set(pp["predictions"]), f"{d} does not cover the test set"
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts, row_pred = {}, {}
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filament_id", "segmentation_rle"])
        for path in paths:
            z = np.load(Path(dump_dir) / f"{path.stem}.npz")
            probs, yolo = z["prob"], z["yolo"]
            if avg_dump_dir and len(probs):
                p2 = np.load(Path(avg_dump_dir) / f"{path.stem}.npz")["prob"]
                assert len(p2) == len(probs), path.name
                size = max(probs.shape[-1], p2.shape[-1])
                def up(a):
                    return a if a.shape[-1] == size else np.stack(
                        [cv2.resize(x, (size, size), interpolation=cv2.INTER_LINEAR) for x in a])
                probs = ((up(probs).astype(np.uint16) + up(p2)) // 2).astype(np.uint8)
                yolo = up(yolo)
            rows_img = list(payload["predictions"][path.name])
            for pd_, pp in pools:
                z2 = np.load(Path(pd_) / f"{path.stem}.npz")
                r2 = pp["predictions"][path.name]
                if not len(r2):
                    continue
                size = probs.shape[-1] if len(probs) else z2["prob"].shape[-1]
                def rs(a):
                    return a if a.shape[-1] == size else np.stack(
                        [cv2.resize(x, (size, size), interpolation=cv2.INTER_LINEAR) for x in a])
                p2, y2 = rs(z2["prob"]), rs(z2["yolo"])
                probs = np.concatenate([probs, p2]) if len(probs) else p2
                yolo = np.concatenate([yolo, y2]) if len(yolo) else y2
                rows_img += list(r2)
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            unet = cv2.imread(str(Path(unet_dir) / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE) if unet_dir else None
            labels, kept_pred = rescored_labels(rows_img, probs, yolo, gray,
                                                model, tau, refiner_threshold, unet=unet)
            if robust:
                labels = scorer_robust(labels)
            n = int(labels.max())
            counts[path.stem] = n
            if not robust:  # selector value per row, used to order instances in stage 2
                row_pred.update({f"{path.stem}_{i + 1}": float(v) for i, v in enumerate(kept_pred)})
            for i in range(1, n + 1):
                w.writerow([f"{path.stem}_{i}", encode_instance(labels == i)])
    shapes = {p.stem: (2048, 2048) for p in paths}
    rows = audit_submission(out, shapes, expected_image_ids=shapes.keys())
    (out.with_name(out.stem + "_pred.json")).write_text(json.dumps(row_pred))
    (out.with_name(out.stem + "_config.json")).write_text(json.dumps({
        "dump_dir": dump_dir, "rescorer": rescorer_path, "tau": tau, "refiner_threshold": refiner_threshold,
        "scorer_robust": robust, "unet_dir": unet_dir, "avg_dump_dir": avg_dump_dir,
        "pool_dump_dirs": pool_dump_dirs,
        "detector": payload["detector_checkpoint"], "refiner": payload["refiner_checkpoint"],
        "rows": rows, "instances_per_image": counts}, indent=2))
    volume.commit()
    print(f"Audit passed: {rows} rows over {len(paths)} images -> {out}", flush=True)

@app.function(image=lab_image, gpu=["A10", "L4"], cpu=16, memory=65536, timeout=4 * 60 * 60,
              volumes={"/data": volume})
def cnn_rescorer_job(
    lab_dir: str = "/data/results/postproc_lab/dump_kf5_fold0_x5",
    dump_dirs: str = ",".join(f"/data/runs/dump_kf5_fold{k}" for k in range(5)),
    threshold: float = 0.65, epochs: int = 8,
):
    """OOF CNN candidate scorer + LightGBM selector re-evaluation (5-fold by image)."""
    import json
    from concurrent.futures import ProcessPoolExecutor

    import cv2
    import lightgbm as lgb
    import numpy as np
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root/src")
    from solar_filament.cnn_rescorer import CandidateScorer, candidate_tensors, d4

    root = _ensure_dataset()
    volume.reload()
    out = Path(lab_dir)
    tab = np.load(out / f"instances_thr{threshold}.npz", allow_pickle=True)
    files = list(tab["files"])
    file_dump, props = {}, {}
    for d in dump_dirs.split(","):
        payload = json.loads((Path(d) / "proposals.json").read_text())
        props.update(payload["predictions"])
        file_dump.update({f: d for f in payload["predictions"]})
    cache = out / f"cnn_inputs_thr{threshold}.npy"
    if not cache.is_file():
        arrays = []
        with ProcessPoolExecutor(14) as ex:
            for f, (arr, owner) in zip(files, ex.map(_cnn_build, [(f, file_dump[f], props[f], str(root), threshold) for f in files])):
                arrays.append(arr)
        inputs = np.concatenate(arrays)
        np.save(cache, inputs)
        volume.commit()
    inputs = np.load(cache)
    y, grp, X = tab["y"].astype(np.float32), tab["grp"], tab["X"]
    assert len(inputs) == len(y), (len(inputs), len(y))
    folds = (np.arange(len(files)) % 5)[grp]
    device = torch.device("cuda")

    def train(idx, seed):
        torch.manual_seed(seed)
        model = CandidateScorer().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        steps = epochs * (len(idx) // 128 + 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        xt = torch.from_numpy(inputs)
        yt = torch.from_numpy(y)
        for _ in range(epochs):
            model.train()
            perm = np.random.permutation(idx)
            for b in range(0, len(perm), 128):
                j = perm[b:b + 128]
                xb = d4(xt[j].to(device), np.random.randint(8))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.mse_loss(model(xb).float(), yt[j].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        return model

    @torch.no_grad()
    def predict(model, idx):
        model.eval()
        xt = torch.from_numpy(inputs)
        res = []
        for b in range(0, len(idx), 512):
            xb = xt[idx[b:b + 512]].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res.append(torch.stack([model(d4(xb, k)).float() for k in range(8)]).mean(0).cpu().numpy())
        return np.concatenate(res)

    cnn = np.zeros_like(y)
    for k in range(5):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        cnn[te] = predict(train(tr, k), te)
        print(f"cnn fold {k}: corr {np.corrcoef(cnn[te], y[te])[0, 1]:.3f}", flush=True)
    final = train(np.arange(len(y)), 99)
    torch.save(final.state_dict(), out / f"cnn_rescorer_thr{threshold}.pt")
    np.save(out / f"cnn_oof_thr{threshold}.npy", cnn)

    def lgb_cv(Xm):
        pred = np.zeros_like(y)
        for k in range(5):
            tr = folds != k
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                                  random_state=k).fit(Xm[tr], y[tr])
            pred[~tr] = m.predict(Xm[~tr])
        return pred

    def sweep(pred):
        rows, off = [], 0
        sizes = np.bincount(grp, minlength=len(files))
        res = []
        for tau in np.arange(0.16, 0.42, 0.02):
            tp = fp = fn = 0
            si = 0.0
            off = 0
            for i in range(len(files)):
                n = sizes[i]
                sel = pred[off:off + n] > tau
                for yv, ng in zip(tab["y_views"][i], tab["n_gt"][i]):
                    yv = np.asarray(yv)
                    t = int((yv[sel] > 0).sum())
                    tp += t; fp += int(sel.sum()) - t; fn += ng - t; si += float(yv[sel].sum())
                off += n
            res.append({"tau": round(float(tau), 2), "tp": tp, "fp": fp, "fn": fn,
                        "pq": si / (tp + 0.5 * fp + 0.5 * fn)})
        return res

    report = {}
    for name, Xm in {"features": X, "features+cnn": np.column_stack([X, cnn]),
                     "cnn_only": cnn[:, None]}.items():
        res = sweep(lgb_cv(Xm) if name != "cnn_only" else cnn)
        report[name] = sorted(res, key=lambda r: -r["pq"])[:4]
        print(name, json.dumps(report[name][:2]), flush=True)
    full = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                             random_state=0).fit(np.column_stack([X, cnn]), y)
    full.booster_.save_model(str(out / f"rescorer_cnn_thr{threshold}.txt"))
    (out / f"cnn_report_thr{threshold}.json").write_text(json.dumps(report, indent=2))
    volume.commit()


def _cnn_build(args):
    import cv2
    import numpy as np

    sys.path.insert(0, "/root/src")
    from solar_filament.cnn_rescorer import candidate_tensors

    f, dump_dir, rows, root, threshold = args
    z = np.load(Path(dump_dir) / f"{Path(f).stem}.npz")
    gray = cv2.imread(str(Path(root) / "train" / "train_images" / f), cv2.IMREAD_GRAYSCALE)
    return candidate_tensors(rows, z["prob"], gray, threshold)

@app.function(image=image, gpu=["A10", "L4", "L40S", "A100", "T4"], cpu=8, memory=32768, timeout=2 * 60 * 60,
              volumes={"/data": volume}, secrets=[hf_secret])
def insample_proposals_job(detector_checkpoint: str, output_json: str, image_size: int = 1024):
    """Detector proposals on all 707 training images (for full-data refiner training)."""
    import json

    import numpy as np
    from PIL import Image
    from ultralytics import YOLO

    root = _ensure_dataset()
    coco = json.loads((root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
    files = sorted({im["file_name"] for im in coco["images"]})
    detector = YOLO(detector_checkpoint)
    preds = {}
    for f in files:
        gray = np.asarray(Image.open(root / "train" / "train_images" / f).convert("L"), dtype=np.uint8)
        r = detector.predict(source=np.repeat(gray[:, :, None], 3, axis=2), imgsz=image_size, device=0,
                             conf=0.05, iou=0.7, max_det=80, verbose=False)[0]
        preds[f] = [{"bbox": [float(v) for v in b], "score": float(c)}
                    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())] if len(r.boxes) else []
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps({"predictions": preds}))
    volume.commit()


@app.function(image=image, cpu=2, memory=4096, timeout=10 * 60 * 60, volumes={"/data": volume})
def final_model_job(detector_checkpoint: str, prefix: str = "final707", image_size: int = 1024):
    """Full-data refiner with the k-fold recipe (ImageNet init, 12 epochs, in-sample
    proposals from the full-data detector), then dump the test set."""
    props = f"/data/runs/{prefix}_insample_proposals.json"
    volume.reload()
    if not Path(props).is_file():
        insample_proposals_job.remote(detector_checkpoint=detector_checkpoint, output_json=props,
                                      image_size=image_size)
    crop_refiner_train_job.remote(run_name=f"{prefix}_refiner", epochs=12, batch_size=32, learning_rate=3e-4,
                                  seed=KFOLD_SEED, oof_predictions_path=props, init_checkpoint="",
                                  full_data=True)
    dump_proposals_job.remote(detector_checkpoint=detector_checkpoint,
                              refiner_checkpoint=f"/data/runs/{prefix}_refiner/last.pt",
                              output_dir=f"/data/runs/dump_test_{prefix}", split="test", aux_detectors="",
                              detector_image_size=image_size)
    print(f"final model ready: /data/runs/dump_test_{prefix}", flush=True)


@app.function(image=image, cpu=2, memory=4096, timeout=20 * 60 * 60, volumes={"/data": volume})
def full_member_job(prefix: str, detector_model: str = "yolo11s-seg", image_size: int = 1024, seed: int = 43):
    """One ensemble member on all 707 images: detector, refiner, test dump (k-fold recipe)."""
    det = f"/data/runs/{prefix}_det/train/weights/last.pt"
    volume.reload()
    if not Path(f"/data/runs/{prefix}_det/train/weights/done").is_file() and not Path(
            f"/data/runs/{prefix}_insample_proposals.json").is_file():
        detector_train_job.remote(run_name=f"{prefix}_det", detector_model=detector_model, epochs=60,
                                  image_size=image_size, batch_size=8 if image_size <= 1024 else 4,
                                  full_data=True, seed=seed, lr0=0.002)
    final_model_job.remote(detector_checkpoint=det, prefix=prefix, image_size=image_size)

@app.function(image=image, gpu=["A10", "L4"], cpu=8, memory=32768, timeout=3 * 60 * 60,
              volumes={"/data": volume}, secrets=[hf_secret])
def unet_prob_job(checkpoint: str, output_dir: str, files_json: str = "", split: str = "train"):
    """Save full-resolution U-Net foreground probability maps as uint8 PNGs."""
    import json

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, "/root/src")
    from solar_filament.inference import load_model, predict_probability

    root = _ensure_dataset()
    volume.reload()
    if split == "test":
        paths = sorted((root / "test" / "test_images").glob("*.jp*g"))
        if len(paths) != 180:
            raise RuntimeError(f"Expected 180 test images, found {len(paths)}")
    else:
        paths = [root / "train" / "train_images" / f for f in json.loads(Path(files_json).read_text())]
    device = torch.device("cuda")
    model, _ = load_model(checkpoint, device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        prob = predict_probability(model, gray, device, 1024, 256, 4)
        Image.fromarray(np.clip(prob * 255 + 0.5, 0, 255).astype(np.uint8)).save(out / f"{path.stem}.png")
    volume.commit()
    print(f"Saved {len(paths)} U-Net probability maps to {out}", flush=True)


@app.function(image=image, cpu=2, memory=4096, timeout=8 * 60 * 60, volumes={"/data": volume})
def unet_family_job(epochs: int = 20):
    """5-fold ResNet34 U-Net++ on the kf5 split (held-out probability maps) plus a full-data model."""
    import json

    root = _ensure_dataset()
    full_json = "/data/runs/unet_full707_train_files.json"
    coco = json.loads((root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
    Path(full_json).write_text(json.dumps(sorted({im["file_name"] for im in coco["images"]})))
    volume.commit()
    calls = []
    for k in range(KFOLDS):
        train_files, held = _kfold_files(root, k)
        tj = f"/data/runs/unet_kf5_fold{k}_train_files.json"
        hj = f"/data/runs/unet_kf5_fold{k}_heldout_files.json"
        Path(tj).write_text(json.dumps(train_files))
        Path(hj).write_text(json.dumps(held))
        volume.commit()
        calls.append((k, hj, train_job.spawn(epochs=epochs, patch_size=768, batch_size=4, architecture="resnet34_unetpp",
                                              run_name=f"unet_kf5_fold{k}", seed=KFOLD_SEED + k,
                                              train_files_json=tj, eval_every=100)))
    full = train_job.spawn(epochs=epochs, patch_size=768, batch_size=4, architecture="resnet34_unetpp",
                           run_name="unet_full707", seed=KFOLD_SEED, train_files_json=full_json, eval_every=100)
    probs = []
    for k, hj, c in calls:
        c.get()
        probs.append(unet_prob_job.spawn(checkpoint=f"/data/runs/unet_kf5_fold{k}/last.pt",
                                         output_dir="/data/runs/unet_prob_oof", files_json=hj))
    full.get()
    probs.append(unet_prob_job.spawn(checkpoint="/data/runs/unet_full707/last.pt",
                                     output_dir="/data/runs/unet_prob_test", split="test"))
    for c in probs:
        c.get()
    print("U-Net family done", flush=True)

@app.function(image=image, cpu=2, memory=4096, timeout=10 * 60 * 60, volumes={"/data": volume})
def refiner_variant_job(variant: str = "_r34c384", encoder: str = "resnet34", crop_size: int = 384):
    """Train a refiner variant on every fold (existing fold detectors) and on all data,
    then dump OOF and test proposals with it."""
    det_prefix = {0: "kf5b", 1: "kf5b", 2: "kf5b", 3: "kf5", 4: "kf5"}
    calls = [kfold_fold_job.spawn(fold=k, run_prefix=det_prefix[k], refiner_variant=variant, encoder=encoder,
                                  crop_size=crop_size, detector_prefix=det_prefix[k]) for k in range(KFOLDS)]
    ref = f"final707_refiner{variant}"
    crop_refiner_train_job.remote(run_name=ref, epochs=12, batch_size=32, learning_rate=3e-4, seed=KFOLD_SEED,
                                  oof_predictions_path="/data/runs/final707_insample_proposals.json",
                                  init_checkpoint="", full_data=True, encoder=encoder, crop_size=crop_size)
    dump_proposals_job.remote(detector_checkpoint="/data/runs/full707_yolo11s_s43_lr002/train/weights/last.pt",
                              refiner_checkpoint=f"/data/runs/{ref}/last.pt",
                              output_dir=f"/data/runs/dump_test_final707{variant}", split="test", aux_detectors="")
    for c in calls:
        c.get()
    print("refiner variant done", flush=True)

@app.local_entrypoint()
def main(
    task: str = "train",
    epochs: int = 20,
    patch_size: int = 768,
    batch_size: int = 4,
    samples_per_image: int = 2,
    base_channels: int = 24,
    architecture: str = "small_unet",
    run_name: str = "unet_baseline",
    learning_rate: float = 3e-4,
    seed: int = 42,
    checkpoint: str = "/data/runs/unet_baseline/best.pt",
    ensemble_checkpoint: str = "",
    output_csv: str = "/data/runs/unet_baseline/submission.csv",
    threshold: float | None = None,
    min_area: int | None = None,
    merge_distance: int = 10,
    detector_checkpoint: str = "/data/runs/detector_fusion_yolo11s_seed42/train/weights/best.pt",
    detector_checkpoints: str = "",
    segmentation_checkpoint: str = "/data/runs/resnet34_unetpp_seed42_detached/best.pt",
    detector_eval_image_size: int = 1024,
    detector_disk_margin_px: int | None = None,
    detector_epochs: int = 60,
    detector_model: str = "yolo11s-seg",
    detector_image_size: int = 1024,
    detector_batch_size: int = 8,
    oof_folds: int = 3,
    oof_epochs: int = 60,
    resume_checkpoint: str = "",
    refiner_checkpoint: str = "/data/runs/seed_refiner_resnet18_seed42/best.pt",
    refiner_epochs: int = 10,
    refiner_batch_size: int = 32,
    refiner_learning_rate: float = 3e-4,
    oof_predictions_path: str = "",
    refiner_init_checkpoint: str = "",
    rfdetr_epochs: int = 25,
    rfdetr_resolution: int = 576,
    rfdetr_batch_size: int = 2,
    rfdetr_grad_accum_steps: int = 4,
    ijepa_epochs: int = 10,
    ijepa_patch_size: int = 448,
    ijepa_batch_size: int = 2,
    extra: str = "",
):
    import json as _json
    extra_kw = _json.loads(extra) if extra else {}
    if task == "train":
        train_job.remote(
            epochs=epochs,
            patch_size=patch_size,
            batch_size=batch_size,
            samples_per_image=samples_per_image,
            base_channels=base_channels,
            architecture=architecture,
            run_name=run_name,
            learning_rate=learning_rate,
            seed=seed,
        )
    elif task == "infer":
        inference_job.remote(
            checkpoint=checkpoint,
            output_csv=output_csv,
            threshold=threshold,
            min_area=min_area,
            merge_distance=merge_distance,
            batch_size=batch_size,
        )
    elif task == "sweep":
        validation_sweep_job.remote(
            checkpoint=checkpoint,
            ensemble_checkpoint=ensemble_checkpoint,
        )
    elif task == "detector_train_full":
        call = detector_train_job.spawn(run_name=run_name, detector_model=detector_model, epochs=detector_epochs,
                                        image_size=detector_image_size, batch_size=detector_batch_size,
                                        full_data=True, seed=seed, lr0=learning_rate)
        print(f"Detached full-data detector call ID: {call.object_id}")
    elif task == "detector_train":
        call = detector_train_job.spawn(
            run_name=run_name,
            detector_model=detector_model,
            epochs=detector_epochs,
            image_size=detector_image_size,
            batch_size=detector_batch_size,
            resume_checkpoint=resume_checkpoint,
        )
        print(f"Detached detector call ID: {call.object_id}")
    elif task == "oof_detector_train":
        call = oof_detector_proposal_job.spawn(
            run_name=run_name,
            folds=oof_folds,
            epochs=oof_epochs,
            image_size=detector_image_size,
            batch_size=detector_batch_size,
        )
        print(f"Detached OOF detector training call ID: {call.object_id}")
    elif task == "oof_refiner_pipeline":
        call = oof_refiner_pipeline_job.spawn(
            oof_predictions_path=oof_predictions_path
            or "/data/runs/oof_yolo11s_seed42_3fold/oof_proposals.json",
            run_name=run_name,
            init_checkpoint=refiner_init_checkpoint
            or "/data/runs/seed_refiner_resnet18_seed42/best.pt",
        )
        print(f"Detached OOF refiner pipeline call ID: {call.object_id}")
    elif task == "detector_eval":
        call = detector_fusion_validation_job.spawn(
            detector_checkpoint=detector_checkpoint,
            segmentation_checkpoint=segmentation_checkpoint,
            output_path=f"/data/runs/{run_name}/fusion_validation.json",
            detector_image_size=detector_eval_image_size,
        )
        print(f"Detached detector evaluation call ID: {call.object_id}")
    elif task == "detector_instance_eval":
        call = detector_instance_validation_job.spawn(
            detector_checkpoint=detector_checkpoint,
            output_path=f"/data/runs/{run_name}/instance_validation.json",
            detector_image_size=detector_eval_image_size,
        )
        print(f"Detached native-instance PQ evaluation call ID: {call.object_id}")
    elif task == "detector_infer":
        call = detector_submission_job.spawn(
            detector_checkpoint=detector_checkpoint,
            segmentation_checkpoint=segmentation_checkpoint,
            output_csv=output_csv,
            detector_image_size=detector_eval_image_size,
            disk_margin_px=detector_disk_margin_px,
        )
        print(f"Detached detector test-inference call ID: {call.object_id}")
    elif task == "refiner_train":
        call = crop_refiner_train_job.spawn(
            run_name=run_name,
            epochs=refiner_epochs,
            batch_size=refiner_batch_size,
            learning_rate=refiner_learning_rate,
            seed=seed,
            oof_predictions_path=oof_predictions_path,
            init_checkpoint=refiner_init_checkpoint,
        )
        print(f"Detached crop refiner training call ID: {call.object_id}")
    elif task == "refiner_eval":
        call = crop_refiner_validation_job.spawn(
            refiner_checkpoint=refiner_checkpoint,
            detector_checkpoint=detector_checkpoint,
            output_path=f"/data/runs/{run_name}/validation.json",
            detector_image_size=detector_eval_image_size,
        )
        print(f"Detached crop refiner evaluation call ID: {call.object_id}")
    elif task == "refiner_ensemble_eval":
        call = crop_refiner_validation_job.spawn(
            refiner_checkpoint=refiner_checkpoint,
            detector_checkpoint=detector_checkpoint,
            detector_checkpoints=detector_checkpoints,
            output_path=f"/data/runs/{run_name}/ensemble_validation.json",
            detector_image_size=detector_eval_image_size,
        )
        print(f"Detached crop-refiner detector-ensemble evaluation call ID: {call.object_id}")
    elif task == "refiner_infer":
        call = crop_refiner_submission_job.spawn(
            refiner_checkpoint=refiner_checkpoint,
            detector_checkpoint=detector_checkpoint,
            output_csv=output_csv,
            detector_image_size=detector_eval_image_size,
            detection_threshold=0.35,
            refiner_threshold=0.65,
            min_area=96,
        )
        print(f"Detached crop refiner test-inference call ID: {call.object_id}")
    elif task == "rfdetr_train":
        call = rfdetr_seg_train_job.spawn(
            run_name=run_name,
            epochs=rfdetr_epochs,
            resolution=rfdetr_resolution,
            batch_size=rfdetr_batch_size,
            grad_accum_steps=rfdetr_grad_accum_steps,
        )
        print(f"Detached RF-DETR-Seg training call ID: {call.object_id}")
    elif task == "rfdetr_eval":
        call = rfdetr_seg_validation_job.spawn(
            run_name=run_name,
            rfdetr_checkpoint=f"/data/runs/{run_name}/checkpoint_best_total.pth",
            resolution=rfdetr_resolution,
        )
        print(f"Detached RF-DETR-Seg validation/ensemble call ID: {call.object_id}")
    elif task == "ijepa_train":
        call = ijepa_train_job.spawn(
            run_name=run_name,
            epochs=ijepa_epochs,
            patch_size=ijepa_patch_size,
            batch_size=ijepa_batch_size,
        )
        print(f"Detached I-JEPA ViT-H/14 segmentation training call ID: {call.object_id}")
    elif task == "ijepa_eval":
        call = validation_sweep_job.spawn(
            checkpoint=f"/data/runs/{run_name}/best.pt",
        )
        print(f"Detached I-JEPA segmentation PQ sweep call ID: {call.object_id}")
    elif task == "dump":
        call = dump_proposals_job.spawn(detector_checkpoint=detector_checkpoint,
                                        refiner_checkpoint=refiner_checkpoint,
                                        output_dir=f"/data/runs/{run_name}", split=output_csv, aux_detectors="",
                                        **extra_kw)
        print(f"Detached dump call ID: {call.object_id}")
    elif task == "lab":
        dirs = ",".join(f"/data/runs/{d}" for d in run_name.split(","))
        call = postproc_lab_job.spawn(dump_dir=dirs, then_cnn=ensemble_checkpoint == "cnn")
        print(f"Detached post-processing lab call ID: {call.object_id}")
    elif task == "soft_refiner":
        call = soft_refiner_experiment_job.spawn(run_name=run_name)
        print(f"Detached soft refiner experiment call ID: {call.object_id}")
    elif task == "rescored_submit":
        call = rescored_submission_job.spawn(dump_dir=f"/data/runs/{run_name}", rescorer_path=ensemble_checkpoint,
                                             output_csv=output_csv, tau=threshold or 0.26,
                                             unet_dir=segmentation_checkpoint if segmentation_checkpoint.startswith("/data/runs/unet_prob") else "",
                                             avg_dump_dir=refiner_checkpoint if refiner_checkpoint.startswith("/data/runs/dump_") else "",
                                             **extra_kw)
        print(f"Detached rescored submission call ID: {call.object_id}")
    elif task == "cnn_rescorer":
        dumps = run_name.split(",")
        lab = f"/data/results/postproc_lab/{dumps[0]}" + (f"_x{len(dumps)}" if len(dumps) > 1 else "")
        call = cnn_rescorer_job.spawn(lab_dir=lab, dump_dirs=",".join(f"/data/runs/{d}" for d in dumps))
        print(f"Detached CNN rescorer call ID: {call.object_id}")
    elif task == "full_member":
        call = full_member_job.spawn(prefix=run_name, detector_model=detector_model,
                                     image_size=detector_image_size, seed=seed)
        print(f"Detached full member call ID: {call.object_id}")
    elif task == "final_model":
        call = final_model_job.spawn(detector_checkpoint=detector_checkpoint, prefix=run_name, **extra_kw)
        print(f"Detached final model call ID: {call.object_id}")
    elif task == "script":
        call = lab_script_job.spawn(script=run_name, args=output_csv)
        print(f"Detached script call ID: {call.object_id}")
    elif task == "script_sam":
        call = sam_script_job.spawn(script=run_name, args=output_csv)
        print(f"Detached SAM script call ID: {call.object_id}")
    elif task == "script_gpu":
        call = lab_script_gpu_job.spawn(script=run_name, args=output_csv)
        print(f"Detached GPU script call ID: {call.object_id}")
    elif task == "dense_pool":
        call = dense_pool_job.spawn(tag=run_name)
        print(f"Detached dense pool call ID: {call.object_id}")
    elif task == "unet_family":
        call = unet_family_job.spawn(epochs=epochs)
        print(f"Detached U-Net family call ID: {call.object_id}")
    elif task == "refiner_variant":
        call = refiner_variant_job.spawn()
        print(f"Detached refiner variant call ID: {call.object_id}")
    elif task == "kfold":
        call = kfold_all_job.spawn(run_prefix=run_name, soft_consensus=bool(ensemble_checkpoint == "soft"),
                                   folds=output_csv if "," in output_csv or output_csv.isdigit() else "0,1,2,3,4",
                                   image_size=detector_image_size, detector_model=detector_model)
        print(f"Detached k-fold call ID: {call.object_id}")
    elif task == "full_data":
        call = full_data_pipeline_job.spawn(run_prefix=run_name, seed=seed)
        print(f"Detached full-data pipeline call ID: {call.object_id}")
    else:
        raise ValueError(
            "--task must be train, infer, sweep, detector_train, oof_detector_train, detector_eval, "
            "detector_instance_eval, "
            "detector_infer, refiner_train, refiner_eval, refiner_ensemble_eval, refiner_infer, "
            "rfdetr_train, rfdetr_eval, ijepa_train, or ijepa_eval"
        )
