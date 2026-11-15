# Solar filament instance segmentation (version 7)

Solution to the Solar Filament Segmentation Challenge 2026 (MAGFiLO, GONG H-alpha full-disk images). Out-of-fold Panoptic Quality 0.4626 on the 707 training images, 0.41 on the public leaderboard.

| Item | Where |
|---|---|
| Technical report | `report/report.pdf` (source in `report/`) |
| Notebook with the entire pipeline | `notebooks/solar_filament_pipeline.ipynb` |
| Model weights | `weights/` (twelve files, 1.4 GB, Git LFS, hashes in `weights/MANIFEST.json`) |
| File submitted to Kaggle | `submission.csv` |
| Packages and versions | `requirements.txt` |
| Inference code | `src/solar_filament/pipeline.py` (`FilamentPipeline`) |
| Training and experiment code | `modal_app.py`, `scripts/` |
| Everything that was tried, with numbers and logs | `experiments/` (start at `experiments/README.md`) |

## Reproduce the predictions

```bash
git lfs install
git clone https://github.com/omkar-334/solar-filament-segmentation.git && cd solar-filament-segmentation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/solar_filament_pipeline.ipynb
```

The notebook loads the weights from `weights/`, runs every stage on three test images stored in `demo_images/`, compares the result with `submission.csv` and shows the validation statistics. No training is needed and no file outside this repository is needed for that.

For the full test set, put the competition data in `MAGFiLO_1.0_Kaggle_2026/` and run:

```bash
python scripts/predict.py --images=MAGFiLO_1.0_Kaggle_2026/test/test_images --weights=weights --out=submission_repro.csv --compare=submission.csv --tag=repro
```

This writes the CSV and a comparison with the submitted file to `results/predict/repro/summary.json`. The copy of that log in this repository comes from a clean environment built from `requirements.txt` on one GPU (`results/verify_release/`). Speed: about 1.2 s per image on one GPU, about 50 s on a CPU.

Python 3.11 was used. On a minimal Linux image OpenCV needs two system libraries: `apt-get install -y libgl1 libglib2.0-0`. If the weight files are a few hundred bytes each, Git LFS did not fetch them: run `git lfs pull`.

## Pipeline

1. Proposals. Two YOLO11s-seg detectors (1024 px and 1536 px) propose filament boxes with confidence 0.05 or more.
2. Crop refinement. A box-seeded U-Net segments each box in its own crop (ResNet18 at 256 px and ResNet34 at 384 px averaged for the 1024 px detector, ResNet18 at 256 px for the 1536 px detector, four flips).
3. Paste. Candidates are pasted in detector-score order with pixel claiming (threshold 0.65, minimum area 96 px), so instances never overlap.
4. Selection. A LightGBM model predicts the expected matched IoU of every pasted instance. Instances at or below 0.24 are dropped.
5. Stage 2. Every kept instance is segmented again in a rotation-normalised 384 x 128 crop by a ResNet34 U-Net (five fold models, four flips, threshold 0.40).

A heavier variant, version 11 (documented in `experiments/v11/`), replaces stage 2 by fine-tuned SAM 2.1 models. It scores 0.4725 out of fold and is about eleven times slower.

## Data and rules

Only the training split of the competition data was used for training. No external labels, no public MAGFiLO test labels and no hard-coded predictions were used. Pretrained initialisations are public (COCO YOLO11s-seg, ImageNet ResNets). The images in `demo_images/` are three GONG H-alpha test images of the competition data (MAGFiLO, CC BY-NC 4.0, NSO Integrated Synoptic Program).

## Layout

| Path | Content |
|---|---|
| `src/solar_filament/` | Library: data, models, post-processing, pipeline |
| `scripts/` | Prediction, training stages, validation labs, metrics. Every script logs to `results/<script name>/` |
| `tests/` | Checks of the metric code |
| `results/` | Logs and summaries cited in the report |
| `report/` | Technical report (PDF and LaTeX source) |
| `experiments/` | Long-form report, dated experiment log, rejected ideas, and the heavier SAM 2.1 variant (v11) |
