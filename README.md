### Pill Classification (OGYEI) — Deep Learning Course Project

This repository implements an image classifier for pill/medication packages using PyTorch and timm. It follows the assignment pipeline: data loading and preprocessing, model declaration, training/fine‑tuning, and evaluation with per‑class metrics.

- **Goal**: Prototype a quality‑control classifier that recognizes pill classes from images.
- **Target metric**: ≥ 75% accuracy on the validation/test split.
- **Dataset**: OGYEI pill images (Hungarian National Institute of Pharmacy and Nutrition) with multiple lighting/pose conditions.


## Environment and Tooling (uv + ruff)

- Python runtime is defined in `pyproject.toml`.
- Use `uv` for all dependency management and execution.
- Use `ruff` for linting (already configured via `uv`).

Install dependencies:

```bash
uv sync
```

Optional tooling (if you want local linters/tests):

```bash
uv add --dev ruff pytest mypy
```

Run linting:

```bash
uv run ruff check --fix --unsafe-fixes .
```


## Dataset

This project expects the OGYEI dataset arranged under `data/`:

```
data/
  train/
    <class_name>/
      *.jpg
  test/
    <class_name>/
      *.jpg
  class_index.json   # auto-generated
```

- Each class is a folder; images are `.jpg`.
- `class_index.json` is written automatically when running the notebook/script.

Download OGYEI (see the course assignment for the official source) and place the split into `data/train` and `data/test`. Ensure you keep only images in the leaf class folders.


## Project Structure

- `solution.ipynb` — primary solution notebook with all stages and analysis
- `data/` — dataset root (not tracked)
- `pyproject.toml` — dependencies and metadata


## How it Works (high level)

- Custom `PillDataset` enumerates images from class folders.
- Preprocessing: `Resize(224×224) → ToTensor → Normalize(IMAGENET_MEAN, IMAGENET_STD)`.
- Model: `timm` ReXNet‑150 (`rexnet_150`) with the classifier head adapted to `num_classes`.
- Training: freeze backbone, train the head with Adam, mixed precision on CUDA, channels‑last memory format for throughput.
- Checkpointing: best validation loss is saved to `data/meds_classifier.pt` (script) and `data/meds_classifier.pt` plus per‑epoch files (notebook variant).
- Evaluation: accuracy + per‑class Precision/Recall/F1 via `sklearn.classification_report`; optional confusion matrices.


## Run the Notebook

Launch Jupyter and open `solution.ipynb`:

```bash
uv run jupyter notebook
```

Execute cells in order. The notebook will:
- Build `train_loader`, `val_loader`, and a `test_loader` over `data/test`.
- Create the ReXNet‑150 model with a trainable head.
- Train for `EPOCHS` (default 30 in the notebook) and save checkpoints under `data/`.
- Evaluate on the test split and print the report, including macro averages.

Artifacts:
- `data/pill_classifier_<EPOCH>_epoch.pt` — per‑epoch state dicts (best by val loss)
- `data/meds_classifier.pt` — latest best model (for assignment compliance)


## Configuration Knobs

Edit these constants in the notebook/script to change behavior:

- `DATA_ROOT`, `TRAIN_DATA_ROOT`, `TEST_DATA_ROOT`
- `DEV_MODE` and caps (`DEV_MAX_TRAIN_SAMPLES`, etc.)
- `VAL_SPLIT_FRACTION`
- `IM_SIZE`, `MEAN`, `STD`
- `BATCH_SIZE`, `NUM_WORKERS`, `PIN_MEMORY`, `PREFETCH_FACTOR`
- `MODEL_NAME` (e.g., `rexnet_150`), `LEARNING_RATE`, `EPOCHS`


## Tips to Reach ≥75% Accuracy

- Unfreeze later backbone blocks and fine‑tune with a lower LR.
- Use stronger augmentations (RandAugment/TrivialAugment, ColorJitter, RandomResizedCrop, RandomErasing).
- Address imbalance: class‑weighted loss or `WeightedRandomSampler`; oversample rare classes.
- Regularization: label smoothing, mixup/cutmix; cosine LR schedule with warmup.
- Increase input size (e.g., 288–320) if VRAM allows; try alternative backbones (ConvNeXt, EfficientNetV2, ViT, Swin).
- Test‑time augmentation (TTA) and ensembling for extra points.


## Reproducing Evaluation

After training, the notebook prints:
- Overall accuracy
- Full per‑class classification report
- Macro averages (Precision/Recall/F1)
- Optional confusion matrices

Example callouts you should extract from your own run:
- Top‑5 classes with lowest recall (most errors) and hypotheses why
- Classes with perfect recall/precision and why
- Concrete improvement ideas you will try next


## Citation and Sources

- Dataset: OGYEI (see `assignment/assignment.md` for link and details)
- Libraries: PyTorch, TorchVision, timm, scikit‑learn, matplotlib, Pillow


## License

This project is for educational purposes within a deep learning course. Dataset usage is subject to its original license/terms.
