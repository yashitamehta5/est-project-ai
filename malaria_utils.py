"""Shared code for the malaria cell classifier, used by train.py, predict.py and the notebook.

The Dataset and the models live in a normal .py file on purpose: on Windows, DataLoader worker
processes are started with "spawn" and can only use classes they can import from a file
(classes defined inside a notebook cell do not work with num_workers > 0).
"""
import copy
import hashlib
import json
import math
import random
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_DIR = Path(__file__).resolve().parent
# The dataset is ONLY cell_images/Parasitized and cell_images/Uninfected.
# cell_images/cell_images/ is an identical nested copy and must never be read.
DATA_DIR = PROJECT_DIR / "cell_images"
OUTPUT_DIR = PROJECT_DIR / "outputs"
SPLIT_DIR = OUTPUT_DIR / "splits"
CLASS_NAMES = ["Uninfected", "Parasitized"]  # label 0 and label 1 (1 = positive class)
SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- reproducibility
def set_seed(seed=SEED):
    """Seed Python, NumPy and PyTorch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Give each DataLoader worker its own fixed seed (so augmentation is reproducible)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------------------------------------------------------- data
def list_images(data_dir=DATA_DIR):
    """Return a DataFrame with one row per PNG image: relative path and label (0/1).

    Only .png files are taken, so Thumbs.db and other non-image files are skipped.
    """
    rows = []
    for label, class_name in enumerate(CLASS_NAMES):
        for path in sorted((data_dir / class_name).iterdir()):
            if path.suffix.lower() == ".png":
                rows.append({"path": path.relative_to(PROJECT_DIR).as_posix(), "label": label})
    return pd.DataFrame(rows)


def make_splits(df, seed=SEED):
    """Stratified 70 / 15 / 15 split of the file list (done before any training)."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return {name: part.reset_index(drop=True)
            for name, part in [("train", train), ("val", val), ("test", test)]}


def load_or_create_splits(split_dir=SPLIT_DIR):
    """Load the saved split CSVs, or create and save them on the first run."""
    files = {name: split_dir / f"{name}.csv" for name in ["train", "val", "test"]}
    if all(f.exists() for f in files.values()):
        return {name: pd.read_csv(f) for name, f in files.items()}
    splits = make_splits(list_images())
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(files[name], index=False)
    return splits


def file_md5(rel_path):
    return hashlib.md5((PROJECT_DIR / rel_path).read_bytes()).hexdigest()


def check_split_overlap(splits):
    """Hash every file and count identical images shared between the splits.

    Returns a dict like {"train-val": 0, "train-test": 0, "val-test": 0}.
    """
    hashes = {name: set(file_md5(p) for p in part["path"]) for name, part in splits.items()}
    names = list(hashes)
    return {f"{a}-{b}": len(hashes[a] & hashes[b])
            for i, a in enumerate(names) for b in names[i + 1:]}


def stratified_subset(df, n, seed=SEED):
    """A small class-balanced sample of a split, used by --quick."""
    if n >= len(df):
        return df
    return (df.groupby("label", group_keys=False)
              .sample(n=n // 2, random_state=seed)
              .reset_index(drop=True))


class CellDataset(Dataset):
    """Reads one image at a time from disk, so the dataset is never held in RAM."""

    def __init__(self, df, transform):
        self.paths = [PROJECT_DIR / p for p in df["path"]]
        self.labels = df["label"].to_numpy(dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            img = img.convert("RGB")
        return self.transform(img), torch.tensor(self.labels[i])


def get_transform(img_size, train):
    """Resize to img_size x img_size and scale pixels to [0, 1].

    Training images are also augmented. Cells have no fixed orientation, so flips and any
    rotation are safe. The background is black, so the corners filled in by rotation match it.
    """
    steps = [transforms.Resize((img_size, img_size), antialias=True)]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=180, scale=(0.9, 1.1)),  # rotation + zoom
            transforms.ColorJitter(brightness=0.1),                   # slight brightness change
        ]
    steps.append(transforms.ToTensor())  # HWC uint8 [0, 255] -> CHW float [0, 1]
    return transforms.Compose(steps)


def make_loader(df, img_size, batch_size, train, num_workers=4, seed=SEED):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        CellDataset(df, get_transform(img_size, train)),
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),   # faster CPU -> GPU copies
        persistent_workers=num_workers > 0,     # keep workers alive between epochs
        worker_init_fn=seed_worker,
        generator=generator,
    )


# ----------------------------------------------------------------------------- models
class MalariaCNN(nn.Module):
    """Custom CNN: 4 x [Conv -> BatchNorm -> ReLU -> MaxPool], then GAP -> Dropout -> Dense.

    The output is one logit. sigmoid(logit) = probability that the cell is Parasitized.
    """

    def __init__(self, filters=(32, 64, 128, 128), dropout=0.5):
        super().__init__()
        layers, in_ch = [], 3
        for out_ch in filters:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x))).squeeze(1)

    def gradcam_layer(self):
        return self.features[-2]  # ReLU of the last conv block


class MobileNetClassifier(nn.Module):
    """MobileNetV2 (ImageNet weights) with a small new head and one output logit.

    It takes the same [0, 1] images as the custom CNN and applies the ImageNet normalisation
    itself, so both models can share the data loaders, predict.py and the Grad-CAM code.
    """

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.features = base.features  # 19 blocks; the last one outputs 1280 channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(1280, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.classifier(self.pool(self.features(x))).squeeze(1)

    def freeze_base(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_top(self, n_blocks=5):
        for p in self.features[-n_blocks:].parameters():
            p.requires_grad = True

    def train(self, mode=True):
        # Keep BatchNorm layers of frozen blocks in eval mode, so their ImageNet
        # statistics are not overwritten while the rest of the network trains.
        super().train(mode)
        if mode:
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
                    m.eval()
        return self

    def gradcam_layer(self):
        return self.features[-1]


MODEL_LABELS = {"cnn": "Custom CNN", "mobilenet": "MobileNetV2 (transfer)"}


def build_model(name, pretrained=True):
    if name == "cnn":
        return MalariaCNN()
    if name == "mobilenet":
        return MobileNetClassifier(pretrained=pretrained)
    raise ValueError(f"unknown model: {name}")


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def save_checkpoint(path, model, model_name, img_size, threshold=0.5, **extra):
    torch.save({"model_name": model_name, "img_size": img_size, "threshold": threshold,
                "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, device="cpu"):
    """Return (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


# ----------------------------------------------------------------------------- training
def autocast(device):
    """Mixed precision (float16) on the GPU; a no-op on the CPU."""
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=device.type == "cuda")


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(y)
        correct += ((logits > 0).float() == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.inference_mode()
def predict_logits(model, loader, device):
    """Return (logits, labels) for a whole loader as NumPy arrays."""
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        with autocast(device):
            logits = model(x.to(device, non_blocking=True))
        all_logits.append(logits.float().cpu())
        all_labels.append(y)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy().astype(int)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bce_from_logits(logits, labels):
    return nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits), torch.from_numpy(labels.astype(np.float32))).item()


def compute_metrics(labels, probs, threshold=0.5):
    """Metrics with Parasitized (1) as the positive class."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),  # sensitivity
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def threshold_for_recall(labels, probs, target_recall=0.98):
    """Highest threshold (<= 0.5) whose recall is at least target_recall.

    Tuned on the VALIDATION set, never on the test set. Capping it at 0.5 means tuning can
    only raise recall compared with the default threshold.
    """
    _, recall, thresholds = precision_recall_curve(labels, probs)
    ok = thresholds[recall[:-1] >= target_recall]
    best = float(ok.max()) if len(ok) else 0.5
    return min(best, 0.5)


class GpuMonitor:
    """Samples GPU utilisation (%) with nvidia-smi once per second in a background thread."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        while not self._stop.is_set():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def pop_mean(self):
        """Mean utilisation since the last call (NaN if no samples)."""
        samples, self.samples = self.samples, []
        return float(np.mean(samples)) if samples else float("nan")


def fit(model, model_name, train_loader, val_loader, *, epochs, lr, device, ckpt_path, img_size,
        patience=4, stage="train", log=print, gpu_monitor=None, best_val_loss=math.inf):
    """Train with Adam + BCE, ReduceLROnPlateau and early stopping on the validation loss.

    The best weights are saved to ckpt_path and restored at the end. Pass best_val_loss from an
    earlier stage so that a later stage only overwrites the checkpoint if it really improves.
    Returns (history as a list of dicts, best validation loss).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    best_state, bad_epochs, history = None, 0, []

    for epoch in range(1, epochs + 1):
        start = time.time()
        if gpu_monitor:
            gpu_monitor.pop_mean()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler,
                                                loss_fn, device)
        logits, labels = predict_logits(model, val_loader, device)
        val_loss = bce_from_logits(logits, labels)
        m = compute_metrics(labels, sigmoid(logits))
        scheduler.step(val_loss)
        seconds = time.time() - start
        gpu_util = gpu_monitor.pop_mean() if gpu_monitor else float("nan")

        history.append({"stage": stage, "epoch": epoch, "seconds": seconds, "gpu_util": gpu_util,
                        "lr": current_lr, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": m["accuracy"],
                        "val_precision": m["precision"], "val_recall": m["recall"],
                        "val_auc": m["auc"]})
        log(f"[{stage}] epoch {epoch:2d}/{epochs} | {seconds:5.1f}s | GPU {gpu_util:3.0f}% | "
            f"lr {current_lr:.0e} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {m['accuracy']:.4f} prec {m['precision']:.4f} "
            f"rec {m['recall']:.4f} auc {m['auc']:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(ckpt_path, model, model_name, img_size, stage=stage, epoch=epoch,
                            val_loss=val_loss)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"[{stage}] early stopping: no val-loss improvement for {patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # restore the best weights
    else:
        log(f"[{stage}] no improvement over the previous best; keeping the earlier weights")
        model.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"])
    return history, best_val_loss


def measure_end_to_end_time(model, rel_paths, img_size, device):
    """ms per image for the full predict.py path: read PNG, resize, forward pass, one at a time."""
    model.eval()
    transform = get_transform(img_size, train=False)

    def run(paths):
        for p in paths:
            with Image.open(PROJECT_DIR / p) as img:
                x = transform(img.convert("RGB")).unsqueeze(0).to(device)
            torch.sigmoid(model(x)).item()  # .item() waits for the GPU to finish

    with torch.inference_mode():
        run(rel_paths[:20])  # warm-up
        start = time.perf_counter()
        run(rel_paths)
    return (time.perf_counter() - start) / len(rel_paths) * 1000


def measure_inference_time(model, img_size, device, runs=200):
    """Return (ms per image at batch size 1, ms per image at batch size 64).

    Covers only the model's forward pass on preprocessed tensors, not reading the PNG file.
    """
    model.eval()
    results = []
    for batch in (1, 64):
        x = torch.rand(batch, 3, img_size, img_size, device=device)
        n = runs if batch == 1 else max(runs // 10, 10)
        with torch.inference_mode(), autocast(device):
            for _ in range(20):  # warm-up
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        results.append((time.perf_counter() - start) / (n * batch) * 1000)
    return tuple(results)


# ----------------------------------------------------------------------------- results summary
# Everything below reads outputs/results.csv (and run_info.json), so reports never go stale.
README_START = "<!-- RESULTS:START (generated by train.py from outputs/results.csv - do not edit) -->"
README_END = "<!-- RESULTS:END -->"
# Files whose RESULTS block train.py rewrites after a full run.
RESULT_DOCS = [PROJECT_DIR / "README.md", PROJECT_DIR / "PROJECT_OVERVIEW.md"]

# The headline model: the brief asks for a custom CNN, and it is the predict.py default.
# MobileNetV2 is the transfer-learning comparison (predict.py --model mobilenet).
DEFAULT_MODEL = "cnn"


def load_results(out_dir=OUTPUT_DIR):
    """(results DataFrame, run_info dict) written by train.py."""
    results = pd.read_csv(out_dir / "results.csv")
    info_path = out_dir / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return results, info


def pct(x):
    return f"{100 * x:.2f} %"


def paired_bootstrap(out_dir=OUTPUT_DIR, n_boot=1000, seed=SEED):
    """95 % bootstrap intervals for MobileNetV2 minus Custom CNN on the same test images.

    Resamples the test set with replacement and recomputes both models' metrics on each sample,
    so the interval shows whether a difference is larger than test-set sampling noise.
    Returns a DataFrame (metric, threshold, difference, ci_low, ci_high), or None if a model is missing.
    """
    preds, thr = {}, {}
    for name in ["cnn", "mobilenet"]:
        npz, ckpt = out_dir / f"test_predictions_{name}.npz", out_dir / f"{name}_best.pt"
        if not (npz.exists() and ckpt.exists()):
            return None
        preds[name] = np.load(npz)
        thr[name] = torch.load(ckpt, map_location="cpu", weights_only=False)["threshold"]
    y, pc, pm = preds["cnn"]["labels"], preds["cnn"]["probs"], preds["mobilenet"]["probs"]
    if len(y) != len(preds["mobilenet"]["labels"]) or (y != preds["mobilenet"]["labels"]).any():
        return None

    def scores(idx):
        yy = y[idx]
        out = {("ROC-AUC", "-"): roc_auc_score(yy, pm[idx]) - roc_auc_score(yy, pc[idx])}
        for tag, tc, tm in [("0.50", 0.5, 0.5), ("tuned", thr["cnn"], thr["mobilenet"])]:
            a, b = pc[idx] >= tc, pm[idx] >= tm
            out[("Accuracy", tag)] = (b == yy).mean() - (a == yy).mean()
            out[("F1", tag)] = f1_score(yy, b) - f1_score(yy, a)
        return out

    point = scores(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    boots = [scores(rng.integers(0, len(y), len(y))) for _ in range(n_boot)]
    rows = []
    for key, diff in point.items():
        lo, hi = np.percentile([b[key] for b in boots], [2.5, 97.5])
        rows.append({"metric": key[0], "threshold": key[1], "difference": diff,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def bootstrap_markdown(out_dir=OUTPUT_DIR):
    """The paired bootstrap as a markdown table (empty string if a model is missing)."""
    boot = paired_bootstrap(out_dir)
    if boot is None:
        return ""
    lines = ["Is the gap bigger than noise? Paired bootstrap over the test images (1,000 resamples), "
             "MobileNetV2 minus Custom CNN:", "",
             "| Metric | Threshold | Difference | 95 % interval | Verdict |", "|---|---|---|---|---|"]
    for _, r in boot.iterrows():
        scale, unit = (100, " pts") if r.metric == "Accuracy" else (1, "")
        verdict = ("within noise" if r.ci_low <= 0 <= r.ci_high else
                   "MobileNetV2 better" if r.ci_low > 0 else "Custom CNN better")
        lines.append(f"| {r.metric} | {r.threshold} | {scale * r.difference:+.4f}{unit} | "
                     f"{scale * r.ci_low:+.4f} to {scale * r.ci_high:+.4f}{unit} | {verdict} |")
    return "\n".join(lines)


def results_markdown(out_dir=OUTPUT_DIR, with_bootstrap=True):
    """Markdown summary of the latest run: metrics table, epochs, training and inference time."""
    results, info = load_results(out_dir)
    n_test = int(results.iloc[0][["tp", "fp", "tn", "fn"]].sum())
    lines = [f"Test set: {n_test:,} images, Parasitized = positive class. "
             f"Run of {info.get('finished', '?')} on {info.get('gpu', '?')}. "
             "Tuned thresholds were chosen on the validation set only.", "",
             "| Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC "
             "| Missed (FN) | False alarms (FP) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in results.iterrows():
        lines.append(f"| {r.model} | {r.threshold} | {pct(r.accuracy)} | {pct(r.precision)} | "
                     f"{pct(r.recall)} | {pct(r.specificity)} | {r.f1:.4f} | {r.auc:.4f} | "
                     f"{int(r.fn)} | {int(r.fp)} |")
    lines += ["", "| Model | Parameters | Training time | Epochs | ms/image, batch 1 "
              "(model only) | ms/image, batch 64 (model only) | ms/image end to end* |",
              "|---|---|---|---|---|---|---|"]
    for _, r in results.drop_duplicates("model").iterrows():
        lines.append(f"| {r.model} | {r.params_M:.2f} M | {r.train_minutes:.1f} min | "
                     f"{r.epoch_summary} | {r.ms_per_image_batch1:.2f} | "
                     f"{r.ms_per_image_batch64:.3f} | {r.ms_end_to_end:.2f} |")
    lines += ["", "\\* one image at a time, including reading and resizing the PNG (what predict.py does).",
              "", f"Total run time (training + evaluation): {info.get('total_minutes', float('nan')):.1f} min. "
              f"Headline model and `predict.py` default: **{info.get('default_model_label', '?')}**. "
              f"Highest validation AUC: {info.get('best_val_auc_model_label', '?')}."]
    per_model = results.drop_duplicates("model").set_index("model")
    cnn, mob = MODEL_LABELS["cnn"], MODEL_LABELS["mobilenet"]
    if cnn in per_model.index and mob in per_model.index:
        c, m = per_model.loc[cnn], per_model.loc[mob]
        lines += ["", f"Custom CNN vs MobileNetV2: {m.params_M / c.params_M:.1f}× fewer parameters, "
                  f"{m.ms_end_to_end / c.ms_end_to_end:.1f}× faster end to end, "
                  f"{m.ms_per_image_batch1 / c.ms_per_image_batch1:.1f}× faster at batch 1 and "
                  f"{m.ms_per_image_batch64 / c.ms_per_image_batch64:.1f}× faster at batch 64 (model only)."]
    if with_bootstrap:
        boot = bootstrap_markdown(out_dir)
        if boot:
            lines += ["", boot]
    return "\n".join(lines)


def update_results_block(doc_path, out_dir=OUTPUT_DIR):
    """Replace the block between the RESULTS markers in a markdown file with the latest results."""
    doc_path = Path(doc_path)
    if not doc_path.exists():
        return False
    text = doc_path.read_text(encoding="utf-8")
    start, end = text.find(README_START), text.find(README_END)
    if start == -1 or end == -1:
        return False
    new = text[:start] + README_START + "\n" + results_markdown(out_dir) + "\n" + text[end:]
    doc_path.write_text(new, encoding="utf-8")
    return True
