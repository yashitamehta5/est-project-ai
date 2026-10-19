"""Train and evaluate the malaria cell classifiers.

Examples:
    python train.py                 # full training of both models
    python train.py --quick         # small subset, 1-2 epochs: checks everything works
    python train.py --model cnn --epochs 10
"""
import argparse
import datetime
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")  # save plots to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

import malaria_utils as mu


def parse_args():
    p = argparse.ArgumentParser(description="Train malaria cell classifiers (PyTorch).")
    p.add_argument("--model", choices=["cnn", "transfer", "both"], default="both")
    p.add_argument("--epochs", type=int, default=50, help="max epochs for the custom CNN")
    p.add_argument("--frozen-epochs", type=int, default=5,
                   help="MobileNetV2 stage 1: epochs with the pretrained base frozen")
    p.add_argument("--ft-epochs", type=int, default=20,
                   help="MobileNetV2 stage 2: max fine-tuning epochs")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate (CNN and frozen stage)")
    p.add_argument("--ft-lr", type=float, default=1e-4, help="learning rate for fine-tuning")
    p.add_argument("--unfreeze-blocks", type=int, default=5,
                   help="number of top MobileNetV2 blocks to fine-tune")
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--target-recall", type=float, default=0.98,
                   help="recall the tuned decision threshold should reach on validation")
    p.add_argument("--quick", action="store_true",
                   help="tiny subset and 1-2 epochs, results go to outputs/quick/")
    return p.parse_args()


class Logger:
    """Prints a line and also appends it to the training log file."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.file.write(msg + "\n")
        self.file.flush()


def describe_stage(h, max_epochs):
    """e.g. 'stopped early at epoch 23 of 50 (best epoch 18)'. Returns (text, stopped_early)."""
    h = h.reset_index(drop=True)
    n, best = len(h), int(h["val_loss"].idxmin()) + 1
    how = f"stopped early at epoch {n} of {max_epochs}" if n < max_epochs else f"ran all {max_epochs} epoch{'s' * (max_epochs != 1)}"
    return f"{how} (best epoch {best})", n < max_epochs


def plot_history(hist, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(1, len(hist) + 1)
    for ax, key, name in [(axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")]:
        ax.plot(x, hist[f"train_{key}"], "o-", label="train")
        ax.plot(x, hist[f"val_{key}"], "o-", label="validation")
        ax.set_xlabel("epoch")
        ax.set_title(f"{title}: {name}")
        ax.grid(alpha=0.3)
        ax.legend()
        if hist["stage"].nunique() > 1:  # mark where fine-tuning starts
            start = int((hist["stage"] == hist["stage"].iloc[-1]).idxmax()) + 1
            ax.axvline(start - 0.5, color="gray", ls="--")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], mu.CLASS_NAMES)
    ax.set_yticks([0, 1], mu.CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    mu.set_seed(mu.SEED)
    torch.backends.cudnn.benchmark = True  # faster convolutions (tiny run-to-run differences)
    out_dir = mu.OUTPUT_DIR / "quick" if args.quick else mu.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "training_log.txt")
    run_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "none (CPU only)"
    log(f"Malaria cell classification - training run {datetime.datetime.now():%Y-%m-%d %H:%M}")
    log(f"Mode       : {'QUICK test run' if args.quick else 'full training'}")
    log(f"GPU        : {gpu_name}")
    log(f"Software   : Python {platform.python_version()}, PyTorch {torch.__version__}")
    log(f"Settings   : {vars(args)}")
    if device.type != "cuda":
        log("WARNING: no GPU found, training on the CPU will be slow.")

    # ---- data: split once, reuse the saved CSVs, check for leakage
    splits = mu.load_or_create_splits()
    overlap = mu.check_split_overlap(splits)
    log(f"Split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    log(f"Identical files shared between splits (MD5): {overlap}")
    if any(overlap.values()):
        log("WARNING: duplicate images found across splits - test scores may be optimistic.")
    if args.quick:
        sizes = {"train": 2000, "val": 400, "test": 400}
        splits = {k: mu.stratified_subset(v, sizes[k]) for k, v in splits.items()}
        args.epochs, args.frozen_epochs, args.ft_epochs = 2, 1, 1
        log("Quick mode : using " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # Only the training loader gets worker processes. On Windows every worker loads PyTorch's
    # CUDA DLLs and commits about 2.5 GB of virtual memory, so 4 workers each for train, val
    # and test ran out of paging-file space ("WinError 1455"). Val/test load in the main process.
    loaders = {name: mu.make_loader(df, args.img_size, args.batch_size, train=(name == "train"),
                                    num_workers=args.num_workers if name == "train" else 0)
               for name, df in splits.items()}
    if args.num_workers > 0:  # start the persistent workers now, so epoch 1 is timed fairly
        t0 = time.time()
        next(iter(loaders["train"]))  # waits until the workers deliver a first batch
        log(f"Started {args.num_workers} DataLoader workers in {time.time() - t0:.1f}s")

    model_names = {"cnn": ["cnn"], "transfer": ["mobilenet"], "both": ["cnn", "mobilenet"]}[args.model]
    gpu_monitor = mu.GpuMonitor().start() if device.type == "cuda" else None
    results, roc_data = [], {}

    for name in model_names:
        label = mu.MODEL_LABELS[name]
        ckpt_path = out_dir / f"{name}_best.pt"
        log("")
        log(f"===== {label} =====")
        mu.set_seed(mu.SEED)
        model = mu.build_model(name).to(device)
        common = dict(device=device, ckpt_path=ckpt_path, img_size=args.img_size,
                      patience=args.patience, log=log, gpu_monitor=gpu_monitor)
        t0 = time.time()
        if name == "cnn":
            log(f"Parameters: {mu.count_params(model):,}")
            hist, _ = mu.fit(model, name, loaders["train"], loaders["val"], epochs=args.epochs,
                             lr=args.lr, stage="cnn", **common)
        else:
            model.freeze_base()
            log(f"Stage 1 (frozen base, lr {args.lr:g}): trainable parameters "
                f"{mu.count_params(model, True):,} of {mu.count_params(model):,}")
            hist1, best = mu.fit(model, name, loaders["train"], loaders["val"],
                                 epochs=args.frozen_epochs, lr=args.lr, stage="frozen", **common)
            model.unfreeze_top(args.unfreeze_blocks)
            log(f"Stage 2 (fine-tune top {args.unfreeze_blocks} blocks, lr {args.ft_lr:g}): "
                f"trainable parameters {mu.count_params(model, True):,}")
            hist2, _ = mu.fit(model, name, loaders["train"], loaders["val"],
                              epochs=args.ft_epochs, lr=args.ft_lr, stage="finetune",
                              best_val_loss=best, **common)
            hist = hist1 + hist2
        train_minutes = (time.time() - t0) / 60
        hist = pd.DataFrame(hist)
        hist.to_csv(out_dir / f"history_{name}.csv", index=False)
        plot_history(hist, label, out_dir / f"curves_{name}.png")
        log(f"Training time: {train_minutes:.1f} min, mean epoch {hist['seconds'].mean():.1f}s, "
            f"mean GPU utilisation {hist['gpu_util'].mean():.0f}%")
        if name == "cnn":
            epoch_summary, stopped_early = describe_stage(hist, args.epochs)
        else:
            frozen, finetune = hist[hist.stage == "frozen"], hist[hist.stage == "finetune"]
            text1, _ = describe_stage(frozen, args.frozen_epochs)
            text2, stopped_early = describe_stage(finetune, args.ft_epochs)
            epoch_summary = f"frozen: {text1}; fine-tune: {text2}"
            if finetune["val_loss"].min() >= frozen["val_loss"].min():
                epoch_summary += "; fine-tuning did not improve, frozen-stage weights kept"
        log(f"Epochs: {epoch_summary}")

        # ---- tune the threshold on validation, then evaluate ONCE on the test set
        val_logits, val_labels = mu.predict_logits(model, loaders["val"], device)
        val_probs = mu.sigmoid(val_logits)
        threshold = mu.threshold_for_recall(val_labels, val_probs, args.target_recall)
        ckpt = torch.load(ckpt_path, weights_only=False)
        ckpt["threshold"] = threshold
        ckpt["val_auc"] = float(mu.compute_metrics(val_labels, val_probs)["auc"])
        torch.save(ckpt, ckpt_path)

        test_logits, test_labels = mu.predict_logits(model, loaders["test"], device)
        test_probs = mu.sigmoid(test_logits)
        np.savez(out_dir / f"test_predictions_{name}.npz", probs=test_probs, labels=test_labels,
                 paths=splits["test"]["path"].to_numpy())
        roc_data[label] = roc_curve(test_labels, test_probs)[:2]
        ms_single, ms_batch = mu.measure_inference_time(model, args.img_size, device)
        ms_e2e = mu.measure_end_to_end_time(model, splits["test"]["path"].tolist()[:200],
                                            args.img_size, device)

        for thr_name, thr in [("0.50 (default)", 0.5), (f"{threshold:.3f} (tuned)", threshold)]:
            m = mu.compute_metrics(test_labels, test_probs, thr)
            results.append({"model": label, "threshold": thr_name, **m,
                            "params_M": mu.count_params(model) / 1e6,
                            "ms_per_image_batch1": ms_single, "ms_per_image_batch64": ms_batch,
                            "ms_end_to_end": ms_e2e, "train_minutes": train_minutes,
                            "epochs_run": len(hist), "stopped_early": stopped_early,
                            "epoch_summary": epoch_summary, "val_auc": ckpt["val_auc"]})
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            tag = "default" if thr == 0.5 else "tuned"
            plot_confusion(cm, f"{label}\nthreshold {thr:.3f}", out_dir / f"confusion_{name}_{tag}.png")
        log(f"Tuned threshold (val recall >= {args.target_recall}): {threshold:.3f}")
        log(f"Inference: {ms_single:.2f} ms/image at batch 1, {ms_batch:.3f} ms/image at batch 64 "
            f"(model only); {ms_e2e:.2f} ms/image end to end incl. reading the PNG")

    if gpu_monitor:
        gpu_monitor.stop()

    # ---- comparison table, ROC curves, headline model
    table = pd.DataFrame(results)
    table.to_csv(out_dir / "results.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, (fpr, tpr) in roc_data.items():
        auc = table.loc[table["model"] == label, "auc"].iloc[0]
        ax.plot(fpr, tpr, label=f"{label} (AUC {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=120)
    plt.close(fig)

    # The headline / predict.py default is the custom CNN (what the brief asks for); the val-AUC
    # leader is only reported. If the CNN was not trained in this run, fall back to what was.
    best = table.sort_values("val_auc", ascending=False).iloc[0]
    best_name = [k for k, v in mu.MODEL_LABELS.items() if v == best["model"]][0]
    trained = [k for k, v in mu.MODEL_LABELS.items() if v in set(table["model"])]
    default_name = mu.DEFAULT_MODEL if mu.DEFAULT_MODEL in trained else trained[0]

    log("")
    log("===== Test-set results (Parasitized = positive class) =====")
    cols = ["model", "threshold", "accuracy", "precision", "recall", "specificity", "f1", "auc",
            "fn", "fp"]
    log(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log("")
    log(f"Highest validation AUC: {best['model']}")
    log(f"Headline model / predict.py default: {mu.MODEL_LABELS[default_name]} -> "
        f"{out_dir / (default_name + '_best.pt')}")
    total_minutes = (time.time() - run_start) / 60
    log(f"Total run time: {total_minutes:.1f} min")

    # ---- machine-readable run info + generated summaries (README / notebook read these)
    (out_dir / "run_info.json").write_text(json.dumps({
        "finished": f"{datetime.datetime.now():%Y-%m-%d %H:%M}", "gpu": gpu_name,
        "torch": torch.__version__, "quick": args.quick, "settings": vars(args),
        "total_minutes": total_minutes,
        "default_model": default_name, "default_model_label": mu.MODEL_LABELS[default_name],
        "best_val_auc_model": best_name, "best_val_auc_model_label": best["model"]}, indent=2))
    (out_dir / "results.md").write_text(mu.results_markdown(out_dir), encoding="utf-8")
    if not args.quick and args.model == "both":
        for doc in mu.RESULT_DOCS:
            if mu.update_results_block(doc, out_dir):
                log(f"{doc.name} results section updated from outputs/results.csv")


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this file
    main()
