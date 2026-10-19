"""Classify one cell image, or every image in a folder, as Parasitized or Uninfected.

Uses the custom CNN (outputs\\cnn_best.pt) by default; --model mobilenet uses the MobileNetV2
transfer-learning model instead.

Examples:
    python predict.py path\\to\\image.png
    python predict.py path\\to\\folder
    python predict.py image.png --model mobilenet
    python predict.py image.png --checkpoint outputs\\cnn_best.pt --threshold 0.5
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

import malaria_utils as mu

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    raise SystemExit(f"Not found: {path}")


def main():
    p = argparse.ArgumentParser(description="Predict malaria infection for cell images.")
    p.add_argument("path", help="an image file or a folder of images")
    p.add_argument("--model", choices=sorted(mu.MODEL_LABELS), default=mu.DEFAULT_MODEL,
                   help="cnn = custom CNN (default), mobilenet = MobileNetV2 transfer-learning model")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="model file; overrides --model (default: outputs/<model>_best.pt)")
    p.add_argument("--threshold", type=float, default=None,
                   help="P(Parasitized) needed to call a cell Parasitized "
                        "(default: the recall-tuned threshold stored in the checkpoint)")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    ckpt_path = args.checkpoint or mu.OUTPUT_DIR / f"{args.model}_best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Model not found: {ckpt_path}. Run train.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = mu.load_checkpoint(ckpt_path, device)
    threshold = args.threshold if args.threshold is not None else ckpt.get("threshold", 0.5)
    transform = mu.get_transform(ckpt["img_size"], train=False)

    images = find_images(args.path)
    if not images:
        raise SystemExit("No images found.")
    print(f"Model: {mu.MODEL_LABELS[ckpt['model_name']]} ({ckpt_path.name}), "
          f"threshold {threshold:.3f}, device {device}")
    w = max(len("image"), *(len(p.name) for p in images))  # column width for file names
    print(f"{'image':<{w}} {'prediction':<12} {'confidence':>10} {'P(parasitized)':>15}")

    counts = {name: 0 for name in mu.CLASS_NAMES}
    for start in range(0, len(images), args.batch_size):
        batch_paths, tensors = [], []
        for path in images[start:start + args.batch_size]:
            try:
                with Image.open(path) as img:
                    tensors.append(transform(img.convert("RGB")))
                batch_paths.append(path)
            except Exception as e:
                print(f"{path.name:<{w}} skipped (cannot read image: {e})")
        if not tensors:
            continue
        with torch.inference_mode():
            probs = torch.sigmoid(model(torch.stack(tensors).to(device)).float()).cpu().tolist()
        for path, prob in zip(batch_paths, probs):
            label = mu.CLASS_NAMES[int(prob >= threshold)]
            confidence = prob if label == "Parasitized" else 1 - prob
            counts[label] += 1
            print(f"{path.name:<{w}} {label:<12} {confidence:>10.1%} {prob:>15.4f}")

    if len(images) > 1:
        print(f"\nSummary: {counts['Parasitized']} Parasitized, {counts['Uninfected']} Uninfected")


if __name__ == "__main__":
    main()
