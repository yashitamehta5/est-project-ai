# Task: Malaria Cell Image Classification with a CNN (college deep-learning assignment)

## Problem statement
Biocon Ltd. is working on automated diagnosis of blood samples. Develop a CNN model to classify
red blood cell images as healthy or malaria-infected, supporting faster and more accurate
medical diagnostics.

## Step 0: Inspect before writing any code
The dataset is in this folder (it is too large to share elsewhere). It is most likely the NIH
"Cell Images for Detecting Malaria" dataset (about 27,558 PNG images in two folders,
`Parasitized` and `Uninfected`), but check rather than assume:
- Find the dataset folder(s) and print the folder tree (two levels deep), the number of images
  per class, the file types and the image size range (min, max and median width/height).
- Look for duplicate folders (the Kaggle zip often contains a nested copy such as
  `cell_images/cell_images/`) and non-image files (e.g. `Thumbs.db`). Make sure each image is
  counted only once. (Done: see "Known dataset facts" in CLAUDE.md. Use only
  `cell_images/Parasitized` and `cell_images/Uninfected`.)
- Check whether any images are corrupt or unreadable.
- Check the hardware: GPU (`nvidia-smi`), RAM, Python version, and whether PyTorch is installed
  with a CUDA build that supports the GPU (see "GPU setup" below). Then confirm the image size,
  batch size and number of epochs so that training finishes in a reasonable time on THIS machine.
- Then show me your plan (framework, image size, models, epochs, estimated training time) and
  wait for my OK.

## GPU setup (this machine)
- GPU: NVIDIA RTX 5050 laptop (8 GB, Blackwell, sm_120).
- Use **PyTorch, not TensorFlow**: TensorFlow has no Windows GPU support for the RTX 50-series.
- Install with `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
  The same index URL must be used in `requirements.txt` and `train.bat`.
- Verify the GPU before training: `torch.cuda.is_available()`, `torch.cuda.get_device_name(0)`,
  and a small test tensor operation on the GPU. A "no kernel image is available" or sm_120
  error means the wrong (non-cu128) build is installed.

## What to build
- `malaria_cnn.ipynb`: the main, well-explained notebook (markdown cells for every step).
- `train.py`: the same training as a standalone script, with argparse options `--epochs`,
  `--img-size`, `--batch-size`, `--model {cnn,transfer,both}`, `--quick`.
- `train.bat` (and `train_quick.bat`): Windows double-click launchers. They `cd /d "%~dp0"`,
  create `.venv` on first run, install requirements.txt (with the cu128 PyTorch index),
  run train.py and end with `pause`.
- `predict.py`: classifies a single image or a folder of images
  (`python predict.py path\to\image.png`) and prints the label and the confidence.
- `requirements.txt` (pinned versions), `README.md`, and an `outputs/` folder for models,
  plots and logs.
- Use PyTorch + torchvision. Keep the code readable for a 2nd-year student. Set seeds (42)
  everywhere for reproducibility.

## 1. Data exploration
- Class balance (it should be roughly 50/50), a grid of sample images from each class, and
  histograms of image sizes.
- A short markdown note on what parasitized cells look like (stained purple ring/dot patterns)
  compared with uninfected cells.

## 2. Preprocessing and data pipeline
- Stratified train / validation / test split of 70 / 15 / 15, done on file paths BEFORE any
  training. Save the split lists to CSV so the split is reproducible.
- After splitting, hash the files and verify there are no duplicate images across
  train / val / test.
- Resize to 128×128 and scale pixels to [0, 1] (or use ImageNet normalisation for
  transfer learning).
- Data augmentation on the TRAINING set only: random flips, rotation, zoom and slight brightness
  changes. Cells have no fixed orientation, so flips and rotations are safe. Show a few
  augmented examples.
- Use a PyTorch `Dataset` + `DataLoader` (batch 64, `num_workers=4`, `pin_memory=True`) so the
  whole dataset is never loaded into RAM at once. Training code must be under
  `if __name__ == "__main__":`, which Windows needs for worker processes.
- Note in the notebook: the Kaggle version has no patient IDs, so images from the same patient
  can land in both train and test. This may make the scores slightly optimistic, and it belongs
  in the limitations.

## 3. Models (keep it simple: 2 models)
1. **Custom CNN from scratch** (the main model the brief asks for): 3–4 blocks of
   Conv2d, BatchNorm, ReLU and MaxPool (e.g. 32, 64, 128 and 128 filters), then
   global average pooling, Dropout(0.5), Linear and a single output logit (sigmoid applied for
   the probability). Explain in markdown what each layer type does.
2. **Transfer learning baseline**: MobileNetV2 from torchvision (light enough for a laptop) with
   frozen ImageNet weights and a small head, then optionally fine-tune the top layers. It is there
   for comparison: does a pretrained network beat the custom CNN?
Write 2–3 lines on why each was chosen.

Training settings:
- Binary cross-entropy (`BCEWithLogitsLoss`, which is safe under mixed precision) with the
  Adam optimizer.
- Mixed precision with `torch.autocast` (and a gradient scaler).
- Early stopping (patience 3–5, restore best weights), a ReduceLROnPlateau scheduler, and
  saving the best model checkpoint to `outputs/`.
- Track accuracy, precision, recall and AUC during training.
- Plot the training and validation loss and accuracy curves, and comment on any overfitting.

## 4. Evaluation (on the held-out TEST set only)
- Accuracy, Precision, Recall (sensitivity), Specificity, F1 and ROC-AUC for both models in one
  comparison table, plus a confusion matrix for each and ROC curves on one plot.
- Explain why **recall on the Parasitized class matters most**: a missed infection (false
  negative) is far more dangerous than a false alarm, which would just lead to a re-test.
  Tune the decision threshold to push recall up, and show the precision/recall trade-off.
- Show a grid of misclassified images and comment on why they might be hard (staining
  artefacts, blur, possible labelling errors in the dataset).
- **Grad-CAM** heatmaps (`pytorch-grad-cam`, or a hook-based Grad-CAM) for a few correct and
  incorrect predictions, to show the model looks at the parasite inside the cell and not at the
  background. This is important for a medical use case.
- Expect about 95–97% test accuracy on this dataset. If you get over 99.5%, check for
  duplicate images across the splits before trusting the result.

## 5. Conclusion
- Name the best model and justify the choice (accuracy vs. size and speed).
- State the inference time per image on this machine (relevant for "faster diagnostics").
- Limitations: no patient-level split, a single dataset or lab source, some label noise, and the
  model is a screening aid, not a replacement for a pathologist.
- Possible future work.

## Working style
- Follow Step 0 first, then STOP and show me the plan before writing the training code.
- After the data exploration, tell me anything surprising (duplicates, corrupt files, imbalance)
  before moving on.
- Run everything end to end with `--quick` first to check that it works, then do the full training.
- Finish with: the files created, the training time, the final results table and how to run
  train.bat and predict.py.
