# Project Overview: Malaria Cell Classification

A deep-learning screening aid that looks at one segmented red blood cell image and says whether it is
**Parasitized** (infected with the malaria parasite) or **Uninfected**. The headline model is a
**custom CNN** trained from scratch, which is what the brief ([TASK.md](TASK.md)) asks for. A fine-tuned
**MobileNetV2** is trained alongside it as the transfer-learning comparison.

Contents:
1. [Files: what each one does and why it exists](#1-files)
2. [Setup and how to run](#2-setup-and-how-to-run)
3. [Which files to run for the final result](#3-which-files-to-run-for-the-final-result)
4. [Report](#4-report): data, preprocessing, models, results, Grad-CAM, limitations

---

## 1. Files

### Code

| File | What it does | Why it exists |
|---|---|---|
| `malaria_utils.py` | Shared library: paths, the 70/15/15 split and hash check, `CellDataset`, transforms, both models, the `fit()` training loop (mixed precision, early stopping, learning-rate schedule, checkpointing), metrics, threshold tuning, GPU monitor, timing, and the results-summary generator. | Every script and the notebook import the same code, so they can't drift apart. On Windows, DataLoader worker processes can only load a Dataset class from an importable `.py` file, not from a notebook cell. |
| `train.py` | Runs a full experiment: split → train the CNN → train MobileNetV2 (frozen, then fine-tuned) → tune each threshold on **validation** → evaluate once on **test** → write everything to `outputs/` and refresh the results blocks in `README.md` and this file. | One command reproduces every number in the report. `--quick` is a 2-minute end-to-end smoke test. |
| `predict.py` | Classifies one image or a whole folder and prints label, confidence and P(Parasitized). Uses the custom CNN by default; `--model mobilenet` switches model, `--threshold` overrides the tuned threshold. | The "use the model" entry point, for new images outside the dataset. |
| `check_gpu.py` | Prints the PyTorch and CUDA versions and the GPU name, then runs a test matrix multiply and an fp16 convolution on the GPU. | The RTX 5050 (Blackwell, sm_120) only works with the CUDA 12.8 PyTorch build. This catches a wrong install before a long training run. |
| `malaria_cnn.ipynb` | The written report: data exploration, pipeline, model design, training curves, evaluation, threshold analysis, error images, Grad-CAM and the conclusion. It loads the trained models from `outputs/` (it only trains if `RETRAIN = True`). | The deliverable for marking. All numbers in it are computed from `outputs/`, so they always match the latest run. |

### Launchers and setup

| File | What it does | Why it exists |
|---|---|---|
| `train.bat` | Double-click launcher. Creates `.venv` if missing, installs the requirements (PyTorch from the cu128 index), runs `check_gpu.py`, then `train.py` (any arguments are passed through). | One-click setup and training on Windows, without touching the global Python install. |
| `train_quick.bat` | Calls `train.bat --quick`. | The same, but a quick test run on a small subset (results in `outputs\quick\`). |
| `requirements.txt` | Pinned package versions, including `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` with the cu128 index URL. | Reproducible environment. The global Python has a CPU-only PyTorch, so the venv must get the CUDA build. |

### Documentation

| File | Contents |
|---|---|
| `README.md` | Short summary, latest results (generated), quick start, options. |
| `PROJECT_OVERVIEW.md` | This file. |
| `TASK.md` | The original assignment brief (problem statement, steps, deliverables). |
| `CLAUDE.md` | Notes for the AI coding assistant: environment, dataset facts, conventions. |

### Data and outputs

| Path | Contents |
|---|---|
| `cell_images/Parasitized/`, `cell_images/Uninfected/` | The dataset: 13,779 PNGs per class. **These are the only folders read.** |
| `cell_images/cell_images/` | A byte-identical nested copy from the Kaggle zip. It is **never read**, because using it would put the same images in train and test. |
| `outputs/splits/train.csv`, `val.csv`, `test.csv` | The fixed stratified split (seed 42), reused by every run. |
| `outputs/cnn_best.pt`, `outputs/mobilenet_best.pt` | Trained models. Each stores its weights, image size and validation-tuned threshold. |
| `outputs/results.csv` | Test metrics for both models at 0.5 and at the tuned threshold, plus timing and epoch info. **The single source for every reported number.** |
| `outputs/run_info.json`, `outputs/results.md` | Run settings, GPU, total time and headline model; the generated results tables. |
| `outputs/training_log.txt` | Per-epoch log: time, GPU utilisation, learning rate, train/validation loss and metrics, final results. |
| `outputs/history_*.csv`, `curves_*.png`, `confusion_*.png`, `roc_curves.png` | Training histories and plots. |
| `outputs/test_predictions_*.npz` | Test-set probabilities, used by the notebook and the bootstrap comparison. |
| `outputs/quick/` | Output of the `--quick` smoke test (not the final result). |

---

## 2. Setup and how to run

Requirements: Windows, Python 3.11 (`py -3.11`), an NVIDIA GPU with a current driver. Training also works
on CPU, just much more slowly.

### One-click (recommended)

1. **`train_quick.bat`**: double-click it first. On the first run it creates `.venv` and installs
   everything (a few minutes, PyTorch is about 3 GB). Then it checks the GPU and runs a small training run
   in about 2 minutes. If this finishes without errors, the setup works.
2. **`train.bat`**: double-click for the full training of both models (roughly 10–15 minutes on
   the RTX 5050 laptop GPU). This produces the final models and results in `outputs\`.

### Manual

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python check_gpu.py          :: must print "CUDA ready" and the GPU name
.venv\Scripts\python train.py --quick      :: smoke test -> outputs\quick\
.venv\Scripts\python train.py              :: full run, both models -> outputs\
```

Useful `train.py` options: `--model {cnn,transfer,both}`, `--epochs 50` (CNN), `--frozen-epochs 5`,
`--ft-epochs 20` (MobileNetV2 fine-tuning), `--patience 5`, `--lr 1e-3`, `--ft-lr 1e-4`,
`--target-recall 0.98`, `--num-workers 4`. Only a full `--model both` run updates the README and this file.

### Classify images: `predict.py`

```bat
.venv\Scripts\python predict.py path\to\cell.png                 :: custom CNN, tuned threshold
.venv\Scripts\python predict.py path\to\folder                   :: every image in a folder + summary
.venv\Scripts\python predict.py path\to\cell.png --model mobilenet
.venv\Scripts\python predict.py path\to\cell.png --threshold 0.5
```

### Open the report notebook

```bat
.venv\Scripts\python -m notebook malaria_cnn.ipynb
```

Run all cells (it takes about a minute and does not retrain). Use `python -m notebook` from the venv:
`.venv\Scripts\jupyter` can start the global Jupyter, which has the CPU-only PyTorch and no `grad-cam`.
In VS Code, select `.venv` as the kernel. To re-execute it non-interactively:

```bat
.venv\Scripts\python -m nbconvert --to notebook --execute --inplace malaria_cnn.ipynb
```

If `check_gpu.py` reports *"no kernel image is available"* or an sm_120 error, the wrong PyTorch build is
installed. Reinstall it from the cu128 index (command above).

---

## 3. Which files to run for the final result

| Step | Run | Produces |
|---|---|---|
| 1 | `train.bat` (or `.venv\Scripts\python train.py`) | `outputs\cnn_best.pt`, `outputs\mobilenet_best.pt`, `results.csv`, plots, `training_log.txt`; updates the results in `README.md` and here |
| 2 | `.venv\Scripts\python -m notebook malaria_cnn.ipynb` → Run All (or the `nbconvert` command above) | The executed report notebook with all figures, Grad-CAM and the conclusion |
| 3 (optional) | `.venv\Scripts\python predict.py <image or folder>` | Predictions for new images |

`train_quick.bat` / `--quick` is only a test of the setup; its results in `outputs\quick\` are not the final result.

---

## 4. Report

### 4.1 Data

* **Source:** NIH *Cell Images for Detecting Malaria* (Kaggle). Thin blood-smear images, already
  segmented into single cells: 13,779 Parasitized and 13,779 Uninfected PNGs (27,558 in total, perfectly
  balanced).
* **Checks before training:** every image opens and is RGB (none corrupt). No exact duplicates exist (MD5
  over all files). Sizes vary from 40 to 394 px (median 130 × 130). `Thumbs.db` files and the nested copy
  `cell_images/cell_images/` are excluded.
* **Split:** stratified 70 / 15 / 15 → 19,290 train, 4,134 validation, 4,134 test images, seed 42, saved
  to CSV and reused. After splitting, every file is hashed and the check confirms **no identical image
  appears in two splits**.
* **Roles:** train = fitting the weights; validation = early stopping, learning-rate schedule and
  **threshold tuning**; test = used once, only for the final numbers.

### 4.2 Preprocessing

* **Resize to 128 × 128.** The models need a fixed size. Some cells are only 40 px, so upscaling (up to about
  3×) is unavoidable and blurs small parasites a little.
* **Scale pixels to [0, 1].** MobileNetV2 additionally applies the ImageNet mean/std normalisation inside the
  model, so both models take the same input.
* **Augmentation (training only):** horizontal and vertical flips, rotation by any angle (cells have no
  orientation, and the black corners rotation creates match the black background), zoom ±10 %, brightness
  ±10 %.
* **Loading:** PyTorch `DataLoader`, batch 64, `pin_memory=True`, 4 persistent worker processes for the
  training loader. Validation and test load in the main process, because each Windows worker reserves about
  2.5 GB of virtual memory for the CUDA libraries.
* **Mixed precision:** `torch.autocast` (float16) with `BCEWithLogitsLoss` and a gradient scaler.

### 4.3 Models

**Custom CNN (headline model), about 0.25 M parameters**

```
input 3×128×128
4 × [Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2]   filters 32, 64, 128, 128
Global average pooling → Dropout 0.5 → Dense 64 + ReLU → Dense 1 (logit)
```

Trained with Adam (learning rate 1e-3), up to 50 epochs. The learning rate halves after 2 epochs without
validation-loss improvement, and training stops early after 5. The weights with the lowest validation
loss are kept.

**MobileNetV2 (transfer-learning comparison), about 2.2 M parameters**

ImageNet-pretrained MobileNetV2 from torchvision, with a new head (global average pooling → Dropout 0.3 →
Dense 1). It is trained in two stages:
1. **Frozen base**, 5 epochs, learning rate 1e-3: only the new head learns.
2. **Fine-tuning**, up to 20 epochs, learning rate 1e-4: the top 5 blocks are unfrozen. The lower learning
   rate avoids destroying the pretrained features. BatchNorm layers in the still-frozen blocks stay in
   eval mode.

**Decision threshold.** Missing an infected cell is much worse than a false alarm (a false alarm only
means a human re-check). So besides the default 0.5, each model gets a **tuned threshold**: the highest
threshold (at most 0.5) that reaches 98 % recall **on the validation set**. It is then applied unchanged
to the test set. The notebook re-checks this on every run: repeating the search on validation reproduces
the stored thresholds exactly, and the same search on the test set would have given lower values, so the
test set was not used for tuning.

### 4.4 Results

The block below is generated by `train.py` from `outputs/results.csv` after every full run. Do not edit
it by hand.

<!-- RESULTS:START (generated by train.py from outputs/results.csv - do not edit) -->
Test set: 4,134 images, Parasitized = positive class. Run of 2026-10-18 16:25 on NVIDIA GeForce RTX 5050 Laptop GPU. Tuned thresholds were chosen on the validation set only.

| Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Missed (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|---|---|
| Custom CNN | 0.50 (default) | 96.57 % | 97.34 % | 95.74 % | 97.39 % | 0.9654 | 0.9927 | 88 | 54 |
| Custom CNN | 0.176 (tuned) | 95.55 % | 93.89 % | 97.44 % | 93.66 % | 0.9563 | 0.9927 | 53 | 131 |
| MobileNetV2 (transfer) | 0.50 (default) | 96.15 % | 97.70 % | 94.53 % | 97.77 % | 0.9609 | 0.9937 | 113 | 46 |
| MobileNetV2 (transfer) | 0.281 (tuned) | 96.52 % | 95.89 % | 97.19 % | 95.84 % | 0.9654 | 0.9937 | 58 | 86 |

| Model | Parameters | Training time | Epochs | ms/image, batch 1 (model only) | ms/image, batch 64 (model only) | ms/image end to end* |
|---|---|---|---|---|---|---|
| Custom CNN | 0.25 M | 7.1 min | stopped early at epoch 46 of 50 (best epoch 44) | 0.74 | 0.072 | 1.44 |
| MobileNetV2 (transfer) | 2.23 M | 3.8 min | frozen: ran all 5 epochs (best epoch 4); fine-tune: stopped early at epoch 18 of 20 (best epoch 13) | 5.00 | 0.111 | 4.72 |

\* one image at a time, including reading and resizing the PNG (what predict.py does).

Total run time (training + evaluation): 12.0 min. Headline model and `predict.py` default: **Custom CNN**. Highest validation AUC: MobileNetV2 (transfer).

Custom CNN vs MobileNetV2: 8.9× fewer parameters, 3.3× faster end to end, 6.8× faster at batch 1 and 1.5× faster at batch 64 (model only).

Is the gap bigger than noise? Paired bootstrap over the test images (1,000 resamples), MobileNetV2 minus Custom CNN:

| Metric | Threshold | Difference | 95 % interval | Verdict |
|---|---|---|---|---|
| ROC-AUC | - | +0.0010 | -0.0003 to +0.0025 | within noise |
| Accuracy | 0.50 | -0.4112 pts | -0.8466 to +0.0490 pts | within noise |
| F1 | 0.50 | -0.0045 | -0.0089 to +0.0003 | within noise |
| Accuracy | tuned | +0.9676 pts | +0.4112 to +1.5481 pts | MobileNetV2 better |
| F1 | tuned | +0.0091 | +0.0037 to +0.0148 | MobileNetV2 better |
<!-- RESULTS:END -->

**How to read it**

* Both models land in the 95–97 % accuracy range expected for this dataset, well below the >99.5 % that
  would suggest leakage.
* **Custom CNN vs MobileNetV2.** ROC-AUC and the 0.5-threshold scores are within test-set noise (see the
  bootstrap table). ImageNet pretraining adds little, because the key visual cue (a compact purple-stained
  dot inside the cell) is easy to learn from 19,000 images. The custom CNN is much smaller and faster, which
  is why it is the headline model.
* **Where MobileNetV2 is better:** at the recall-tuned thresholds it reaches similar recall with fewer false
  alarms, and the bootstrap shows this gap is larger than noise. For a lab where the re-check workload
  matters more than model size, `predict.py --model mobilenet` is the better option.
* **Threshold tuning** clearly reduces the missed infections for both models (see the FN column), at the cost of more false
  alarms. Test recall lands slightly below the 98 % target set on validation, which is normal when a
  threshold moves to new data.

### 4.5 Grad-CAM findings

Grad-CAM highlights the image regions that pushed the model towards its prediction (computed on the last
convolutional block of each model; examples are in section 4.3 of the notebook).

* **Custom CNN.** For correctly detected infections, the heatmap is a sharp hot spot **on the purple
  parasite**, ignoring the rest of the cell and the background. That is the behaviour a medical model
  should have. For healthy cells, the evidence for "Uninfected" is spread over the cell body and border.
  False alarms tend to have their hot spot on the **cell edge**: purple material just outside the cell, or
  jagged edges left by the segmentation. The model reacts to "small dark object" without checking that it
  lies inside the cell.
* **MobileNetV2.** Its last feature map is only 4 × 4 for a 128 × 128 input, so the heatmaps are coarse
  blobs. For clearly infected cells they still cover the parasite. For false alarms they tend to point at
  faint purple patches near the cell edge.
* **No shortcut learning seen:** neither model focuses mainly on the black background or the image corners,
  which would have suggested it was using image size or crop shape instead of the cell itself.

*These observations come from the example images in the notebook. They are written by hand, so re-check
them after retraining.*

### 4.6 Limitations

* **Label noise.** Among the most confident "errors" are cells labelled Uninfected that clearly show a
  compact stained dot or ring, and cells labelled Parasitized that look completely uniform. These look like
  labelling mistakes in the dataset, which has been reported for this dataset before. Label noise caps the
  accuracy any model can reach, and it makes the measured accuracy a slight underestimate (correct
  predictions on mislabelled cells count as errors).
* **No patient-level split.** The Kaggle release has no patient or slide IDs, so cells from the same
  patient or slide end up in train, validation and test. The model has effectively seen each patient's
  staining and imaging conditions before, so **the test scores are probably optimistic for new patients**.
  The hash check only rules out *identical* images, not near-identical neighbouring cells.
* **One data source.** All slides come from one collection (one lab, staining protocol and microscope).
  Performance on other labs' slides is unknown and probably lower without retraining or stronger
  colour augmentation.
* **Single segmented cells only.** A real system also needs cell detection on the whole slide and a
  slide-level decision. The dataset does not label parasite species or parasitemia.
* **Resizing.** Small cells are upscaled up to about 3×, which blurs small or faint parasites.
* **Screening aid, not a diagnosis.** Even at the tuned threshold a few percent of infected cells are
  missed. Positive or uncertain results must be confirmed by a trained microscopist.

### 4.7 Future work

* A patient-level split (if IDs can be recovered from the original NIH release) with k-fold
  cross-validation.
* Expert review of the confidently wrong images to clean the labels, then retrain.
* Test-time augmentation, probability calibration, and evaluation on external datasets.
* A full-slide pipeline (detect cells → classify → parasite count per slide), and export to ONNX or
  TorchScript for a low-power device next to the microscope.
