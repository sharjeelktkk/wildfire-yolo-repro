# Reproduction of "YOLO-Based Models for Smoke and Wildfire Detection in Ground and Aerial Images"

Deep Learning – Assignment 02 (baseline reproduction) · Sharjeel Tahir, 01-134232-235, BSCS-7A

| | |
|---|---|
| **Baseline paper** | L. A. O. Gonçalves, R. Ghali, M. A. Akhloufi, *Fire* 7(4):140, 2024. DOI [10.3390/fire7040140](https://doi.org/10.3390/fire7040140) · [full text](https://www.mdpi.com/2571-6255/7/4/140) |
| **Task** | Object detection of smoke and fire (YOLO family) |
| **Datasets** | D-Fire (smoke, fire) · WSDY – WildFire Smoke Dataset YOLO (smoke) |
| **Reproduced models** | YOLOv8n, YOLOv8s on both datasets (YOLOv7x optional, see below) |
| **Framework** | Python, PyTorch, Ultralytics |

> **Status of results:** `results/` and `figures/` are filled by running the pipeline below.
> Nothing in this repository claims exact reproduction – the comparison table reports the measured
> differences against the paper, and `Deviations` lists every assumption.

## 1. Repository layout
```
configs/baseline.yaml        all hyper-parameters, each tagged [PAPER] or [ASSUME]
src/prepare_dataset.py       Task 1: split, statistics, figures for D-Fire and WSDY
src/train.py                 Task 2-3: YOLO baseline training (+ environment/time logging)
src/evaluate.py              Task 4: test-set metrics (P, R, mAP@0.5, mAP@0.5:0.95, per-class AP, GFLOPs, latency)
src/make_figures.py          training curves + original-vs-reproduced comparison table
notebooks/reproduce_baseline.ipynb   end-to-end Colab / Kaggle notebook
paper_reference/paper_results.csv    numbers reported by the original authors
results/  figures/           outputs (JSON, CSV, markdown table, PNG)
```

## 2. Setup
```bash
git clone <your-repo-url> && cd wildfire-yolo-repro
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```
A CUDA GPU is strongly recommended (Colab, Kaggle, or local). Software/hardware details are
written automatically to `results/<run>_train_info.json` and `runs/<run>/experiment_config.json`.

## 3. Datasets (not stored in this repo)
* **D-Fire** – 21,527 images (1,164 fire only, 5,867 smoke only, 4,658 both, 9,838 negatives).
  Download from the authors' page: <https://github.com/gaiasd/DFireDataset> (Google Drive link in the README).
  Extract to `data/D-Fire/{train,test}/{images,labels}` (a `det_labels` folder is also accepted).
* **WSDY** – 737 smoke images, YOLO format: <https://www.kaggle.com/datasets/ahemateja19bec1025/wildfiresmokedatasetyolo>.
  Extract to `data/WSDY/`.
* Cite D-Fire as Venâncio et al., *Neural Comput. Appl.* 34, 15349–15368 (2022), doi:10.1007/s00521-022-07467-z.

## 4. Run the experiment
```bash
# Task 1 – split + statistics + figures
python src/prepare_dataset.py dfire --root data/D-Fire --out data/prepared
python src/prepare_dataset.py wsdy  --root data/WSDY   --out data/prepared

# Task 2/3 – train (100 epochs, batch 16, 640 px, SGD lr 0.01, COCO-pretrained)
python src/train.py --data data/prepared/dfire/dfire.yaml --model yolov8n.pt --name dfire_yolov8n
python src/train.py --data data/prepared/dfire/dfire.yaml --model yolov8s.pt --name dfire_yolov8s
python src/train.py --data data/prepared/wsdy/wsdy.yaml  --model yolov8n.pt --name wsdy_yolov8n
python src/train.py --data data/prepared/wsdy/wsdy.yaml  --model yolov8s.pt --name wsdy_yolov8s
#   interrupted?  add --resume          reduced budget?  add --epochs 30 (and report it)

# Task 4 – evaluate on the held-out TEST split (run once per model)
python src/evaluate.py --data data/prepared/dfire/dfire.yaml --weights runs/dfire_yolov8s/weights/best.pt --name dfire_yolov8s

# curves + comparison table (results/comparison_table.md)
python src/make_figures.py curves --run runs/dfire_yolov8s --name dfire_yolov8s
python src/make_figures.py compare
```
Run names must follow `<dataset>_<model>` (e.g. `dfire_yolov8s`) so the comparison script can match them to
`paper_reference/paper_results.csv`.

**Compute guide.** The paper used a Tesla V100 and reports ≈2.0 h (YOLOv8n) and ≈2.8 h (YOLOv8s) on D-Fire,
≈0.2 h on WSDY. A free Kaggle/Colab GPU is typically slower, so expect several hours for D-Fire; WSDY is quick.
Kaggle allows ~12 h per session; on Colab use `--resume` and save `runs/` to Google Drive.

## 5. Experimental protocol
| Item | Setting | Source |
|---|---|---|
| Input size / epochs / batch | 640 / 100 / 16 | paper |
| Optimizer | SGD, lr0 0.01, momentum 0.937, weight decay 5e-4 | paper (SGD inferred) |
| Initialisation | COCO-pretrained weights | paper |
| Augmentation | rotation, translation, scale, shear, mixup, flip (+ default mosaic / HSV) | types from paper, magnitudes assumed |
| D-Fire split | train 15,498 / val 1,723 / test 4,306 | paper; val = seeded 10 % of the official train set |
| WSDY split | 517 / 110 / 110 | paper; shipped folders used, else seeded random split |
| Test set | used only by `evaluate.py` after training; never for tuning | assignment rule |
| Metrics | precision, recall, AP per class, mAP@0.5 (main); mAP@0.5:0.95, latency additionally | paper + extras for Assignment 03 |

## 6. Deviations and assumptions (see also `configs/baseline.yaml`)
1. **Validation list** – the paper does not publish which images form its validation set; a seeded 10 % hold-out of the official D-Fire training set is used.
2. **Augmentation magnitudes** – only the augmentation *types* are given; magnitudes are assumptions.
3. **Optimizer** – set explicitly to SGD, because Ultralytics' `auto` mode may choose AdamW and override the stated learning rate.
4. **Framework version** – the paper does not state versions; the installed Ultralytics/PyTorch versions are recorded in `results/*_train_info.json`.
5. **Hardware** – paper: Tesla V100 16 GB; this work: see `results/*_train_info.json`. Training time is therefore not directly comparable.
6. **Scope** – YOLOv5n/v5u/YOLOv7x/YOLOv8x and Faster R-CNN from the paper are not all re-trained (compute). YOLOv7x is not in the Ultralytics package; to reproduce it use <https://github.com/WongKinYiu/yolov7> with the same split lists in `data/prepared/*/*.txt`.
7. **Evaluation thresholds** – conf 0.001 / NMS-IoU 0.7 (Ultralytics validation defaults); the paper does not state them.
8. **Seeds** – single run (seed 0), like the paper; multi-seed analysis is planned for Assignment 03.

## 7. Outputs produced by the pipeline
`figures/dfire_*.png`, `figures/wsdy_*.png` (dataset analysis) · `figures/<run>_curves.png` · `results/<run>_test_metrics.json` ·
`results/<run>_train_info.json` · `results/comparison_table.{md,csv}` · `runs/<run>/` (weights, Ultralytics plots, `results.csv`).
Large weight files are not committed; upload them to a GitHub Release or Drive and link here: **<add link>**.

## 8. References and licences
1. Gonçalves et al., *Fire* 7(4):140, 2024, doi:10.3390/fire7040140.
2. Venâncio et al., *Neural Comput. Appl.* 34:15349–15368, 2022, doi:10.1007/s00521-022-07467-z (D-Fire).
3. Hemateja, WildFire Smoke Dataset YOLO, Kaggle.
4. Ultralytics YOLOv8 – <https://github.com/ultralytics/ultralytics> (AGPL-3.0).
