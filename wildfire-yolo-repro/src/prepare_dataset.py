"""
Task 1 - Dataset preparation, statistics and visualisations.

D-Fire  (Venancio et al., 2022)  : https://github.com/gaiasd/DFireDataset
WSDY    (Hemateja, Kaggle)       : https://www.kaggle.com/datasets/ahemateja19bec1025/wildfiresmokedatasetyolo

Usage
-----
python src/prepare_dataset.py dfire --root data/D-Fire --out data/prepared
python src/prepare_dataset.py wsdy  --root data/WSDY    --out data/prepared

D-Fire ships with a train (17,221 imgs) / test (4,306 imgs) split. The paper's split is
Train 15,498 / Val 1,723 / Test 4,306, i.e. the validation set is ~10% of the official
training set. We reproduce that by a seeded random 10% hold-out (the paper does not publish
the validation image list -> small documented deviation).

WSDY: paper split is 517 / 110 / 110 (train / val / test). If the downloaded copy already has
split folders they are used; otherwise a seeded 517/110/110-proportional random split is made.
"""
import argparse
import json
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# Published D-Fire statistics (used only as a sanity check on class ids)
DFIRE_BOXES = {"fire": 14692, "smoke": 11865}
DFIRE_IMAGES = {"fire_only": 1164, "smoke_only": 5867, "fire_and_smoke": 4658, "none": 9838}


# ----------------------------------------------------------------------------- helpers
def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def label_path(img):
    """Ultralytics convention: .../images/x.jpg -> .../labels/x.txt"""
    s = str(img)
    sep = os.sep
    if f"{sep}images{sep}" in s:
        s = s.replace(f"{sep}images{sep}", f"{sep}labels{sep}")
    return Path(s).with_suffix(".txt")


def read_label(txt):
    rows = []
    if txt.exists():
        for line in txt.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append((int(float(parts[0])), *map(float, parts[1:5])))
    return rows


def ensure_labels_dir(split_dir):
    """D-Fire archives use 'labels' or 'det_labels'; Ultralytics needs 'labels'."""
    labels, det = split_dir / "labels", split_dir / "det_labels"
    if not labels.exists() and det.exists():
        os.symlink(det.resolve(), labels, target_is_directory=True)
        print(f"[info] linked {labels} -> {det}")
    if not labels.exists():
        raise FileNotFoundError(f"No labels/ or det_labels/ folder in {split_dir}")


def write_list(paths, out_txt):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(str(p.resolve()) for p in paths) + "\n")


def collect_stats(images, split_name, names):
    """Per-image and per-box statistics for one split."""
    rows, boxes = [], []
    for img in images:
        lab = read_label(label_path(img))
        cls = {c for c, *_ in lab}
        try:
            w, h = Image.open(img).size
        except Exception:
            w = h = np.nan
        rows.append({"split": split_name, "image": img.name, "n_boxes": len(lab),
                     "classes": ",".join(names[c] for c in sorted(cls)) if cls else "none",
                     "width": w, "height": h})
        for c, cx, cy, bw, bh in lab:
            boxes.append({"split": split_name, "cls": names.get(c, str(c)),
                          "rel_w": bw, "rel_h": bh, "rel_area": bw * bh})
    return pd.DataFrame(rows), pd.DataFrame(boxes)


def make_figures(img_df, box_df, samples, names, out_dir, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) boxes per class per split
    fig, ax = plt.subplots(figsize=(6, 4))
    box_df.groupby(["split", "cls"]).size().unstack(fill_value=0).plot.bar(ax=ax, rot=0)
    ax.set_title(f"{tag}: bounding boxes per class"); ax.set_ylabel("# boxes")
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_boxes_per_class.png", dpi=200); plt.close(fig)
    # 2) image category distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    img_df.groupby(["split", "classes"]).size().unstack(fill_value=0).plot.bar(ax=ax, rot=0)
    ax.set_title(f"{tag}: images per category"); ax.set_ylabel("# images")
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_image_categories.png", dpi=200); plt.close(fig)
    # 3) object size distribution (relative area, log-x) -> evidence for the 'small object' limitation
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, g in box_df.groupby("cls"):
        ax.hist(np.log10(g.rel_area.clip(lower=1e-6)), bins=60, alpha=0.55, label=c)
    ax.set_xlabel("log10(box area / image area)"); ax.set_ylabel("# boxes"); ax.legend()
    ax.set_title(f"{tag}: object size distribution")
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_object_size_hist.png", dpi=200); plt.close(fig)
    # 4) annotated samples
    colors = {0: "red", 1: "yellow", 2: "cyan"}
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, img in zip(axes.ravel(), samples):
        im = Image.open(img).convert("RGB"); d = ImageDraw.Draw(im); W, H = im.size
        for c, cx, cy, bw, bh in read_label(label_path(img)):
            d.rectangle([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H],
                        outline=colors.get(c, "white"), width=max(2, W // 200))
        ax.imshow(im); ax.axis("off")
        ax.set_title(", ".join(sorted({names[c] for c, *_ in read_label(label_path(img))})) or "negative", fontsize=9)
    fig.suptitle(f"{tag}: random annotated samples (verify class names!)")
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_samples.png", dpi=150); plt.close(fig)


def infer_dfire_names(all_images):
    """Class ids differ between D-Fire copies. Published totals: fire 14,692 boxes > smoke 11,865 boxes,
    so the id with more boxes over the *whole* dataset is fire."""
    counts = Counter()
    for img in all_images:
        for c, *_ in read_label(label_path(img)):
            counts[c] += 1
    print("[info] boxes per class id over whole dataset:", dict(counts))
    if len(counts) != 2:
        raise RuntimeError(f"Expected 2 classes in D-Fire, found ids {sorted(counts)}")
    fire_id = max(counts, key=counts.get)
    names = {fire_id: "fire", 1 - fire_id if fire_id in (0, 1) else min(counts): "smoke"}
    print(f"[info] inferred names: {dict(sorted(names.items()))} "
          f"(published totals: fire={DFIRE_BOXES['fire']}, smoke={DFIRE_BOXES['smoke']})")
    return dict(sorted(names.items())), counts


def write_yaml(path, train, val, test, names):
    cfg = {"path": str(path.parent.resolve()), "train": str(train.resolve()), "val": str(val.resolve()),
           "test": str(test.resolve()), "names": names}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[ok] wrote {path}")


# ----------------------------------------------------------------------------- D-Fire
def prepare_dfire(a):
    root, out = Path(a.root), Path(a.out)
    tr_dir, te_dir = root / "train", root / "test"
    for d in (tr_dir, te_dir):
        ensure_labels_dir(d)
    train_all, test = list_images(tr_dir / "images"), list_images(te_dir / "images")
    print(f"[info] official train={len(train_all)}  test={len(test)}  (expected 17,221 / 4,306)")

    names, counts = (infer_dfire_names(train_all + test) if a.names == "auto"
                     else ({int(k): v for k, v in (kv.split(":") for kv in a.names.split(","))}, Counter()))

    rng = random.Random(a.seed)
    idx = list(range(len(train_all))); rng.shuffle(idx)
    n_val = math.ceil(a.val_frac * len(train_all))          # 10% -> 1,723 for 17,221 images
    val = sorted(train_all[i] for i in idx[:n_val]); train = sorted(train_all[i] for i in idx[n_val:])
    print(f"[info] split -> train={len(train)} val={len(val)} test={len(test)} (paper: 15,498 / 1,723 / 4,306)")

    ds = out / "dfire"
    write_list(train, ds / "train.txt"); write_list(val, ds / "val.txt"); write_list(test, ds / "test.txt")
    write_yaml(ds / "dfire.yaml", ds / "train.txt", ds / "val.txt", ds / "test.txt", names)

    dfs = [collect_stats(s, n, names) for s, n in ((train, "train"), (val, "val"), (test, "test"))]
    img_df, box_df = pd.concat([d[0] for d in dfs]), pd.concat([d[1] for d in dfs])
    res = Path(a.results); res.mkdir(parents=True, exist_ok=True)
    img_df.to_csv(res / "dfire_image_stats.csv", index=False)
    summary = {
        "images_per_split": img_df.groupby("split").size().to_dict(),
        "image_categories_all": img_df.classes.value_counts().to_dict(),
        "boxes_per_class_per_split": box_df.groupby(["split", "cls"]).size().unstack(fill_value=0).to_dict("index"),
        "published_image_categories": DFIRE_IMAGES, "published_boxes": DFIRE_BOXES,
        "median_rel_area_by_class": box_df.groupby("cls").rel_area.median().to_dict(),
        "share_boxes_under_1pct_area": box_df.groupby("cls").rel_area.apply(lambda s: float((s < 0.01).mean())).to_dict(),
        "image_size_top5": img_df.assign(sz=img_df.width.astype(str) + "x" + img_df.height.astype(str)).sz.value_counts().head(5).to_dict(),
        "split_seed": a.seed, "class_names": names,
    }
    (res / "dfire_dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    samples = random.Random(1).sample(train, min(8, len(train)))
    make_figures(img_df, box_df, samples, names, Path(a.figures), "dfire")


# ----------------------------------------------------------------------------- WSDY
def prepare_wsdy(a):
    root, out = Path(a.root), Path(a.out)
    imgs = list_images(root)
    if not imgs:
        raise FileNotFoundError(f"No images found under {root}")

    def find_label(img):
        for cand in (label_path(img), img.with_suffix(".txt")):
            if cand.exists():
                return cand
        hits = list(root.rglob(img.stem + ".txt"))
        return hits[0] if hits else None

    def split_of(img):
        parts = {p.lower() for p in img.relative_to(root).parts}
        for key, name in (("train", "train"), ("valid", "val"), ("val", "val"), ("validation", "val"), ("test", "test")):
            if key in parts:
                return name
        return None

    pairs = [(i, find_label(i), split_of(i)) for i in imgs]
    pairs = [p for p in pairs if p[1] is not None]
    print(f"[info] {len(pairs)} image/label pairs found")
    if all(p[2] for p in pairs):
        print("[info] using split folders shipped with the dataset")
        split = {s: [p for p in pairs if p[2] == s] for s in ("train", "val", "test")}
    else:
        rng = random.Random(a.seed); rng.shuffle(pairs)
        n = len(pairs); n_val = n_test = round(n * 110 / 737)    # 110/737 each, as in the paper
        split = {"val": pairs[:n_val], "test": pairs[n_val:n_val + n_test], "train": pairs[n_val + n_test:]}
    print({k: len(v) for k, v in split.items()}, "(paper: train 517 / val 110 / test 110)")

    ds = out / "wsdy"
    if ds.exists():
        shutil.rmtree(ds)
    names = {0: "smoke"}
    stats_i, stats_b = [], []
    for s, items in split.items():
        (ds / s / "images").mkdir(parents=True); (ds / s / "labels").mkdir(parents=True)
        for img, lab, _ in items:
            shutil.copy2(img, ds / s / "images" / img.name); shutil.copy2(lab, ds / s / "labels" / (img.stem + ".txt"))
        imgs_s = list_images(ds / s / "images")
        di, db = collect_stats(imgs_s, s, names); stats_i.append(di); stats_b.append(db)
        write_list(imgs_s, ds / f"{s}.txt")
    write_yaml(ds / "wsdy.yaml", ds / "train.txt", ds / "val.txt", ds / "test.txt", names)
    img_df, box_df = pd.concat(stats_i), pd.concat(stats_b)
    res = Path(a.results); res.mkdir(parents=True, exist_ok=True)
    img_df.to_csv(res / "wsdy_image_stats.csv", index=False)
    summary = {"images_per_split": img_df.groupby("split").size().to_dict(),
               "boxes_per_split": box_df.groupby("split").size().to_dict(),
               "median_rel_area": float(box_df.rel_area.median()),
               "share_boxes_under_1pct_area": float((box_df.rel_area < 0.01).mean()), "split_seed": a.seed}
    (res / "wsdy_dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    train_imgs = list_images(ds / "train" / "images")
    make_figures(img_df, box_df, random.Random(1).sample(train_imgs, min(8, len(train_imgs))), names, Path(a.figures), "wsdy")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("dfire", "wsdy"):
        s = sub.add_parser(name)
        s.add_argument("--root", required=True, help="extracted dataset folder")
        s.add_argument("--out", default="data/prepared")
        s.add_argument("--results", default="results")
        s.add_argument("--figures", default="figures")
        s.add_argument("--seed", type=int, default=0)
        if name == "dfire":
            s.add_argument("--val-frac", type=float, default=0.10)
            s.add_argument("--names", default="auto", help="'auto' or e.g. '0:smoke,1:fire'")
    a = p.parse_args()
    prepare_dfire(a) if a.cmd == "dfire" else prepare_wsdy(a)


if __name__ == "__main__":
    main()
