"""
Task 4 - Evaluate trained weights on the held-out TEST split (never used for tuning).

python src/evaluate.py --data data/prepared/dfire/dfire.yaml --weights runs/dfire_yolov8s/weights/best.pt --name dfire_yolov8s
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import load_config, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--split", default=None, help="test (default) or val")
    p.add_argument("--project", default="runs")
    p.add_argument("--results", default="results")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    split = a.split or cfg["eval"]["split"]
    model = YOLO(a.weights)
    kw = dict(data=a.data, split=split, imgsz=cfg["imgsz"], batch=cfg["batch"], conf=cfg["eval"]["conf"],
              iou=cfg["eval"]["iou"], plots=True, project=str(Path(a.project).resolve()), name=f"{a.name}_{split}_eval",
              exist_ok=True)
    if a.device is not None:
        kw["device"] = a.device
    m = model.val(**kw)

    names = model.names
    per_class = {names[int(c)]: float(v) * 100 for c, v in zip(m.box.ap_class_index, m.box.ap50)}
    per_class_5095 = {names[int(c)]: float(v) * 100 for c, v in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])}
    out = {
        "run": a.name, "split": split, "weights": a.weights,
        "precision": float(m.box.mp) * 100, "recall": float(m.box.mr) * 100,
        "map50": float(m.box.map50) * 100, "map50_95": float(m.box.map) * 100,
        "ap50_per_class": per_class, "ap50_95_per_class": per_class_5095,
        "speed_ms_per_image": {k: float(v) for k, v in m.speed.items()},   # measured latency (bonus for Assignment 03)
    }
    try:
        n_layers, n_params, _, gflops = model.info(verbose=False)
        out.update(params=int(n_params), gflops=float(gflops))
    except Exception as e:  # older/newer API differences
        out["model_info_error"] = str(e)
    save_json(out, Path(a.results) / f"{a.name}_{split}_metrics.json")
    print(out)


if __name__ == "__main__":
    main()
