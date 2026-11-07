"""
Task 2 + 3 - Baseline model implementation and training (Ultralytics YOLOv8).

python src/train.py --data data/prepared/dfire/dfire.yaml --model yolov8s.pt --name dfire_yolov8s
python src/train.py --data data/prepared/wsdy/wsdy.yaml  --model yolov8n.pt --name wsdy_yolov8n
Add --resume to continue after a Colab/Kaggle disconnect.
Use --epochs N only for a documented reduced-budget run (the paper uses 100).
"""
import argparse
import time
from pathlib import Path

from ultralytics import YOLO

from utils import collect_env, load_config, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolov8s.pt", help="yolov8n.pt / yolov8s.pt (COCO-pretrained) or a .yaml for scratch")
    p.add_argument("--name", required=True, help="run name, convention: <dataset>_<model> e.g. dfire_yolov8s")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--project", default="runs")
    p.add_argument("--results", default="results")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    cfg = load_config(a.config)
    project = Path(a.project).resolve()
    run_dir = project / a.name
    env = collect_env()
    print("[env]", env)

    if a.resume:
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        t0 = time.time()
        model.train(resume=True)
    else:
        model = YOLO(a.model)
        kwargs = dict(
            data=a.data, imgsz=cfg["imgsz"], epochs=a.epochs or cfg["epochs"], batch=a.batch or cfg["batch"],
            optimizer=cfg["optimizer"], lr0=cfg["lr0"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
            pretrained=cfg["pretrained"], patience=cfg["patience"], cos_lr=cfg["cos_lr"], amp=cfg["amp"],
            seed=cfg["seed"], deterministic=cfg["deterministic"], workers=cfg["workers"], cache=cfg["cache"],
            project=str(project), name=a.name, exist_ok=False, plots=True, val=True, **cfg["augment"],
        )
        if a.device is not None:
            kwargs["device"] = a.device
        save_json({"args": kwargs, "model": a.model, "env": env}, run_dir / "experiment_config.json") \
            if not run_dir.exists() else None
        t0 = time.time()
        model.train(**kwargs)
        save_json({"args": kwargs, "model": a.model, "env": env}, run_dir / "experiment_config.json")

    hours = (time.time() - t0) / 3600
    info = {"run": a.name, "train_time_h": round(hours, 3),
            "estimated_energy_wh_paper_formula": round(300 * hours * 1.1, 2),   # 300 W x h x PUE 1.1 (paper's formula)
            "note": "wall-clock incl. per-epoch validation; if --resume was used this only covers the resumed part",
            "env": env, "best_weights": str(run_dir / "weights" / "best.pt")}
    save_json(info, Path(a.results) / f"{a.name}_train_info.json")
    print(info)


if __name__ == "__main__":
    main()
