"""
Task 3/4 - Training curves + original-vs-reproduced comparison table.

python src/make_figures.py curves --run runs/dfire_yolov8s --name dfire_yolov8s
python src/make_figures.py compare --results results --paper paper_reference/paper_results.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def curves(a):
    df = pd.read_csv(Path(a.run) / "results.csv"); df.columns = [c.strip() for c in df.columns]
    ep = df["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, k in zip(axes[:2], ("box_loss", "cls_loss")):
        ax.plot(ep, df[f"train/{k}"], label="train"); ax.plot(ep, df[f"val/{k}"], label="validation")
        ax.set_title(k.replace("_", " ")); ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=.3)
    axes[2].plot(ep, df["metrics/mAP50(B)"] * 100, label="mAP@0.5")
    axes[2].plot(ep, df["metrics/mAP50-95(B)"] * 100, label="mAP@0.5:0.95")
    axes[2].set_title("validation mAP (%)"); axes[2].set_xlabel("epoch"); axes[2].legend(); axes[2].grid(alpha=.3)
    fig.suptitle(a.name); fig.tight_layout()
    out = Path(a.figures); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{a.name}_curves.png", dpi=200); plt.close(fig)
    print("saved", out / f"{a.name}_curves.png")


def compare(a):
    paper = pd.read_csv(a.paper); res = Path(a.results); rows = []
    for f in sorted(res.glob("*_test_metrics.json")):
        m = json.loads(f.read_text()); ds, model = m["run"].split("_", 1)
        ref = paper[(paper.dataset == ds) & (paper.model == model)]
        ref = ref.iloc[0] if len(ref) else None
        ti = res / f"{m['run']}_train_info.json"
        hours = json.loads(ti.read_text())["train_time_h"] if ti.exists() else None

        def add(metric, orig, rep):
            diff = None if (orig is None or pd.isna(orig) or rep is None) else rep - float(orig)
            rows.append({"Dataset": ds, "Model": model, "Evaluation Metric": metric,
                         "Original Paper": None if orig is None or pd.isna(orig) else float(orig),
                         "Reproduced Model": rep, "Difference": diff})
        g = (lambda c: None if ref is None else ref[c])
        add("mAP@0.5 (%)", g("map50"), m["map50"])
        add("AP smoke (%)", g("ap_smoke"), m["ap50_per_class"].get("smoke"))
        if "fire" in m["ap50_per_class"]:
            add("AP fire (%)", g("ap_fire"), m["ap50_per_class"]["fire"])
        add("GFLOPs", g("gflops"), m.get("gflops"))
        add("Training time (h)", g("train_time_h"), hours)
        add("Precision (%)", None, m["precision"]); add("Recall (%)", None, m["recall"])
        add("mAP@0.5:0.95 (%)", None, m["map50_95"])
    t = pd.DataFrame(rows)
    if t.empty:
        print("no *_test_metrics.json found in", res); return
    t.to_csv(res / "comparison_table.csv", index=False)
    md = t.round(2).fillna("n/r").to_markdown(index=False) if hasattr(t, "to_markdown") else t.round(2).to_string()
    (res / "comparison_table.md").write_text(md + "\n\nn/r = not reported in the paper / not available. "
                                             "Difference = reproduced - original (percentage points for % metrics).\n")
    print(md)
    sub = t[t["Evaluation Metric"].str.startswith("mAP@0.5 ")].dropna(subset=["Original Paper"])
    if len(sub):
        fig, ax = plt.subplots(figsize=(7, 4)); lab = sub.Dataset + " " + sub.Model; x = range(len(sub))
        ax.bar([i - .2 for i in x], sub["Original Paper"], .4, label="Original paper")
        ax.bar([i + .2 for i in x], sub["Reproduced Model"], .4, label="Reproduced")
        ax.set_xticks(list(x)); ax.set_xticklabels(lab, rotation=20); ax.set_ylabel("mAP@0.5 (%)"); ax.legend()
        fig.tight_layout(); fig.savefig(Path(a.figures) / "map50_original_vs_reproduced.png", dpi=200)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curves"); c.add_argument("--run", required=True); c.add_argument("--name", required=True)
    c.add_argument("--figures", default="figures")
    k = sub.add_parser("compare"); k.add_argument("--results", default="results")
    k.add_argument("--paper", default="paper_reference/paper_results.csv"); k.add_argument("--figures", default="figures")
    a = p.parse_args(); curves(a) if a.cmd == "curves" else compare(a)


if __name__ == "__main__":
    main()
