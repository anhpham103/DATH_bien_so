import argparse
import csv
import json
import shutil
import time
from pathlib import Path

from _common import ROOT, git_commit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "data_chars_enh.yaml"))
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="chars_b")
    ap.add_argument("--project", default=str(ROOT / "runs" / "detect"),
                    help="point this at Google Drive so results survive a dropped session")
    ap.add_argument("--resume", action="store_true", help="resume from last.pt")
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for a quick test")
    args = ap.parse_args()

    from ultralytics import YOLO

    run_dir = Path(args.project) / args.name
    t0 = time.time()
    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"{last} not found, cannot resume")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        cfg = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                   patience=args.patience, seed=args.seed, deterministic=True,
                   fliplr=0.0,
                   mosaic=1.0, degrees=5, perspective=0.0005, hsv_v=0.4, hsv_s=0.5,
                   project=args.project, name=args.name, exist_ok=True)
        if args.device is not None:
            cfg["device"] = args.device
        model = YOLO(args.model)
        model.train(**cfg)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "train_config.json").write_text(
            json.dumps({**cfg, "model": args.model, "commit": git_commit()}, indent=2),
            encoding="utf-8")
    hours = (time.time() - t0) / 3600

    best = run_dir / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / "models" / "char_yolo26n.pt")

    m = YOLO(str(best)).val(data=args.data, split="val", imgsz=args.imgsz, plots=True,
                            project=args.project, name=f"{args.name}_val", exist_ok=True)
    out = ROOT / "results" / "chars_b_train.csv"
    out.parent.mkdir(exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "model", "imgsz", "epochs", "seed", "fliplr", "split",
                        "precision", "recall", "mAP50", "mAP50_95", "train_hours", "commit"])
        w.writerow([args.name, args.model, args.imgsz, args.epochs, args.seed, 0.0, "val",
                    f"{m.box.mp:.4f}", f"{m.box.mr:.4f}", f"{m.box.map50:.4f}",
                    f"{m.box.map:.4f}", f"{hours:.2f}", git_commit()])
    print(f"best.pt -> models/char_yolo26n.pt; mAP@0.5 (val) = {m.box.map50:.3f}")
    print(f"Results written to {out}; plots in {run_dir}")


if __name__ == "__main__":
    main()
