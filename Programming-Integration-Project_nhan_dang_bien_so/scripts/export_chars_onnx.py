import argparse
import csv
import shutil
import statistics
import time
from pathlib import Path

import cv2

from _common import IMG_EXTS, ROOT, git_commit
from compare_ab import cpu_name


def timed_reads(rc, imgs, two_flags):
    rc.read_chars(imgs[0], two_flags[0])
    outs, ms = [], []
    for img, two in zip(imgs, two_flags):
        t0 = time.perf_counter()
        outs.append(rc.read_chars(img, two))
        ms.append((time.perf_counter() - t0) * 1000)
    return outs, statistics.median(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "char_yolo26n.pt"))
    ap.add_argument("--out", default=str(ROOT / "models" / "char_yolo26n.onnx"))
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--images", default=str(ROOT / "datasets" / "chars_enh" / "images" / "val"),
                    help="enhanced images used to compare .pt and .onnx")
    ap.add_argument("--strings", default=str(ROOT / "datasets" / "plate_strings.csv"))
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    from ultralytics import YOLO
    onnx_path = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, simplify=True,
                                          dynamic=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(onnx_path).resolve() != Path(args.out).resolve():
        shutil.copy2(onnx_path, args.out)
    print(f"Exported {args.out}")

    from compare_ab import load_strings
    import read_chars as rc
    gts = load_strings(args.strings) if Path(args.strings).exists() else {}
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXTS)
    paths = paths[:args.n]
    if not paths:
        print("No images to check the ONNX model against; skipping comparison.")
        return
    imgs = [cv2.imread(str(p)) for p in paths]
    twos = ["|" in gts[p.stem] if p.stem in gts else im.shape[1] / im.shape[0] < 3
            for p, im in zip(paths, imgs)]

    rc.set_model(args.weights)
    pt_out, pt_ms = timed_reads(rc, imgs, twos)
    rc.set_model(args.out)
    ox_out, ox_ms = timed_reads(rc, imgs, twos)
    same = sum(a == b for a, b in zip(pt_out, ox_out)) / len(paths)

    out = ROOT / "results" / "chars_onnx_check.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_images", "pt_onnx_match_rate", "ms_per_plate_pt", "ms_per_plate_onnx",
                    "imgsz", "cpu", "commit"])
        w.writerow([len(paths), f"{same:.4f}", f"{pt_ms:.1f}", f"{ox_ms:.1f}", args.imgsz,
                    cpu_name(), git_commit()])
    print(f"String match between .pt and .onnx: {same:.1%} on {len(paths)} images")
    print(f"Median ms/plate: .pt {pt_ms:.1f}, .onnx {ox_ms:.1f}  ->  {out}")
    if same < 0.99:
        diffs = [(p.name, a, b) for p, a, b in zip(paths, pt_out, ox_out) if a != b][:5]
        print("[note] ONNX differs from .pt on more than 1% of images; e.g.:", diffs)


if __name__ == "__main__":
    main()
