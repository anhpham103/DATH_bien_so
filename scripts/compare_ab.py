import argparse
import csv
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2

from _common import IMG_EXTS, ROOT, git_commit, levenshtein
from _deps import clean, enhance, load_read_baseline, postprocess, rectify, split_raw

PATH_COLS = ("crop_path", "crop", "image", "img", "file", "filename", "path", "anh")
TEXT_COLS = ("text", "plate", "label", "string", "chuoi", "plate_string", "gt")


def sort_chars_guide(boxes, two_lines):
    if not boxes:
        return ""
    if not two_lines:
        rows = [boxes]
    else:
        ys = sorted(b[1] for b in boxes)
        _, i = max(((ys[k + 1] - ys[k], k) for k in range(len(ys) - 1)), default=(0, 0))
        cut = (ys[i] + ys[i + 1]) / 2 if len(ys) > 1 else ys[0]
        rows = [[b for b in boxes if b[1] < cut], [b for b in boxes if b[1] >= cut]]
    return "|".join("".join(b[3] for b in sorted(r, key=lambda b: b[0])) for r in rows)


def load_strings(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in reader.fieldnames}
        pc = next((cols[c] for c in PATH_COLS if c in cols), reader.fieldnames[0])
        tc = next((cols[c] for c in TEXT_COLS if c in cols), reader.fieldnames[1])
        return {Path(r[pc]).stem: r[tc].strip().upper() for r in reader if r[tc].strip()}


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def summarize(rows, engine, kind):
    rs = [r for r in rows if r["engine"] == engine and (kind == "all" or r["plate_type"] == kind)]
    if not rs:
        return None
    n_chars = sum(len(r["gt"]) for r in rs)
    cer_b = sum(r["edits_before"] for r in rs) / n_chars
    cer_a = sum(r["edits_after"] for r in rs) / n_chars
    return {
        "method": engine, "plate_type": kind, "n_plates": len(rs),
        "cer_before_pp": f"{cer_b:.4f}", "cer_after_pp": f"{cer_a:.4f}",
        "char_accuracy": f"{max(0.0, 1 - cer_a):.4f}",
        "exact_before_pp": f"{sum(r['exact_before'] for r in rs) / len(rs):.4f}",
        "exact_after_pp": f"{sum(r['exact_after'] for r in rs) / len(rs):.4f}",
        "order_errors": sum(r["order_error"] for r in rs),
        "median_ms_per_plate": f"{statistics.median(r['ms'] for r in rs):.1f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--images", default=None, help="default: datasets/chars/images/<split>")
    ap.add_argument("--strings", default=str(ROOT / "datasets" / "plate_strings.csv"))
    ap.add_argument("--engines", default="A,B", help="A, B or A,B")
    ap.add_argument("--model", default=None, help="character model weights (.pt or .onnx)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-rectify", action="store_true", help="skip rectification (diagnostic)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test":
        print("[WARNING] Evaluating on the test split: do this only once, in Step 7, after M2.")

    gts = load_strings(args.strings)
    img_dir = Path(args.images or ROOT / "datasets" / "chars" / "images" / args.split)
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.stem in gts)
    if args.limit:
        imgs = imgs[:args.limit]
    print(f"{len(imgs)} plate crops with string labels in {img_dir}")

    engines = [e.strip().upper() for e in args.engines.split(",")]
    readers = {}
    if "A" in engines:
        fn = load_read_baseline()
        if fn:
            readers["A"] = fn
    if "B" in engines:
        import read_chars as rc
        if args.model:
            rc.set_model(args.model)
        rc.get_model()
        readers["B"] = rc

    rows = []
    for idx, p in enumerate(imgs, 1):
        gt_raw = gts[p.stem]
        two = "|" in gt_raw
        gt = clean(gt_raw)
        crop = cv2.imread(str(p))
        plate_img = enhance(crop if args.no_rectify else rectify(crop, two))
        outs = {}
        for name, reader in readers.items():
            if name == "B":
                if idx == 1:
                    reader.detect_chars(plate_img, args.conf)
                t0 = time.perf_counter()
                boxes = [b[:4] for b in reader.detect_chars(plate_img, args.conf)]
                raw = reader.sort_chars(boxes, two, img_h=plate_img.shape[0])
                ms = (time.perf_counter() - t0) * 1000
                outs["B"] = (raw, ms)
                outs["B_guide"] = (sort_chars_guide(boxes, two), ms)
            else:
                if idx == 1:
                    reader(plate_img, two)
                t0 = time.perf_counter()
                raw = reader(plate_img, two)
                outs[name] = (raw, (time.perf_counter() - t0) * 1000)
        for name, (raw, ms) in outs.items():
            before = clean(raw)
            plate, ok = postprocess(*split_raw(raw))
            rows.append({
                "image": p.name, "plate_type": "2_line" if two else "1_line", "engine": name,
                "gt": gt, "raw": raw, "after_pp": plate, "valid": int(ok),
                "exact_before": int(before == gt), "exact_after": int(plate == gt),
                "edits_before": levenshtein(before, gt), "edits_after": levenshtein(plate, gt),
                "order_error": int(before != gt and sorted(before) == sorted(gt)),
                "ms": ms,
            })
        if idx % 100 == 0:
            print(f"  {idx}/{len(imgs)}")

    suffix = f"_{args.split}" + (f"_{args.tag}" if args.tag else "")
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    commit, cpu = git_commit(), cpu_name()

    pred_path = out_dir / f"ab_predictions{suffix}.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[*rows[0].keys(), "commit"] if rows else ["image"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "ms": f"{r['ms']:.1f}", "commit": commit})

    summary = [s for e in readers.keys() | ({"B_guide"} if "B" in readers else set())
               for k in ("all", "1_line", "2_line") if (s := summarize(rows, e, k))]
    order = {"A": 0, "B_guide": 1, "B": 2}
    summary.sort(key=lambda s: (order.get(s["method"], 9), s["plate_type"]))
    sum_path = out_dir / f"ab_comparison{suffix}.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[*summary[0].keys(), "split", "cpu", "commit"])
        w.writeheader()
        for s in summary:
            w.writerow({**s, "split": args.split, "cpu": cpu, "commit": commit})

    print(f"\n{'Method':<9}{'Type':<8}{'N':>6}{'CER':>8}{'Exact pre':>11}"
          f"{'Exact post':>12}{'Order err':>11}{'ms':>8}")
    for s in summary:
        print(f"{s['method']:<9}{s['plate_type']:<8}{s['n_plates']:>6}{s['cer_after_pp']:>8}"
              f"{s['exact_before_pp']:>11}{s['exact_after_pp']:>12}{s['order_errors']:>11}"
              f"{s['median_ms_per_plate']:>8}")
    print(f"\nCPU: {cpu}\nSummary: {sum_path}\nPer image: {pred_path}")
    by_type = defaultdict(list)
    for r in rows:
        if r["engine"] == "B" and not r["exact_after"]:
            by_type[r["plate_type"]].append(r)
    for k, v in by_type.items():
        print(f"B wrong on {len(v)} {k} plates; e.g.: " +
              ", ".join(f"{r['image']} ({r['raw']} != {r['gt']})" for r in v[:3]))


if __name__ == "__main__":
    main()
