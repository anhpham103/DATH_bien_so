import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

from _common import CHARS, IMG_EXTS, ROOT, SPLITS, git_commit
from _deps import enhance, rectify, split_lines
from compare_ab import load_strings

H, W = 32, 160
VALID = set(CHARS)


def to_line(img, stretch=False):
    if stretch:
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    new_w = max(1, min(W, round(w * H / h)))
    line = cv2.resize(img, (new_w, H), interpolation=cv2.INTER_AREA)
    if new_w < W:
        line = cv2.copyMakeBorder(line, 0, 0, 0, W - new_w, cv2.BORDER_REPLICATE)
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets" / "chars" / "images"))
    ap.add_argument("--strings", default=str(ROOT / "datasets" / "plate_strings.csv"))
    ap.add_argument("--dst", default=str(ROOT / "datasets" / "lines"))
    ap.add_argument("--stretch", action="store_true")
    args = ap.parse_args()

    gts = load_strings(args.strings)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "dict_chars.txt").write_text("\n".join(CHARS) + "\n", encoding="utf-8")
    stats = {s: Counter() for s in SPLITS}

    for split in SPLITS:
        src = Path(args.src) / split
        if not src.is_dir():
            continue
        (dst / split).mkdir(exist_ok=True)
        lines_out = []
        for p in sorted(q for q in src.iterdir() if q.suffix.lower() in IMG_EXTS):
            gt = gts.get(p.stem)
            if gt is None:
                stats[split]["no_string_label"] += 1
                continue
            parts = gt.split("|")
            if any(set(t) - VALID for t in parts) or not all(parts):
                stats[split]["label_outside_30_chars"] += 1
                continue
            two = len(parts) == 2
            crop = cv2.imread(str(p))
            if crop is None:
                stats[split]["unreadable_image"] += 1
                continue
            plate = enhance(rectify(crop, two))
            segs = split_lines(plate) if two else (plate,)
            if any(s.shape[0] < 4 or s.shape[1] < 4 for s in segs):
                stats[split]["bad_line_split"] += 1
                continue
            for k, (seg, label) in enumerate(zip(segs, parts)):
                name = f"{p.stem}_l{k}.png"
                cv2.imwrite(str(dst / split / name), to_line(seg, args.stretch))
                lines_out.append(f"{split}/{name}\t{label}")
            stats[split]["plates_2_line" if two else "plates_1_line"] += 1
        (dst / f"{split}.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        stats[split]["line_images"] = len(lines_out)

    keys = ["plates_1_line", "plates_2_line", "line_images", "bad_line_split",
            "label_outside_30_chars", "no_string_label", "unreadable_image"]
    out = ROOT / "results" / "lines_c_stats.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", *keys, "commit"])
        for s in SPLITS:
            w.writerow([s, *[stats[s][k] for k in keys], git_commit()])
    for s in SPLITS:
        print(s, {k: stats[s][k] for k in keys if stats[s][k]})
    print(f"Line images in {dst}; stats in {out}")
    print("Inspect a few 2-line plate crops by eye before training method C (week 5).")


if __name__ == "__main__":
    main()
