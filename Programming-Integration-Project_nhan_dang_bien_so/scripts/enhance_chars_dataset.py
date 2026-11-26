import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

from _common import CHARS, IMG_EXTS, ROOT, SPLITS, git_commit
from _deps import enhance


def check_label(path):
    classes, errors = [], []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f"{path.name}:{n} expected 5 columns, got {len(parts)}")
            continue
        try:
            c = int(parts[0])
            xs = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"{path.name}:{n} not a number")
            continue
        if not 0 <= c < len(CHARS):
            errors.append(f"{path.name}:{n} class id {c} outside 0..29")
        if not all(0.0 <= v <= 1.0 for v in xs):
            errors.append(f"{path.name}:{n} coordinates outside [0, 1]")
        classes.append(c)
    return classes, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets" / "chars"))
    ap.add_argument("--dst", default=str(ROOT / "datasets" / "chars_enh"))
    ap.add_argument("--yaml", default=str(ROOT / "datasets" / "data_chars_enh.yaml"))
    ap.add_argument("--stats", default=str(ROOT / "results" / "char_stats.csv"))
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    counts = {s: Counter() for s in SPLITS}
    n_img = Counter()
    errors, missing = [], []

    for split in SPLITS:
        img_dir = src / "images" / split
        if not img_dir.is_dir():
            print(f"[skip] {img_dir} not found")
            continue
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"{img_path} cannot be read")
                continue
            cv2.imwrite(str(dst / "images" / split / img_path.name), enhance(img))
            n_img[split] += 1
            lbl = src / "labels" / split / f"{img_path.stem}.txt"
            if not lbl.exists():
                missing.append(str(lbl))
                continue
            cls, errs = check_label(lbl)
            errors += errs
            counts[split].update(cls)
            shutil.copy2(lbl, dst / "labels" / split / lbl.name)

    Path(args.yaml).write_text(
        f"path: {dst.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names: [" + ", ".join(f'"{c}"' for c in CHARS) + "]\n", encoding="utf-8")

    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["char", *SPLITS, "total", "commit"])
        commit = git_commit()
        for i, ch in enumerate(CHARS):
            row = [counts[s][i] for s in SPLITS]
            w.writerow([ch, *row, sum(row), commit])
        w.writerow(["n_images", *[n_img[s] for s in SPLITS], sum(n_img.values()), commit])

    print(f"Enhanced images: {dict(n_img)} -> {dst}")
    print(f"Training YAML: {args.yaml}")
    print(f"Character stats: {args.stats}")
    rare = [CHARS[i] for i in range(len(CHARS)) if counts["train"][i] < 50]
    if rare:
        print(f"[note] characters with fewer than 50 training samples: {', '.join(rare)}")
    if missing:
        print(f"[note] {len(missing)} images have no label file (treated as background), "
              f"e.g. {missing[0]}")
    if errors:
        print(f"[ERROR] {len(errors)} label errors, e.g.:", *errors[:10], sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
