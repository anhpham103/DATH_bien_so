import argparse
import csv
from pathlib import Path

import cv2

from _common import CHARS, IMG_EXTS, ROOT, SPLITS, git_commit, is_two_lines
from _deps import clean, postprocess, split_raw
from read_chars import sort_chars

NAME_COLS = ("crop_path", "crop", "crop_name", "image", "img", "file", "filename", "path")
TYPE_COLS = ("plate_type", "type", "cls", "class", "class_name", "loai_bien", "two_lines")


def load_plate_types(path):
    if not path or not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in reader.fieldnames}
        nc = next((cols[c] for c in NAME_COLS if c in cols), None)
        tc = next((cols[c] for c in TYPE_COLS if c in cols), None)
        if nc is None or tc is None:
            print(f"[note] {path}: name/type columns not recognized, using aspect ratio instead")
            return {}
        return {Path(r[nc]).stem: is_two_lines(r[tc]) for r in reader}


def read_boxes(label_path, w, h):
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, xc, yc, _, bh = int(parts[0]), *map(float, parts[1:])
        boxes.append((xc * w, yc * h, bh * h, CHARS[c]))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets" / "chars"))
    ap.add_argument("--crops-index", default=str(ROOT / "datasets" / "crops_index.csv"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "plate_strings.csv"))
    ap.add_argument("--review", default=str(ROOT / "results" / "plate_strings_review.csv"))
    ap.add_argument("--ratio", type=float, default=3.0,
                    help="width/height below this is treated as a 2-line plate")
    args = ap.parse_args()

    types = load_plate_types(args.crops_index)
    src = Path(args.src)
    rows, review = [], []
    n_type_from_index = 0

    for split in SPLITS:
        img_dir = src / "images" / split
        if not img_dir.is_dir():
            continue
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            lbl = src / "labels" / split / f"{img_path.stem}.txt"
            if not lbl.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                review.append((img_path.name, split, "", "unreadable image"))
                continue
            h, w = img.shape[:2]
            if img_path.stem in types:
                two = types[img_path.stem]
                n_type_from_index += 1
            else:
                two = w / h < args.ratio
            boxes = read_boxes(lbl, w, h)
            if not boxes:
                review.append((img_path.name, split, "", "empty label file"))
                continue
            text = sort_chars(boxes, two, img_h=h, min_rel_h=0.0)
            plate, ok = postprocess(*split_raw(text))
            rows.append((img_path.name, split, text, int(two), int(ok)))
            if not ok:
                review.append((img_path.name, split, text, "fails format or province code"))
            elif plate != clean(text):
                review.append((img_path.name, split, text, f"postprocess would change it to {plate}"))
            elif two and (text.startswith("|") or text.endswith("|")):
                review.append((img_path.name, split, text, "one line is empty"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop_path", "split", "text", "two_lines", "valid"])
        w.writerows(rows)

    Path(args.review).parent.mkdir(parents=True, exist_ok=True)
    with open(args.review, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop_path", "split", "text", "reason", "commit"])
        commit = git_commit()
        w.writerows((*r, commit) for r in review)

    n2 = sum(r[3] for r in rows)
    print(f"{len(rows)} plate strings -> {args.out} ({len(rows) - n2} one-line, {n2} two-line)")
    print(f"Plate type from crops_index.csv: {n_type_from_index}; "
          f"from aspect ratio: {len(rows) - n_type_from_index}")
    print(f"{len(review)} entries to check by hand -> {args.review}")
    for r in review[:10]:
        print("  ", *r)


if __name__ == "__main__":
    main()
