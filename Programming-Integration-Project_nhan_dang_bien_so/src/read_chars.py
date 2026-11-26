import os
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    ROOT / "models" / "char_yolo26n.onnx",
    ROOT / "models" / "char_yolo26n.pt",
    ROOT / "runs" / "detect" / "chars_b" / "weights" / "best.pt",
]
IMGSZ = 320

_model = None


def set_model(path):
    global _model
    from ultralytics import YOLO
    path = str(path)
    _model = YOLO(path, task="detect") if path.endswith(".onnx") else YOLO(path)
    return _model


def get_model():
    if _model is None:
        env = os.environ.get("CHAR_MODEL")
        paths = [Path(env)] if env else CANDIDATES
        for p in paths:
            if p.exists():
                return set_model(p)
        raise FileNotFoundError("Character model not found; set CHAR_MODEL or copy it to "
                                "models/char_yolo26n.onnx")
    return _model


def sort_chars(boxes, two_lines, img_h=None, min_rel_h=0.4, min_gap_rel=0.5):
    if not boxes:
        return "|" if two_lines else ""
    med_h = median(b[2] for b in boxes)
    boxes = [b for b in boxes if b[2] >= min_rel_h * med_h] or list(boxes)

    if not two_lines:
        return "".join(b[3] for b in sorted(boxes, key=lambda b: b[0]))

    ys = sorted(b[1] for b in boxes)
    gap, cut = 0.0, None
    for k in range(len(ys) - 1):
        if ys[k + 1] - ys[k] > gap:
            gap, cut = ys[k + 1] - ys[k], (ys[k] + ys[k + 1]) / 2
    if (cut is None or gap < min_gap_rel * med_h) and img_h:
        cut = img_h / 2
    elif cut is None:
        cut = ys[0] + 1
    rows = [[b for b in boxes if b[1] < cut], [b for b in boxes if b[1] >= cut]]
    return "|".join("".join(b[3] for b in sorted(r, key=lambda b: b[0])) for r in rows)


def detect_chars(plate_img, conf=0.25):
    model = get_model()
    r = model.predict(plate_img, imgsz=IMGSZ, conf=conf, agnostic_nms=True, verbose=False)[0]
    return [((x1 + x2) / 2, (y1 + y2) / 2, y2 - y1, model.names[int(c)], float(s))
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                              r.boxes.conf.tolist())]


def read_chars(plate_img, two_lines, conf=0.25):
    boxes = [b[:4] for b in detect_chars(plate_img, conf)]
    return sort_chars(boxes, two_lines, img_h=plate_img.shape[0])
