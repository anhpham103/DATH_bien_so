import re

import cv2
import numpy as np


def order_corners(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]],
                    dtype=np.float32)


def crop_plate(img, box, pad=0.08):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(int(x1 - pad * w), 0), max(int(y1 - pad * h), 0)
    x2, y2 = min(int(x2 + pad * w), img.shape[1]), min(int(y2 + pad * h), img.shape[0])
    return img[y1:y2, x1:x2]


def rectify(crop, two_lines):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_h = 200 if two_lines else 100
    if contours:
        c = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.3 * crop.shape[0] * crop.shape[1]:
            tl, tr, br, bl = src = order_corners(approx)
            w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
            h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
            out_w = int(out_h * w / h)
            dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                           dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            return cv2.warpPerspective(crop, M, (out_w, out_h))
        (cx, cy), _, angle = cv2.minAreaRect(c)
        if angle > 45:
            angle -= 90
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        crop = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                              borderMode=cv2.BORDER_REPLICATE)
    out_w = int(out_h * crop.shape[1] / crop.shape[0])
    return cv2.resize(crop, (out_w, out_h))


def enhance(plate_bgr):
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


def split_lines(plate_bgr):
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    profile = bw.sum(axis=1).astype(np.float64)
    h = len(profile)
    lo, hi = int(0.3 * h), int(0.7 * h)
    seg = profile[lo:hi]
    low = seg <= seg.min() + 0.05 * seg.max()
    best_len, best_mid, start = 0, int(np.argmin(seg)), None
    for i, v in enumerate(np.append(low, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best_len:
                best_len, best_mid = i - start, (start + i) // 2
            start = None
    cut = lo + best_mid
    return plate_bgr[:cut], plate_bgr[cut:]


PROVINCE_CODES = {
    "11", "12", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "39", "40", "41", "43", "47", "48", "49", "50", "51",
    "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63",
    "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75",
    "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "88",
    "89", "90", "92", "93", "94", "95", "97", "98", "99",
}
LETTERS = "ABCDEFGHKLMNPSTUVXYZ"
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1",
            "Z": "2", "S": "5", "G": "6", "B": "8"}
TO_LETTER = {"0": "D", "2": "Z", "5": "S", "6": "G", "8": "B"}
PLATE_RE = re.compile(rf"^(\d{{2}})([{LETTERS}]{{1,2}}|[{LETTERS}][1-9])(\d{{4,5}})$")


def clean(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper().replace("Đ", "D"))


def split_raw(raw):
    if "|" in raw:
        top, bottom = raw.split("|", 1)
        return top, bottom
    s = clean(raw)
    two_letters = len(s) >= 9 or (len(s) == 8 and s[3].isalpha() and s[3] not in TO_DIGIT)
    k = 4 if two_letters else 3
    return s[:k], s[k:]


def postprocess(top, bottom, valid_codes=PROVINCE_CODES):
    top, bottom = clean(top), clean(bottom)
    if len(top) < 3:
        return top + bottom, False
    code = "".join(TO_DIGIT.get(c, c) for c in top[:2])
    serie = TO_LETTER.get(top[2], top[2]) + top[3:]
    if len(serie) == 2 and serie[1] == "I":
        serie = serie[0] + "1"
    number = "".join(TO_DIGIT.get(c, c) for c in bottom)
    plate = code + serie + number
    ok = PLATE_RE.match(plate) is not None and code in valid_codes
    return plate, ok
