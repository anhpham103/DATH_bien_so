import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHARS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
         "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


def git_commit():
    try:
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_two_lines(value):
    v = str(value).strip().lower()
    return v in {"plate_2line", "2line", "2", "true", "1.0"} or v.endswith("2line")


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
