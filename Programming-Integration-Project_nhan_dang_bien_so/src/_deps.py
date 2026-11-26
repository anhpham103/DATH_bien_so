import sys
import warnings

try:
    from rectify import crop_plate, enhance, rectify, split_lines  # noqa: F401
    USING_FALLBACK_RECTIFY = False
except ImportError:
    from _guide_fallback import crop_plate, enhance, rectify, split_lines  # noqa: F401
    USING_FALLBACK_RECTIFY = True
    warnings.warn("src/rectify.py not found: using the Part 1 guide fallback.", stacklevel=2)

try:
    from postprocess import clean, postprocess, split_raw  # noqa: F401
    USING_FALLBACK_POSTPROCESS = False
except ImportError:
    from _guide_fallback import clean, postprocess, split_raw  # noqa: F401
    USING_FALLBACK_POSTPROCESS = True
    warnings.warn("src/postprocess.py not found: using the Part 1 guide fallback.", stacklevel=2)


def load_read_baseline():
    for mod in ("read_baseline", "baseline", "ocr_baseline", "rectify"):
        try:
            m = __import__(mod)
        except ImportError:
            continue
        fn = getattr(m, "read_baseline", None)
        if fn is not None:
            return fn
    print("[warning] read_baseline (method A) not found in src/: skipping A.", file=sys.stderr)
    return None
