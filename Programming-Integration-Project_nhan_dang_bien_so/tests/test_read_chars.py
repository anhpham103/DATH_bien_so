import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from read_chars import sort_chars  # noqa: E402


def row(text, y, x0=10, dx=20, h=30, dy=0.0):
    return [(x0 + i * dx, y + i * dy, h, ch) for i, ch in enumerate(text)]


def shuffled(boxes):
    return boxes[::2] + boxes[1::2]


def test_one_line():
    assert sort_chars(shuffled(row("30A12345", 50)), two_lines=False) == "30A12345"


def test_one_line_tilted_7_degrees():
    boxes = row("51LD12345", 50, dy=20 * 0.123)
    assert sort_chars(shuffled(boxes), two_lines=False, img_h=100) == "51LD12345"


def test_two_lines_motorbike():
    boxes = row("59X1", 50, x0=30, dx=35, h=70) + row("12345", 150, x0=15, dx=30, h=70)
    assert sort_chars(shuffled(boxes), two_lines=True, img_h=200) == "59X1|12345"


def test_two_lines_slightly_tilted():
    boxes = (row("29AB", 50, x0=30, dx=35, h=70, dy=3)
             + row("12345", 150, x0=15, dx=30, h=70, dy=3))
    assert sort_chars(boxes, two_lines=True, img_h=200) == "29AB|12345"


def test_two_lines_only_bottom_detected():
    boxes = row("12345", 150, x0=15, dx=30, h=70, dy=4)
    assert sort_chars(boxes, two_lines=True, img_h=200) == "|12345"


def test_two_lines_only_top_detected():
    assert sort_chars(row("59X1", 50, h=70, dy=3), two_lines=True, img_h=200) == "59X1|"


def test_two_lines_single_char_top():
    assert sort_chars([(40, 45, 70, "5")], two_lines=True, img_h=200) == "5|"


def test_noise_box_ignored():
    boxes = row("59X1", 50, h=70) + row("12345", 150, h=70) + [(90, 100, 10, "8")]
    assert sort_chars(boxes, two_lines=True, img_h=200) == "59X1|12345"


def test_empty():
    assert sort_chars([], two_lines=False) == ""
    assert sort_chars([], two_lines=True) == "|"


def test_output_feeds_postprocess():
    try:
        from postprocess import postprocess, split_raw
    except ImportError:
        from _guide_fallback import postprocess, split_raw
    raw = sort_chars(row("59X1", 50, h=70) + row("I2345", 150, h=70), True, img_h=200)
    assert postprocess(*split_raw(raw)) == ("59X112345", True)
