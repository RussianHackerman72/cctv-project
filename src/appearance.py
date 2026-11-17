"""What a person's clothing looks like, as a small colour breakdown.

Used to test ideas such as "green or orange jacket = probably Gojek, Grab or Shopee"
and "same clothes as the person seen ten minutes ago = probably the same person".
"""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

# Named colour buckets, in a fixed order (the detection log stores fractions in this order).
COLOR_NAMES = ("dark", "muted", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink")

# OpenCV hue runs 0-180 (half of degrees). Upper bounds for each hue bucket:
_HUE_BUCKETS = ((8, "red"), (20, "orange"), (33, "yellow"), (85, "green"), (100, "cyan"),
                (130, "blue"), (150, "purple"), (170, "pink"), (181, "red"))
DARK_VALUE = 50       # below this brightness a pixel is "dark", whatever its hue
# Below these saturations a pixel is "muted": white, grey, beige, skin tones, faded clothes.
# On camera, daylight and video compression wash uniforms out: Gojek lime measures ~80,
# Grab mint ~55, Astro blue ~120 (out of 255). Only the warm hues (red, orange, yellow, pink)
# need a high bar, because skin (up to ~135) and beige live there; Shopee orange-red is ~150.
# Tuned on 98 hand-checked courier and household crops from the SD card (97 right, was 61).
WARM_MIN_SATURATION = 140
COOL_MIN_SATURATION = 50
_WARM_HUES = ((0, 33), (150, 181))  # OpenCV hue ranges of red/orange/yellow and pink/red


def torso_box(box: Sequence[float]) -> tuple[int, int, int, int]:
    """The upper-body part of a person box: where a jacket or uniform shows best.

    Skips the head (top 20%) and legs (below 55%), and the outer 20% at each side,
    which is often background.
    """
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return (int(x1 + 0.2 * width), int(y1 + 0.2 * height), int(x2 - 0.2 * width), int(y1 + 0.55 * height))


def clothing_colors(image: np.ndarray, box: Sequence[float]) -> dict[str, float]:
    """Share of each named colour (0..1) in the person's upper body. Empty if the box is too small."""
    x1, y1, x2, y2 = torso_box(box)
    height, width = image.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return {}

    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    hue, saturation, value = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    dark = value < DARK_VALUE
    warm = np.zeros(len(hue), dtype=bool)
    for low, high in _WARM_HUES:
        warm |= (hue >= low) & (hue < high)
    muted = ~dark & (saturation < np.where(warm, WARM_MIN_SATURATION, COOL_MIN_SATURATION))
    colourful = ~dark & ~muted

    counts = dict.fromkeys(COLOR_NAMES, 0)
    counts["dark"] = int(dark.sum())
    counts["muted"] = int(muted.sum())
    lower = 0
    for upper, name in _HUE_BUCKETS:
        counts[name] += int((colourful & (hue >= lower) & (hue < upper)).sum())
        lower = upper
    total = len(hsv)
    return {name: round(count / total, 3) for name, count in counts.items()}


def as_vector(colors: dict[str, float]) -> list[float]:
    """The colour shares as a list in COLOR_NAMES order (compact for the log)."""
    return [colors.get(name, 0.0) for name in COLOR_NAMES]
