import cv2
import numpy as np
import pytest

from src.appearance import COLOR_NAMES, as_vector, clothing_colors, torso_box

# A person box 100 px wide and 200 px tall, at (100, 50)
BOX = (100, 50, 200, 250)


def person_image(jacket_bgr, background_bgr=(128, 128, 128)) -> np.ndarray:
    """A 360x640 frame where the person's upper body is painted in `jacket_bgr`."""
    image = np.full((360, 640, 3), background_bgr, dtype=np.uint8)
    x1, y1, x2, y2 = torso_box(BOX)
    image[y1:y2, x1:x2] = jacket_bgr
    return image


def test_torso_box_skips_head_legs_and_sides():
    assert torso_box(BOX) == (120, 90, 180, 160)


@pytest.mark.parametrize(
    "jacket_bgr, expected",
    [
        ((19, 170, 0), "green"),      # Gojek-style green (#00AA13)
        ((79, 177, 0), "green"),      # Grab-style green (#00B14F)
        ((0, 77, 238), "orange"),     # Shopee-style orange (#EE4D00)
        ((200, 80, 20), "blue"),
        ((20, 20, 20), "dark"),
        ((235, 235, 235), "muted"),   # white
        ((140, 180, 210), "muted"),   # beige coat: pale orange, but not a brand colour
        ((80, 120, 170), "muted"),    # tan skin
        ((0, 50, 140), "orange"),     # Shopee orange at dusk (darker, still vivid)
        (("hsv", 58, 80, 185), "green"),    # Gojek lime as measured on camera (washed out)
        (("hsv", 70, 55, 210), "green"),    # Grab mint as measured on camera
        (("hsv", 104, 120, 190), "blue"),   # Astro blue as measured on camera
        (("hsv", 5, 160, 230), "red"),      # Shopee orange-red as measured on camera
        (("hsv", 15, 44, 215), "muted"),    # beige uniform as measured on camera
        (("hsv", 10, 135, 170), "muted"),   # strongly tanned skin
    ],
)
def test_jacket_colour_is_named(jacket_bgr, expected):
    if jacket_bgr[0] == "hsv":   # given as OpenCV hue (0-180), saturation, value
        hsv = np.array([[jacket_bgr[1:]]], dtype=np.uint8)
        jacket_bgr = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    colors = clothing_colors(person_image(jacket_bgr), BOX)
    assert max(colors, key=colors.get) == expected
    assert colors[expected] == pytest.approx(1.0)
    assert sum(colors.values()) == pytest.approx(1.0, abs=0.01)


def test_tiny_or_off_frame_box_gives_nothing():
    image = person_image((19, 170, 0))
    assert clothing_colors(image, (10, 10, 12, 14)) == {}
    assert clothing_colors(image, (700, 400, 800, 500)) == {}


def test_vector_follows_color_names_order():
    vector = as_vector({"green": 0.75, "dark": 0.25})
    assert len(vector) == len(COLOR_NAMES)
    assert vector[COLOR_NAMES.index("green")] == 0.75
    assert vector[COLOR_NAMES.index("dark")] == 0.25
    assert as_vector({}) == [0.0] * len(COLOR_NAMES)
