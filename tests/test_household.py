from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from src.appearance import legs_box, signature, signature_distance, torso_box
from src.household import HouseholdMemory, track_signature

WIB = timezone(timedelta(hours=7))
BOX = (100, 50, 200, 250)

WHITE_SHIRT_BLACK_TROUSERS = (235, 128, 128, 30, 128, 128)
WHITE_SHIRT_BLUE_JEANS = (235, 128, 128, 80, 135, 100)
ASTRO_BLUE_JACKET = (150, 110, 85, 40, 128, 128)


def at(hour: int, minute: int = 0, day: int = 17) -> float:
    return datetime(2026, 11, day, hour, minute, tzinfo=WIB).timestamp()


def dressed(top_bgr, bottom_bgr) -> np.ndarray:
    image = np.full((360, 640, 3), 128, dtype=np.uint8)
    x1, y1, x2, y2 = torso_box(BOX)
    image[y1:y2, x1:x2] = top_bgr
    x1, y1, x2, y2 = legs_box(BOX)
    image[y1:y2, x1:x2] = bottom_bgr
    return image


def test_signature_describes_top_and_bottom():
    sig = signature(dressed((255, 255, 255), (20, 20, 20)), BOX)
    assert len(sig) == 6
    assert sig[0] > 240 and sig[3] < 30                       # light top, dark bottom
    assert sig[1:3] == pytest.approx((128, 128), abs=1)        # both neutral (no colour)
    assert signature(dressed((255, 255, 255), (20, 20, 20)), (10, 10, 14, 14)) is None


def test_distance_cares_less_about_brightness():
    shirt = signature(dressed((200, 200, 200), (20, 20, 20)), BOX)
    same_shirt_in_shade = signature(dressed((140, 140, 140), (15, 15, 15)), BOX)
    green_shirt = signature(dressed((19, 170, 0), (20, 20, 20)), BOX)
    assert signature_distance(shirt, same_shirt_in_shade) < 40
    assert signature_distance(shirt, green_shirt) > 2 * signature_distance(shirt, same_shirt_in_shade)


def test_remember_and_match_the_same_clothes():
    memory = HouseholdMemory(WIB)
    member = memory.remember(WHITE_SHIRT_BLACK_TROUSERS, at(9))
    nearly_the_same = (230, 129, 127, 35, 128, 129)            # same clothes, a bit different light
    assert memory.match(nearly_the_same, at(11)) is member
    assert memory.match(ASTRO_BLUE_JACKET, at(11)) is None


def test_top_alone_is_not_enough():
    memory = HouseholdMemory(WIB)
    memory.remember(WHITE_SHIRT_BLACK_TROUSERS, at(9))
    assert memory.match(WHITE_SHIRT_BLUE_JEANS, at(10)) is None   # same shirt, different trousers


def test_seeing_someone_again_refreshes_instead_of_adding():
    memory = HouseholdMemory(WIB)
    first = memory.remember(WHITE_SHIRT_BLACK_TROUSERS, at(9))
    again = memory.remember((233, 128, 128, 32, 128, 128), at(10))
    assert again is first and first.sightings == 2 and first.last_seen == at(10)
    memory.remember(ASTRO_BLUE_JACKET, at(10))
    assert len(memory.members) == 2


def test_memory_resets_at_four_in_the_morning():
    memory = HouseholdMemory(WIB, reset_hour=4)
    memory.remember(WHITE_SHIRT_BLACK_TROUSERS, at(20))
    assert memory.match(WHITE_SHIRT_BLACK_TROUSERS, at(2, day=18)) is not None   # home late: still today
    assert memory.match(WHITE_SHIRT_BLACK_TROUSERS, at(5, day=18)) is None       # next morning: forgotten
    assert memory.members == []


def test_track_signature_ignores_odd_frames():
    frames = [WHITE_SHIRT_BLACK_TROUSERS] * 4 + [ASTRO_BLUE_JACKET, None]   # one bad frame, one too small
    assert track_signature(frames) == WHITE_SHIRT_BLACK_TROUSERS
    assert track_signature([None, None]) is None


def test_real_frame_signature_uses_lab():
    # The L*a*b* conversion must agree with OpenCV's for a plain colour.
    bgr = (19, 170, 0)
    expected = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0]
    sig = signature(dressed(bgr, bgr), BOX)
    assert sig[:3] == pytest.approx(tuple(float(v) for v in expected))
