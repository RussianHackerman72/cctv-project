from src.uniforms import ASTRO, GOJEK_GRAB, SHOPEE, best_hint, uniform_hint


def frames(count, **shares):
    return [dict(shares) for _ in range(count)]


def test_mostly_green_top_hints_gojek_or_grab():
    assert uniform_hint(frames(10, green=0.4, muted=0.6)) == GOJEK_GRAB


def test_orange_and_red_together_hint_shopee():
    assert uniform_hint(frames(10, orange=0.15, red=0.15, dark=0.7)) == SHOPEE


def test_blue_top_hints_astro():
    assert uniform_hint(frames(10, blue=0.3, dark=0.7)) == ASTRO


def test_plain_or_barely_coloured_clothes_give_no_hint():
    assert uniform_hint(frames(10, dark=0.5, muted=0.5)) is None
    assert uniform_hint(frames(10, green=0.1, muted=0.9)) is None


def test_average_over_frames_and_strongest_colour_wins():
    # green in a few frames only (a passing reflection) doesn't count; blue is steady
    shares = frames(2, green=0.9) + frames(8, blue=0.4)   # averages: green 0.18, blue 0.32
    assert uniform_hint(shares) == ASTRO


def test_too_few_frames_give_no_hint():
    assert uniform_hint(frames(2, green=0.9)) is None
    assert uniform_hint([]) is None


def test_best_hint_picks_the_most_uniform_looking_person():
    courier = frames(10, green=0.5)
    neighbour = frames(10, blue=0.3)
    plain = frames(10, muted=1.0)
    assert best_hint([plain, neighbour, courier]) == GOJEK_GRAB
    assert best_hint([plain]) is None
    assert best_hint([]) is None
