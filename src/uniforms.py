"""A hint about who might be at the gate, from the colour of their top: a courier uniform or not.

Only a hint for the alert message, never part of the decision. Couriers often wear no uniform
(the 9 Nov gas delivery, an Astro rider on 7 Nov), and plenty of people wear the same colours.

Measured on the SD-card footage (share of the upper body in that colour, averaged over a person's frames):
- green, Gojek and Grab: 0.32 on the 9 Nov courier. On 11 Nov, 4 of 347 people reached 0.25.
- orange-red, Shopee: 0.29 on the 7 Nov courier. On 11 Nov, 2 of 347 people.
- blue, Astro: 0.10-0.25 on Astro couriers. Weak: on 11 Nov, 28 of 347 people reached 0.25
  (blue shirts are common).

The colour shares come from src/appearance.py (clothing_colors).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

GOJEK_GRAB = "green top: Gojek or Grab?"
SHOPEE = "orange-red top: Shopee?"
ASTRO = "blue top: Astro?"

# Colour buckets that make up each uniform, in the order to report ties.
UNIFORMS = ((GOJEK_GRAB, ("green",)), (SHOPEE, ("orange", "red")), (ASTRO, ("blue",)))
MIN_SHARE = 0.25   # average share of the top in that colour
MIN_FRAMES = 3     # fewer frames than this: too little to go on


def uniform_score(shares: Sequence[Mapping[str, float]]) -> tuple[float, str | None]:
    """(average share, hint) of the most uniform-looking colour in one person's frames."""
    if len(shares) < MIN_FRAMES:
        return 0.0, None
    best_share, best_hint_ = 0.0, None
    for hint, names in UNIFORMS:
        share = sum(sum(frame.get(name, 0.0) for name in names) for frame in shares) / len(shares)
        if share >= MIN_SHARE and share > best_share:
            best_share, best_hint_ = share, hint
    return best_share, best_hint_


def uniform_hint(shares: Sequence[Mapping[str, float]]) -> str | None:
    """The hint for one person, from the colour shares of each frame they were seen in."""
    return uniform_score(shares)[1]


def best_hint(people: Iterable[Sequence[Mapping[str, float]]]) -> str | None:
    """The hint for whoever looks most like a uniform, among several people."""
    best_share, best = 0.0, None
    for shares in people:
        share, hint = uniform_score(shares)
        if hint is not None and share > best_share:
            best_share, best = share, hint
    return best
