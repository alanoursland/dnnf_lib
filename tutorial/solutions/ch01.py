"""Chapter 1 exercise solutions."""
from itertools import product


def survivors(constraints, domains):
    names = list(domains)
    out = []
    for values in product(*domains.values()):
        w = dict(zip(names, values))
        if all(c(w) for c in constraints):
            out.append(w)
    return out


DOMAINS = {
    "pump": ("ok", "weak", "dead"),
    "drip": (False, True),
    "moist": (False, True),
    "bulb": ("ok", "burnt_out"),
    "light": (False, True),
}

BASE = [
    lambda w: w["drip"] == (w["pump"] != "dead"),
    lambda w: (not w["moist"]) or w["drip"],
    lambda w: w["light"] == (w["bulb"] == "ok"),
]

# Exercise 1: light subsystem is independent -> 5 * 2 = 10 worlds.
ex1 = survivors(BASE, DOMAINS)
assert len(ex1) == 10, len(ex1)

# Exercise 2: moist => light removes worlds where moist and not light.
ex2 = survivors(BASE + [lambda w: (not w["moist"]) or w["light"]], DOMAINS)
assert len(ex2) == 8, len(ex2)
# Dropped exactly the 2 worlds with moist=True, bulb=burnt_out.

# Exercise 3: count() is survivors() plus len(); reused in ch03.
print("ch01 solutions OK:", len(ex1), len(ex2))
