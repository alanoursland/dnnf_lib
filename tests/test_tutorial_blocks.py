"""Execute every Python code block in the tutorial chapters.

The tutorial's standing promise: copy-paste works. This test enforces
it by extracting ```python fences from each chapter and executing them
in order in a shared per-chapter namespace. Chapters that continue a
prior chapter's build declare a preamble here.
"""

import os
import re
import subprocess
import sys

import pytest

TUTORIAL = os.path.join(os.path.dirname(__file__), "..", "tutorial")

CH3_MODEL = """
from modenexus import SystemModel, iff, fd
m = SystemModel()
pump = m.mode("pump", ("ok", "weak", "dead"), priors=(0.90, 0.07, 0.03))
drip = m.bool("drip")
moist = m.bool("moist")
m.add(iff(drip, pump != "dead"))
m.add(moist >> drip)
system = m.compile()
"""

PREAMBLES = {"04_semirings.md": CH3_MODEL}

CHAPTERS = sorted(
    f for f in os.listdir(TUTORIAL)
    if re.match(r"\d\d_.*\.md$", f)
)


def blocks(path):
    text = open(path, encoding="utf-8").read()
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


@pytest.mark.parametrize("chapter", CHAPTERS)
def test_chapter_blocks_run(chapter):
    path = os.path.join(TUTORIAL, chapter)
    ns: dict = {}
    pre = PREAMBLES.get(chapter)
    if pre:
        exec(pre, ns)
    for code in blocks(path):
        exec(code, ns)  # a failure here means copy-paste is broken


def test_solutions_run():
    sol = os.path.join(TUTORIAL, "solutions")
    for f in sorted(os.listdir(sol)):
        if f.endswith(".py"):
            subprocess.run(
                [sys.executable, os.path.join(sol, f)],
                check=True, timeout=120,
                stdout=subprocess.DEVNULL,
            )
