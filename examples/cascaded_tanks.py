"""QSIM's cascaded tanks as a compiled, queryable, *weighted* envisionment.

The classic qualitative-reasoning demo system: constant inflow -> tank A
-> tank B -> drain, with monotonic flow laws (outflow = M+(level)).  Each
tank has a qualitative magnitude {zero, between, full} and a direction of
change {dec, std, inc}.  See tests/test_qsim_tanks.py for the constraint
derivation and hand-checked envisionment sizes.

What compilation adds over classic QSIM:

1. The envisionment (all consistent qualitative states) is the circuit's
   model set — counting it or filtering it by observations is one sweep,
   not a generate-and-test search.
2. Priors on qualitative values turn QSIM's ambiguity explosion into a
   *ranked* list: most plausible qualitative states first, with exact
   posteriors.

Run:  python examples/cascaded_tanks.py
"""

from dnnf import SystemModel, iff

LEVELS = ("zero", "between", "full")
DIRS = ("dec", "std", "inc")

m = SystemModel()

# Magnitudes and directions carry mild plausibility priors: tanks are
# usually somewhere in the middle, change is usually happening.  These
# are ranking weights, not physics.
la = m.mode("levelA", LEVELS, priors=(0.25, 0.5, 0.25))
da = m.mode("dA", DIRS, priors=(0.35, 0.3, 0.35))
lb = m.mode("levelB", LEVELS, priors=(0.25, 0.5, 0.25))
db = m.mode("dB", DIRS, priors=(0.35, 0.3, 0.35))
inflow = m.finite("inflow", ("zero", "plus"))
oa = m.finite("outflowA", ("zero", "plus"))
ob = m.finite("outflowB", ("zero", "plus"))

# Monotonic flow laws with corresponding value at zero.
m.add(iff(oa == "zero", la == "zero"))
m.add(iff(ob == "zero", lb == "zero"))


def sign(p, q, d):  # d = sign(p - q); plus-plus stays ambiguous
    m.add(((p == "zero") & (q == "zero")) >> (d == "std"))
    m.add(((p == "zero") & (q == "plus")) >> (d == "dec"))
    m.add(((p == "plus") & (q == "zero")) >> (d == "inc"))


sign(inflow, oa, da)
sign(oa, ob, db)

# Landmark consistency: can't fall below empty or rise above full.
m.add((la == "zero") >> (da != "dec"))
m.add((la == "full") >> (da != "inc"))
m.add((lb == "zero") >> (db != "dec"))
m.add((lb == "full") >> (db != "inc"))

system = m.compile()
print(f"compiled envisionment: {system.circuit!r}\n")

SCENARIOS = [
    ("faucet on, nothing else known", {"inflow": "plus"}),
    ("faucet on, tank B observed rising", {"inflow": "plus", "dB": "inc"}),
    ("faucet off, tank A still draining", {"inflow": "zero", "dA": "dec"}),
    ("faucet on, system fully settled", {"inflow": "plus", "dA": "std", "dB": "std"}),
]

for label, evidence in SCENARIOS:
    print(f"--- {label}: {evidence}")
    for d in system.map_diagnoses(evidence, k=4):
        s = d.modes
        print(
            f"  p={d.posterior:.3f}  "
            f"A: {s['levelA']:8}{s['dA']:4}  "
            f"B: {s['levelB']:8}{s['dB']}"
        )
    print()

print("design query: P(tank B full AND still rising | faucet on) -> "
      "impossible by landmark consistency:",
      system.log_evidence({"inflow": "plus", "levelB": "full", "dB": "inc"}))
