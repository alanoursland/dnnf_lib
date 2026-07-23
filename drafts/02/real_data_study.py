"""Real-telemetry study: UCI Condition Monitoring of Hydraulic Systems
(2205 real load cycles from a hydraulic test rig; labeled component
conditions). Diagnoses the COOLER (3 classes) and PUMP leakage (3
classes) from cycle-mean sensor features via a compiled model whose
mode->bucket constraints and priors come from the TRAIN split only.
Data: /tmp/hydraulic (see case_study.md for download)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from modenexus import SystemModel

D = "/tmp/hydraulic"
def rows(f):
    return [[float(x) for x in line.split()] for line in open(os.path.join(D, f))]
def mean(r): return sum(r) / len(r)

CE = [mean(r) for r in rows("CE.txt")]
CP = [mean(r) for r in rows("CP.txt")]
SE = [mean(r) for r in rows("SE.txt")]
FS1 = [mean(r) for r in rows("FS1.txt")]
prof = rows("profile.txt")
cooler = [int(r[0]) for r in prof]   # 3 / 20 / 100
pump = [int(r[2]) for r in prof]     # 0 / 1 / 2
N = len(prof)
train = [i for i in range(N) if i % 2 == 0]
test = [i for i in range(N) if i % 2 == 1]

K = 8
def buckets(feat):
    vals = sorted(feat[i] for i in train)
    bnds = [vals[int(len(vals) * q / K)] for q in range(1, K)]
    def b(x):
        lo = 0
        for j, t in enumerate(bnds):
            if x >= t: lo = j + 1
        return lo
    return b

FEATS = {"ce": (CE, buckets(CE)), "cp": (CP, buckets(CP)),
         "se": (SE, buckets(SE)), "fs1": (FS1, buckets(FS1))}
TARGETS = {"cooler": (cooler, (3, 20, 100), ["ce", "cp"]),
           "pump": (pump, (0, 1, 2), ["se", "fs1"])}
EPS = 0.02

def build(target):
    labels, classes, feats = TARGETS[target]
    counts = {c: 0 for c in classes}
    for i in train: counts[labels[i]] += 1
    m = SystemModel()
    mode = m.mode(target, tuple(str(c) for c in classes),
                  priors=tuple(counts[c] / len(train) for c in classes))
    support = {}
    for f in feats:
        feat, bf = FEATS[f]
        var = m.finite(f, tuple(range(K)))
        for c in classes:
            sup = sorted({bf(feat[i]) for i in train if labels[i] == c})
            support[(f, c)] = sup
            m.add((mode == str(c)) >> var.in_(sup))
    return m.compile(), support

print("target  acc     mean-post-on-truth  n_test")
for target in TARGETS:
    labels, classes, feats = TARGETS[target]
    sysm, _ = build(target)
    correct, post_sum, bins = 0, 0.0, [[0, 0] for _ in range(5)]
    for i in test:
        ev = {}
        for f in feats:
            feat, bf = FEATS[f]
            one = [EPS] * K
            one[bf(feat[i])] = 1.0
            ev[f] = tuple(one)
        post = sysm.posteriors(ev, names=[target])[target]
        pred = max(post, key=post.get)
        conf = post[pred]
        hit = pred == str(labels[i])
        correct += hit
        post_sum += post[str(labels[i])]
        b = min(int(conf * 5), 4)
        bins[b][0] += hit
        bins[b][1] += 1
    n = len(test)
    print(f"{target:7} {correct/n:.4f}  {post_sum/n:.4f}          {n}")
    print(f"  calibration (conf-bin: acc/n): " + " ".join(
        f"[{0.2*b:.1f}-{0.2*(b+1):.1f}]:{h/max(t,1):.3f}/{t}"
        for b, (h, t) in enumerate(bins) if t))
