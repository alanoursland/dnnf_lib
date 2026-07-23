"""End-to-end case study for the DX paper: residential solar+battery.

Produces every number quoted in case_study.md. Run:
    python drafts/02/case_study.py
"""
import math, os, random, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from modenexus import ModeTracker, SystemModel, iff
from modenexus.diagnosis import CompiledSystem


def build(battery_priors=(0.94, 0.05, 0.01)):
    m = SystemModel()
    pv = m.mode("pv", ("ok", "degraded", "dead"), priors=(0.95, 0.04, 0.01))
    ctrl = m.mode("ctrl", ("ok", "stuck_off"), priors=(0.99, 0.01))
    batt = m.mode("battery", ("ok", "weak", "dead"), priors=battery_priors)
    inv = m.mode("inverter", ("ok", "fault"), priors=(0.99, 0.01))
    sun = m.bool("sun")
    pv_i = m.bool("pv_i")          # PV current present (internal truth)
    chg = m.bool("chg")            # charge current present (internal truth)
    m.add(iff(pv_i, sun & (pv != "dead")))
    m.add(iff(chg, pv_i & (ctrl == "ok") & (batt != "dead")))
    m.sensor("pv_current", pv_i, false_positive=0.02, false_negative=0.02)
    m.sensor("charging", chg, false_positive=0.02, false_negative=0.02)
    m.sensor("soc_rising", chg & (batt == "ok"),
             false_positive=0.02, false_negative=0.05)
    m.sensor("house_powered", (inv == "ok") & (batt != "dead"),
             false_positive=0.01, false_negative=0.01)
    return m.compile()


TRANS = {"battery": {"ok": {"ok": 0.995, "weak": 0.004, "dead": 0.001},
                     "weak": {"weak": 0.98, "dead": 0.02, "ok": 0.0},
                     "dead": {"dead": 1.0, "ok": 0.0, "weak": 0.0}},
         "pv": {"ok": {"ok": 0.999, "degraded": 0.0009, "dead": 0.0001},
                "degraded": {"degraded": 0.999, "dead": 0.001, "ok": 0.0},
                "dead": {"dead": 1.0, "ok": 0.0, "degraded": 0.0}}}


def main():
    rng = random.Random(2026)

    # --- A. offline compile + artifact -----------------------------------
    t0 = time.time()
    system = build()
    t_compile = time.time() - t0
    path = "/tmp/battery_system.json"
    system.save(path)
    size_kb = os.path.getsize(path) / 1024
    t0 = time.time()
    CompiledSystem.load(path)
    t_load = time.time() - t0
    print(f"A: {system.circuit!r}")
    print(f"A: compile={t_compile*1000:.0f}ms save={size_kb:.1f}KiB load={t_load*1000:.1f}ms")

    # --- B. snapshot diagnosis (ambiguous fault) --------------------------
    ev = {"sun": True, "pv_current": True, "charging": False,
          "house_powered": True}
    t0 = time.time()
    diags = system.map_diagnoses(ev, k=4)
    t_q = time.time() - t0
    print(f"B: query={t_q*1000:.1f}ms")
    for d in diags:
        faults = {k: v for k, v in d.modes.items() if v not in ("ok",)}
        print(f"B: p={d.posterior:.4f} {faults or 'all nominal'}")

    # --- C. active sensing -------------------------------------------------
    voi = system.value_of_information(ev)
    print("C:", [(n, round(v, 4)) for n, v in voi])
    ev2 = dict(ev); ev2["soc_rising"] = False
    print("C after soc_rising=False:",
          [(f"{k}={v}", round(d.posterior, 4)) for d in
           system.map_diagnoses(ev2, k=2)
           for k, v in d.modes.items() if v != "ok"])

    # --- D. one month of tracking, fault injected on day 13 ---------------
    _ = {"battery": {"ok": {"ok": 0.995, "weak": 0.004, "dead": 0.001},
                         "weak": {"weak": 0.98, "dead": 0.02, "ok": 0.0},
                         "dead": {"dead": 1.0, "ok": 0.0, "weak": 0.0}},
             "pv": {"ok": {"ok": 0.999, "degraded": 0.0009, "dead": 0.0001},
                    "degraded": {"degraded": 0.999, "dead": 0.001, "ok": 0.0},
                    "dead": {"dead": 1.0, "ok": 0.0, "degraded": 0.0}}}


    def flip(truth, fp, fn):
        return (rng.random() > fn) if truth else (rng.random() < fp)


    tracker = ModeTracker(build(), TRANS, beam=8)
    detect_day = None
    for day in range(1, 31):
        true_batt = "ok" if day < 13 else "weak"
        chg_truth = True                     # sun, pv ok, ctrl ok all month
        soc_truth = chg_truth and (true_batt == "ok")
        evd = {"sun": True,
               "pv_current": flip(True, 0.02, 0.02),
               "charging": flip(chg_truth, 0.02, 0.02),
               "soc_rising": flip(soc_truth, 0.02, 0.05),
               "house_powered": flip(True, 0.01, 0.01)}
        tracker.step(evd)
        p_weak = tracker.marginals()["battery"]["weak"]
        if detect_day is None and p_weak > 0.5 and day >= 13:
            detect_day = day
        if day in (12, 13, 14, 15, 16, 30):
            print(f"D: day {day:2d} P(battery)= " + " ".join(
                f"{k}:{v:.3f}" for k, v in tracker.marginals()["battery"].items()))
    print(f"D: injected day 13, detected (P>0.5) day {detect_day}")

    # --- E. fleet learning --------------------------------------------------
    true_fleet = build(battery_priors=(0.90, 0.08, 0.02))
    OBS = ["pv_current", "charging", "soc_rising", "house_powered"]
    tele = []
    for _ in range(2000):
        s = true_fleet.sample_state(rng)
        rec = {k: s[k] for k in OBS}
        rec["sun"] = s["sun"]
        tele.append(rec)
    results = []
    for start in (0.33, 0.10, 0.60):
        learner = build(battery_priors=(1 - start - start / 2, start, start / 2))
        hist = learner.fit_priors(tele, names=["battery"])
        spec = learner.circuit.spec
        fv = learner.vars["battery"]
        fitted = [learner._weights[spec.mvlit(fv.fd_var, i)] for i in range(3)]
        results.append((start, fitted, len(hist), hist[0], hist[-1]))
    for start, fitted, iters, l0, l1 in results:
        print(f"E: start weak={start:.2f} -> fitted "
              + "/".join(f"{x:.4f}" for x in fitted)
              + f" iters={iters} ll {l0:.4f}->{l1:.4f}")
    spread = max(r[1][1] for r in results) - min(r[1][1] for r in results)
    print(f"E: multi-start spread on weak: {spread:.2e} (true 0.08)")


if __name__ == "__main__":
    main()
