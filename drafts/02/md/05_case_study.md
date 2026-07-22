# 5. Case study: a residential solar+battery system

One model, the full lifecycle. Four components (PV array 3 modes,
charge controller 2, battery 3, inverter 2), two internal flow states,
four sensors with 1–5% error rates. All numbers from one seeded,
CPU-only script in the repository.

**Offline.** Compile: 152 nodes, 48 finite-domain variables (sensor
glitch auxiliaries included), **4 ms**. Serialized artifact: **5.2
KiB**, loading in 0.2 ms from a header that states all resource bounds
up front — single-allocation loading, after the flight-software
practice of this architecture's ancestors.

**Diagnosis under ambiguity.** Evidence: sunny, PV current present,
*no charging*, house powered. Exact ranked posteriors in 1.9 ms:
all-nominal-with-a-sensor-glitch 0.603, controller stuck 0.299,
battery weak 0.032, PV degraded 0.025. Two properties are structural,
not tuned: the glitch hypothesis outranks component faults because the
declared noise rates and priors say it should, and the system is
honestly torn rather than falsely decisive.

**Active sensing.** The VOI query ranks unread signals: charge-current
state 0.599 nats expected entropy reduction, SOC trend 0.458, PV
internals 0.043. Reading the top recommendation (SOC not rising)
resolves the ambiguity: controller-stuck 0.740, glitch story
collapsed. One loop turn, two queries.

**A tracked month.** Battery degradation dynamics (ok→weak 0.4%/day,
absorbing dead); thirty days of readings sampled from ground truth
with the modeled error rates; fault injected on day 13:

| day | 12 | 13 (onset) | 14 | 15 | 30 |
|---|---|---|---|---|---|
| P(weak) | 0.000 | 0.074 | **0.618** | 0.969 | 0.995 |

Detection (P > 0.5) on day 14: one day of corroboration after onset,
which is what 5% sensor noise should cost. No thresholds exist to
tune; the latency *is* the posterior arithmetic.

**Fleet learning.** 2,000 snapshot records of sensor readings only —
mode labels never observed — sampled from a fleet with true battery
prior (0.90, 0.08, 0.02). EM from three starting priors (weak-rate
starts 0.10 / 0.33 / 0.60) converges to (0.879, 0.101, 0.020) in 6–18
iterations, multi-start agreement 3×10⁻⁴ (identifiable), likelihood
non-decreasing in every run, recovered rates within sampling error of
truth. The learned priors immediately parameterize every query above.

**The claim this licenses.** One 5 KiB compiled circuit served exact
ranked diagnosis, information-optimal sensor selection, a month of
fault tracking with one-day detection latency, and fleet-scale prior
learning — with zero per-query modeling work after a 4 ms compile.
