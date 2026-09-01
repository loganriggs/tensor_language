# Ops efficiency log (hourly; user directive 2026-09-01)
| hour (UTC) | heavy rungs | heavy median s | light median s | mean landing gap min | idle gaps>5min | action |
|---|---|---|---|---|---|---|
| 14:30-15:30 | 11 | 132.0 | 16.4 | 7.3 | 28/59 | Baseline. CORRECTED profile: model load only 4.5s, census import 3.1s, state ~0s (warm caches) — heavy rungs are compute-bound on their own evals, little pipeline fat. Real lever: 28/59 landing gaps >5min are inter-rung COMPOSITION time; queue depth >=2 would hide it behind GPU. Added ops/covcache.py (saves ~10-30s covariance forwards on reuse, bit-identical). |
