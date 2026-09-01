# Ops efficiency log (hourly; user directive 2026-09-01)
| hour (UTC) | heavy rungs | heavy median s | light median s | mean landing gap min | idle gaps>5min | action |
|---|---|---|---|---|---|---|
| 14:30-15:30 | 11 | 132.0 | 16.4 | 7.3 | 28/59 | Baseline measured. Profile: model load + census state dominate heavy rungs (see board). Added ops/covcache.py (memoized context covariances, bit-identical on miss). Proposed to Codex for new-script adoption. |
