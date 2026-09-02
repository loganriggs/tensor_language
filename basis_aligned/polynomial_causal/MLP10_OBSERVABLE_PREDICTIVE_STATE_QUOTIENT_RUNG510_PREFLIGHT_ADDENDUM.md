# Rung510 preflight hash correction

Status: frozen after the first managed smoke stopped in `validate_inputs`, before checkpoint loading and before any
rung510 task, circuit, or intervention outcome.

The preregistration hash was computed as
`678dbef1794719143b7bb366de37e7fc72ffc585275206e49a0bc20bc681c47f` before a trailing blank line was removed by
the repository diff check. The committed preregistration therefore has SHA256
`e344760333af378ea5604c211c259a27d9ff030b60bad8054ca962d465f46055`.

The only byte-level difference is that final blank line. No question, node, circuit family, document, threshold,
prediction, route, computation, price, or implementation behavior changed. The source must pin the committed hash
and this addendum. The failed smoke log remains preserved; the repaired managed smoke may open no scientific effect.
