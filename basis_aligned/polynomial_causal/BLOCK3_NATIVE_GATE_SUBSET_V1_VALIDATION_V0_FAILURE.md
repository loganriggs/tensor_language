# Block-3 native-gate subset validation V0 — implementation failure

V0 was launched from pushed source commit `dbc2ef2e` after independent GO.  It wrote
its authority first, then terminated after 14.82 seconds with no result and no receipt.
The create-only failure artifact is preserved.

The failure occurred on the first native algebraic replay, before a candidate arm was
scored:

```text
RuntimeError: native validation polarization did not replay MLP3
```

This is not evidence for or against the K=256/512 candidate.  The guard compared an
absolute float32 discrepancy with `3e-4`.  A read-only diagnostic on the same first
batch found:

| Quantity | Value |
|---|---:|
| native write maximum absolute value | 5493.65381 |
| replay maximum absolute discrepancy | 0.009765625 |
| maximum discrepancy / native maximum | 1.77762e-6 |
| replay RMS discrepancy | 0.000526577 |
| native write RMS | 600.992615 |
| RMS discrepancy / native RMS | 8.76179e-7 |
| RMS polarization `max_abs(u+v-z)` | 9.53674e-7 |

The absolute guard was therefore scale-invalid: ordinary float32 summation error was
about 33 times its threshold while remaining below two parts per million relative to
the native output.  The correct repair is a prospective V1 namespace with finite,
scale-relative maximum and RMS replay guards.  V0 must not be deleted or overwritten.
