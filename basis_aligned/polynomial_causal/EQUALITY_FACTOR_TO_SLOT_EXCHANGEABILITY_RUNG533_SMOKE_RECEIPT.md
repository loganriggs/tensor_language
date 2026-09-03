# Rung 533 managed smoke receipt

**Executed:** 2026-09-03 12:56 UTC

**Status:** passed; full registered run authorized

The source-frozen, natural-only smoke completed through the managed GPU runner with exit code 0. It executed exactly
23 model forwards: one direct native forward plus all 11 physical arms in both donor backgrounds. It computed no
cross-entropy outcomes and made no model forward on the `ood_code` role.

```text
native/replay maximum logit difference      0
target factor-product maximum difference    0
factor reconstruction maximum error         4.3824e-14
minimum donor removal RMS                    9.2833
minimum target edit RMS                      2.4824
zero intended edits                          0
support accumulation exercised               true
model forwards                               23 / 23
peak allocated GPU memory                    3,159,521,280 bytes
```

The following hashes bind the passed instrument:

- core: `6ba3a9e5fa4e0fa23c461610451bfc8d65eea909f14fe563131a1441228528fd`
- smoke wrapper: `a43fb9be9836e7655d84d031f7fb6c7e8f27a2e37d74a6f5698a1136a1f77688`
- managed log: `e9b746c56972f4efcc2e2f290db833ccab73c7cfbca2cbf22b5436c65dbea62c`
- preregistration: `d5ed32a7a4268768ed170e4a0fdd282fb49e3be97190c077366e77353a6ad1eb`
- source-freeze commit: `abad68dbe3fea48a85a7751d4e06e04f54b45f86`

The full 2,208-forward run may now open the registered per-document natural/code loss effects. Its thresholds,
factor scales, arm set, corpus halves, and matched controls remain unchanged.
